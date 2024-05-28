from collections import defaultdict
from pathlib import Path

from typing import Union
import h5py
from beartype import beartype
from einops import rearrange, pack
import numpy as np
import torch

from helpers import logger


@beartype
def save_dict_h5py(data: dict[str, np.ndarray], fname: Union[str, Path]):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, "w") as hf:
        for key in data:
            hf.create_dataset(key, data=data[key])


@beartype
def load_dict_h5py(fname: Union[str, Path],
    ) -> tuple[dict[str, np.ndarray],
               dict[str, Union[np.floating, np.integer]]]:
    """Restore dictionary containing numpy arrays from h5py file."""
    data, stts = {}, {}
    with h5py.File(fname, "r") as hf:
        for key in hf:
            dset = hf[key]
            if isinstance(dset, h5py.Dataset):
                dk = dset[()]
                assert isinstance(dk, (np.ndarray, np.floating, np.integer)), f"{type(dk) = }"
                if isinstance(dk, (np.floating, np.integer)):
                    stts[key] = dk
                else:  # last option: np.ndarray
                    data[key] = dk
            else:
                raise TypeError(f"dset for key {key} has wrong type")
    return data, stts


class DemoDataset(object):

    @beartype
    def __init__(self,
                 generator: torch.Generator,
                 np_rng: np.random.Generator,
                 device: torch.device,
                 expert_path: str,
                 num_demos: int,
                 max_ep_steps: int,
                 *,
                 wrap_absorb: bool):
        logger.info("creating dataset")
        logger.info(f"spec::expert path: {expert_path}")
        logger.info(f"spec::num_demos: {num_demos}")
        self.rng = generator
        self.np_rng = np_rng
        self.device = device
        self.num_demos = num_demos
        self.stts = defaultdict(list)
        self.data = defaultdict(list)
        logger.info("::::loading demos")

        if wrap_absorb:
            # we added one before if wrapping the absorbing transitions
            max_ep_steps -= 1

        # go over the demos, sorted in alphabetical order
        for i, f in enumerate(sorted(Path(expert_path).glob("*.h5"))):

            # if the desired number of demos has been aggregated, leave
            if i == num_demos:
                break

            # log the location of the loaded demo
            logger.info(f"[DEMO DATASET]::demo #{str(i).zfill(3)} loaded from: {f}")

            # load the demo from the file
            tmp_data, tmp_stts = load_dict_h5py(f)

            # remove undesirable keys (at least in this application)
            assert tmp_data["dones1"][-1], "by construction"  # making sure every ep ends with done
            try:
                tmp_data.pop("pix_obs0")
                tmp_data.pop("pix_obs1")
            except KeyError:
                logger.warn("keys to filter out are not in")
                logger.info("so no need to remove them")
            else:
                logger.info("keys properly removed")

            # get episode statistics
            try:
                self.ep_len = tmp_stts.pop("ep_lens")  # return and delete key
                self.ep_ret = tmp_stts.pop("ep_env_rets")  # return and delete key
            except KeyError as ke:
                logger.error("required keys are missing")
                raise KeyError from ke

            keys_ = ["ep_len", "ep_ret"]
            for a, b in zip(keys_, [k.ljust(20, "-") for k in keys_]):
                logger.info(f"[DEMO DATASET]::{b}{a}")
                self.stts[a].append(getattr(self, a))

            # determine if terminal because of timeout or real termination
            terminal = self.ep_len != max_ep_steps
            logger.info(f"is the trj terminated? {self.ep_len}/{max_ep_steps} -> {terminal}")

            # subsample trajectory: trajectories are not contiguous sequences
            sub_rate = 20  # N=20 in the original GAIL paper
            start = self.np_rng.integers(low=0, high=sub_rate)
            indices = [start + (i * sub_rate) for i in range(int(self.ep_len) // sub_rate)]
            ep_len = len(indices)  # overwrite ep_len

            padd_substr = "subsample".ljust(15, "-")
            substr = f"{ep_len}(sub_rate={sub_rate})"
            logger.info(f"[DEMO DATASET]::{padd_substr}{substr}")
            for k in tmp_data:
                tdk = tmp_data[k]
                tmp_data[k] = tdk[indices]

            # collect the demo content
            if wrap_absorb:

                # add the originals in case we need them
                self.data["obs0_orig"].append(tmp_data["obs0"])
                self.data["acs_orig"].append(tmp_data["acs"])
                self.data["obs1_orig"].append(tmp_data["obs1"])

                # treat differently depending on whether we have a terminal transition
                if tmp_data["dones1"][-1] and terminal:
                    # if the last transition of the subsampled trajectory is done, then it must be
                    # the very last transition of the episode, and testing whether it is
                    # a true terminal state is given by "terminal" determined above
                    logger.info("[DEMO DATASET]::wrapping with absorbing transition")

                    # wrap transition with an absorbing state
                    # add lines of zeros: one for each ob of obs0 and each ac of acs
                    obs0, _ = pack([tmp_data["obs0"], np.zeros((ep_len, 1))], "b *")
                    acs, _ = pack([tmp_data["acs"], np.zeros((ep_len, 1))], "b *")
                    # add a line of zeros except the last one: one zero for each ob of obs1
                    # except the last one (terminal ob) to which a one is concatenated
                    zeros_and_one, _ = pack([np.zeros((ep_len - 1, 1)), np.ones((1, 1))], "* d")
                    obs1, _ = pack([tmp_data["obs1"], zeros_and_one], "b *")

                    # add absorbing transition: done by concatenating a row to previous matrices
                    obs0_last_row = np.append(np.zeros_like(tmp_data["obs0"][-1]), 1)
                    obs0_last_row = rearrange(obs0_last_row, "b -> 1 b")  # replaces np.expand_dims
                    obs0, _ = pack([obs0, obs0_last_row], "* d")
                    acs_last_row = np.append(np.zeros_like(tmp_data["acs"][-1]), 1)
                    acs_last_row = rearrange(acs_last_row, "b -> 1 b")  # replaces np.expand_dims
                    acs, _ = pack([acs, acs_last_row], "* d")
                    obs1_last_row = np.append(np.zeros_like(tmp_data["obs1"][-1]), 1)
                    obs1_last_row = rearrange(obs1_last_row, "b -> 1 b")  # replaces np.expand_dims
                    obs1, _ = pack([obs1, obs1_last_row], "* d")
                else:
                    # the last transition of the subsampled trajectory is not a terminal one,
                    # so we just add a zero to each row of obs0, acs, and obs1 to indicate that
                    obs0, _ = pack([tmp_data["obs0"], np.zeros((ep_len, 1))], "b *")
                    acs, _ = pack([tmp_data["acs"], np.zeros((ep_len, 1))], "b *")
                    obs1, _ = pack([tmp_data["obs1"], np.zeros((ep_len, 1))], "b *")

                # collect the extracted and processed contents
                self.data["obs0"].append(obs0)
                self.data["acs"].append(acs)
                self.data["obs1"].append(obs1)

            else:
                # if not wrapping the absorbing states, the originals is what we use
                self.data["obs0"].append(tmp_data["obs0"])
                self.data["acs"].append(tmp_data["acs"])
                self.data["obs1"].append(tmp_data["obs1"])

        # transform structures into arrays
        self.np_stts, self.np_data = {}, {}
        for k, v in self.stts.items():
            self.np_stts[k] = np.array(v)
        logger.info("[DEMO DATASET]::keys extracted:")
        for k, v in self.data.items():
            self.np_data[k], _ = (test := pack(v, "* d"))
            logger.info(f"        {k=} shape={test[0].shape}")

        self.stts = self.np_stts
        # but transform the contents of data into torch tensors
        self.data = {k: torch.Tensor(v).to(self.device) for k, v in self.np_data.items()}

        lens = self.stts["ep_len"]
        rets = self.stts["ep_ret"]
        logger.info(f"[DEMO DATASET]::got {len(self)} transitions, from {self.num_demos} eps")
        logger.info(f"[DEMO DATASET]::episodic length: {np.mean(lens)}({np.std(lens)})")
        logger.info(f"[DEMO DATASET]::episodic return: {np.mean(rets)}({np.std(rets)})")

    @beartype
    def sample(self, batch_size: int, keys: list[str]) -> dict[str, torch.Tensor]:
        idxs = torch.randint(
            low=0,
            high=len(self),
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
        )
        samples = {}
        for k, v in self.data.items():
            if k not in keys:
                continue
            samples[k] = v[idxs]
        return samples

    @beartype
    def __repr__(self) -> str:
        return f"DemoDataset(num_demos={self.num_demos})"

    @beartype
    def __len__(self) -> int:
        return len(next(iter(self.data.values())))
