from typing import Optional, Callable

from beartype import beartype
from einops import repeat, pack, unpack, rearrange
import numpy as np
import torch

from helpers import logger


class RingBuffer(object):

    @beartype
    def __init__(self, maxlen: int, shape: tuple[int, ...], device: torch.device):
        """Ring buffer impl"""
        self.maxlen = maxlen
        self.device = device
        self.start = 0
        self.length = 0
        self.data = torch.zeros((maxlen, *shape), dtype=torch.float32, device=self.device)

    @beartype
    def __len__(self):
        return self.length

    @beartype
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise KeyError
        return self.data[(self.start + idx) % self.maxlen]

    @beartype
    def get_batch(self, idxs: torch.Tensor) -> torch.Tensor:
        # important: idxs is a tensor, and start and maxlen are ints
        return self.data[(self.start + idxs) % self.maxlen]

    @beartype
    def append(self, *, v: torch.Tensor):
        if self.length < self.maxlen:
            # we have space, simply increase the length
            self.length += 1
            self.data[(self.start + self.length - 1) % self.maxlen] = v
        elif self.length == self.maxlen:
            # no space, remove the first item
            self.start = (self.start + 1) % self.maxlen
            self.data[(self.start + self.length - 1) % self.maxlen] = v
        else:
            # this should never happen
            raise RuntimeError

    @beartype
    @property
    def latest_entry_idx(self) -> int:
        return (self.start + self.length - 1) % self.maxlen


class TrajectStore(object):

    @beartype
    def __init__(self,
                 generator: torch.Generator,
                 capacity: int,
                 em_mxlen: int,
                 erb_shapes: dict[str, tuple[int, ...]],
                 *,
                 state_only: bool,  # state-only discriminator
                 lstm_mode: bool,
                 device: torch.device):
        """Replay buffer impl"""
        self.rng = generator
        self.capacity = capacity
        self.em_mxlen = em_mxlen
        self.erb_shapes = erb_shapes
        self.state_only = state_only
        # remove unused key to save on memory  TODO(lionel): clean this up
        if not lstm_mode:
            self.erb_shapes.pop("dones1", None)
        if self.state_only:
            self.erb_shapes.pop("acs", None)
            self.erb_shapes.pop("acs_orig", None)
        elif not lstm_mode:
            self.erb_shapes.pop("obs1", None)
            self.erb_shapes.pop("obs1_orig", None)

        self.pdd_shapes = {
            k: (self.em_mxlen, *s) for k, s in self.erb_shapes.items()}
        self.device = device
        self.ring_buffers = {}
        for k in self.pdd_shapes:
            self.ring_buffers.update({
                k: RingBuffer(self.capacity, self.pdd_shapes[k], self.device)})

    @beartype
    def get_trjs(self, idxs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collect a batch from indices"""
        trjs = {}
        for k, v in self.ring_buffers.items():
            trjs[k] = v.get_batch(idxs)
        return trjs

    @beartype
    def sample(self,
               batch_size: int,
               *,
               patcher: Optional[Callable[[torch.Tensor,
                                           Optional[torch.Tensor],
                                           Optional[torch.Tensor]],
                                          torch.Tensor]],
        ) -> Optional[dict[str, torch.Tensor]]:
        """Sample transitions uniformly from the replay buffer"""
        if self.num_entries == 0:
            logger.warn("trajectory store still empty; skipping for now")
            return None
        idxs = torch.randint(
            low=0,
            high=self.num_entries,
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
        )
        trjs = self.get_trjs(idxs)
        if patcher is not None:
            # patch the rewards
            with torch.no_grad():
                # get the shape of the reward tensor by packing it
                _, rews_ps = pack([(rew := trjs["rews"])], "* d")
                # build a mask to identify the zero-padding
                mask = rew.clone().detach()  # security detach
                mask[mask > 0.] = 1
                mask[mask < 0.] = 1
                mask[mask == 0.] = 0
                # pack the inputs of the reward patcher
                obs0, _ = pack([trjs["obs0"]], "* d")
                if self.state_only:
                    acs = None
                    obs1, _ = pack([trjs["obs1"]], "* d")
                else:
                    acs, _ = pack([trjs["acs"]], "* d")
                    obs1 = None
                # obtain the new reward
                rews = patcher(obs0, acs, obs1)
                # get the reward in back in its nominal shape
                # by unpacking it with the packing size from earlier
                [trjs["rews"]] = unpack(rews, rews_ps, "* d")
                # apply mask to remove rewards computed on zero-padding
                assert mask.size() == trjs["rews"].size(), "wrong shape"
                trjs["rews"] *= mask
        return trjs

    @beartype
    def append(self, trj: list[dict[str, torch.Tensor]]):
        new_trj = self.rearrange_and_zeropad(trj)
        assert "rews" in new_trj, "ensure transitions are always appended to the RB before the TS"
        # it is necessary for each transition to first be added to the RB since its append method
        # adds the reward to the reward ring buffer and then returns the augmented transition for
        # the orchestrator to append to the currently ongoing trajectory, eventually stored here.
        for k in self.ring_buffers:
            self.ring_buffers[k].append(v=new_trj[k])

    @beartype
    def rearrange_and_zeropad(self, trj: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Utility func that transforms a list of transitions (dictionaries)
        into a dictionary of an aggreagated and zero-padded tensor.
        """
        tmp_trj = {k: [] for k in self.pdd_shapes}  # could use defaultdict but keys known
        pdd_trj = {}
        for e in trj:
            for k, v in e.items():
                if k not in self.pdd_shapes:
                    continue
                tmp_trj[k].append(rearrange(v, "d -> 1 d"))
        for k, v in tmp_trj.items():
            tmp_trj_k, _ = pack(v, "* d")
            assert tmp_trj_k.device == self.device, "wrong device"
            # this assertion implicly also asserts that v was a list of tensors, all on self.device
            pdd_trj[k] = torch.zeros(self.pdd_shapes[k], dtype=torch.float32, device=self.device)
            pdd_trj[k][:tmp_trj_k.size(0), :] = tmp_trj_k
        return pdd_trj

    @beartype
    @property
    def num_entries(self) -> int:
        return len(self.ring_buffers["obs0"])  # could pick any other key

    @beartype
    @property
    def how_filled(self) -> str:
        num = f"{self.num_entries:,}".rjust(10)
        denomi = f"{self.capacity:,}".rjust(10)
        return f"{num} / {denomi}"


class ReplayBuffer(object):

    @beartype
    def __init__(self,
                 generator: torch.Generator,
                 capacity: int,
                 erb_shapes: dict[str, tuple[int, ...]],
                 device: torch.device):
        """Replay buffer impl"""
        self.rng = generator
        self.capacity = capacity
        self.erb_shapes = erb_shapes
        self.device = device
        self.ring_buffers = {
            k: RingBuffer(self.capacity, s, self.device) for k, s in self.erb_shapes.items()}

    @beartype
    def get_trns(self, idxs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collect a batch from indices"""
        trns = {}
        for k, v in self.ring_buffers.items():
            trns[k] = v.get_batch(idxs)
        return trns

    @beartype
    def discount(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute gamma-discounted sum"""
        c = x.size(0)
        reps = repeat(x, "k 1 -> c k", c=c)  # note: k in einstein notation is c
        mats = [
            (gamma ** (c - j)) *
                torch.diagflat(torch.ones(j, device=self.device), offset=(c - j))
            for j in reversed(range(1, c + 1))]
        mats, _ = pack(mats, "* h w")
        out = rearrange(torch.sum(reps * torch.sum(mats, dim=0), dim=1), "k -> k 1")
        assert out.size() == x.size()
        return out[0]  # would be simpler to just compute the 1st elt, but only used in n-step rets

    @beartype
    def sample(self,
               batch_size: int,
               *,
               patcher: Optional[Callable[[torch.Tensor,
                                           Optional[torch.Tensor],
                                           Optional[torch.Tensor]],
                                          torch.Tensor]],
               n_step_returns: bool = False,
               lookahead: Optional[int] = None,
               gamma: Optional[float] = None,
        ) -> dict[str, torch.Tensor]:
        """Sample transitions uniformly from the replay buffer"""
        idxs = torch.randint(
            low=0,
            high=self.num_entries,
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
        )
        if n_step_returns:
            assert lookahead is not None and gamma is not None
            assert 0 <= gamma <= 1
            # initiate the batch of transition data necessary to perform n-step TD backups
            la_keys = list(self.erb_shapes.keys())
            la_keys.extend(["td_len", "obs1_td1"])
            la_batch = {k: [] for k in la_keys}  # could use defaultdict but keys known
            # iterate over the indices to deploy the n-step backup for each
            for _idx in idxs:
                idx = _idx.item()
                # create indexes of transitions in lookahead
                # of lengths max `lookahead` following sampled one
                la_end_idx = min(idx + lookahead, self.num_entries) - 1
                assert isinstance(idx, int) and isinstance(la_end_idx, int)

                # the following are all tensors
                la_idxs = torch.arange(idx, la_end_idx + 1, device=self.device)
                # collect the batch for the lookahead rollout indices
                la_trns = self.get_trns(la_idxs)
                if patcher is not None:
                    with torch.no_grad():
                        # patch the rewards
                        la_trns["rews"] = patcher(la_trns["obs0"], la_trns["acs"], la_trns["obs1"])
                # only keep data from the current episode,
                # drop everything after episode reset, if any
                dones = la_trns["dones1"]

                # the following are all ints
                term_idx = 1.0

                ep_end_idx = int(
                    idx + torch.argmax(dones.float()).item() if term_idx in dones else la_end_idx)
                # doc: if there are multiple maximal values in a reduced row
                # then the indices of the first maximal value are returned.

                la_is_trimmed = 0 if ep_end_idx == la_end_idx else 1
                # compute lookahead length
                td_len = ep_end_idx - idx + 1

                # trim down the lookahead transitions
                la_rews = la_trns["rews"][:td_len]
                # compute discounted cumulative reward
                la_discounted_sum_n_rews = self.discount(la_rews, gamma)
                # populate the batch for this n-step TD backup
                la_batch["obs0"].append(la_trns["obs0"][0])
                la_batch["obs1"].append(la_trns["obs1"][td_len - 1])
                la_batch["acs"].append(la_trns["acs"][0])
                la_batch["rews"].append(la_discounted_sum_n_rews)
                la_batch["dones1"].append(torch.Tensor([la_is_trimmed]).to(self.device))
                la_batch["td_len"].append(torch.Tensor([td_len]).to(self.device))
                # add the first next state too: needed in state-only discriminator
                la_batch["obs1_td1"].append(la_trns["obs1"][0])
                # when dealing with absorbing states
                if "obs0_orig" in la_keys:
                    la_batch["obs0_orig"].append(la_trns["obs0_orig"][0])
                if "obs1_orig" in la_keys:
                    la_batch["obs1_orig"].append(la_trns["obs1_orig"][td_len - 1])
                if "acs_orig" in la_keys:
                    la_batch["acs_orig"].append(la_trns["acs_orig"][0])
            # turn the list dict into a dict of np.ndarray
            trns = {k: pack(v, "* d")[0] for k, v in la_batch.items()}
            for k, v in trns.items():
                assert v.device == self.device, f"v for {k=} is on wrong device"
        else:
            trns = self.get_trns(idxs)
            if patcher is not None:
                # patch the rewards
                with torch.no_grad():
                    trns["rews"] = patcher(trns["obs0"], trns["acs"], trns["obs1"])
        return trns

    @beartype
    def append(self, trn: dict[str, np.ndarray],
               *,
               rew_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> dict[str, torch.Tensor]:
        """Add a transition to the replay buffer.
        Returns the latest transition (i.e. the one just added) for subsequent modules to use
        since we transform into tensor and create the reward here already.
        """
        assert {k for k in self.ring_buffers if k != "rews"} == set(trn.keys()), "key mismatch"
        for k in self.ring_buffers:
            if k == "rews":
                continue
            if not isinstance(trn[k], np.ndarray):
                raise TypeError(k)
            new_tensor = torch.Tensor(trn[k]).to(self.device)  # cap T tensor to force FloatTensor
            self.ring_buffers[k].append(v=new_tensor)
        # also add the synthetic reward to the replay buffer
        # note: by this point everything is already as a tensor on device
        rew = rew_func(
            *(rearrange(x, "d -> 1 d") for x in [
                self.ring_buffers["obs0"][self.latest_entry_idx],
                self.ring_buffers["acs"][self.latest_entry_idx],
                self.ring_buffers["obs1"][self.latest_entry_idx],
            ]),
        )
        self.ring_buffers["rews"].append(v=rew)
        # sanity-check that all the ring buffers are at the same stage
        last_idxs = [li := v.latest_entry_idx for v in self.ring_buffers.values()]
        assert all(ll == li for ll in last_idxs), "not all equal"
        return self.latest_entry  # returned for subsequent modules to use

    @beartype
    @property
    def latest_entry(self) -> dict[str, torch.Tensor]:
        return self.get_trns(torch.tensor(self.latest_entry_idx))

    @beartype
    @property
    def latest_entry_idx(self) -> int:
        return self.ring_buffers["obs0"].latest_entry_idx  # could pick any other key

    @beartype
    @property
    def num_entries(self) -> int:
        return len(self.ring_buffers["obs0"])  # could pick any other key

    @beartype
    @property
    def how_filled(self) -> str:
        num = f"{self.num_entries:,}".rjust(10)
        denomi = f"{self.capacity:,}".rjust(10)
        return f"{num} / {denomi}"

    @beartype
    def __repr__(self) -> str:
        shapes = "|".join([f"[{k}:{s}]" for k, s in self.erb_shapes.items()])
        return f"ReplayBuffer(capacity={self.capacity}, shapes={shapes})"
