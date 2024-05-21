from typing import Optional, Callable, NamedTuple

from beartype import beartype
from einops import repeat, pack, rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as ff


class MemoryState(NamedTuple):
    cnt: torch.Tensor
    mem: torch.Tensor


class EpisodicMemory(nn.Module):  # defined as a module for the signature to be similar to a LSTM
    """Episodic Memory implementation"""

    @beartype
    def __init__(self, memory_size: int, capacity: int, device: torch.device):
        # `memory_size`: size of the vectors to be stored
        # `capacity`: max number of memories (verctors) to store
        self.memory_size = memory_size
        self.capacity = capacity
        self.device = device

    def forward(self, inputs: torch.Tensor, prev_state: MemoryState):
        # explaination of the signature:
        # `inputs`: the new input to be stored in memory
        # `prev_state`: a tuple containing the previous counter and memories
        assert inputs.device == self.device, "wrong device"
        inputs = inputs.detach()  # stop gradient (note: everything else is differentiable)
        counter, memories = prev_state

        # compute the position to update in the memory using modulo operation
        counter_mod = torch.fmod(counter, self.capacity)
        # create a one-hot vector to select the memory slot to be updated
        slot_selector = ff.one_hot(counter_mod, num_classes=self.capacity).unsqueeze(-1).float()
        # update the memory slots
        # only the selected slot is updated with the new input
        memories = memories * (1 - slot_selector) + (slot_selector * inputs.unsqueeze(1))
        # increment the counter to keep track of the memory slot to be updated next
        counter = counter + 1

        # return the updated memories and the current, updated version of `prev_state`
        return (memories, MemoryState(counter, memories))  # parens for legibility

    def initial_state(self, batch_size):
        counter = torch.zeros((batch_size, 1), dtype=torch.long)
        memories = torch.zeros((batch_size, self.capacity, self.memory_size))
        return counter, memories


class RingBuffer(object):

    @beartype
    def __init__(self, maxlen: int, shape: tuple[int, ...], device: torch.device):
        """Ring buffer implementation"""
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = torch.zeros((maxlen, *shape), dtype=torch.float32, device=device)

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
        # important: idxs is a numpy array, and start and maxlen are ints
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


class ReplayBuffer(object):

    @beartype
    def __init__(self,
                 generator: torch.Generator,
                 capacity: int,
                 erb_shapes: dict[str, tuple[int, ...]],
                 device: torch.device):
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
               patcher: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
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
            # turn the list defaultdict into a dict of np.ndarray
            trns = {k: pack(v, "* d")[0] for k, v in la_batch.items()}
            for k, v in trns.items():
                assert v.device == self.device, f"v for {k=} is on wrong device"
        else:
            trns = self.get_trns(idxs)
            if patcher is not None:
                # patch the rewards
                trns["rews"] = patcher(trns["obs0"], trns["acs"], trns["obs1"])
        return trns

    @beartype
    def append(self, trn: dict[str, np.ndarray],
               *,
               rew_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        """Add a transition to the replay buffer"""
        assert {k for k in self.ring_buffers if k != "rews"} == set(trn.keys()), "key mismatch"
        for k in self.ring_buffers:
            if k == "rews":
                continue
            if not isinstance(trn[k], np.ndarray):
                raise TypeError(k)
            new_tensor = torch.Tensor(trn[k]).to(self.device)
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

    @beartype
    def __repr__(self) -> str:
        shapes = "|".join([f"[{k}:{s}]" for k, s in self.erb_shapes.items()])
        return f"ReplayBuffer(capacity={self.capacity}, shapes={shapes})"

    @beartype
    @property
    def latest_entry_idx(self) -> int:
        return self.ring_buffers["obs0"].latest_entry_idx  # could pick any other key

    @beartype
    @property
    def num_entries(self) -> int:
        return len(self.ring_buffers["obs0"])  # could pick any other key
