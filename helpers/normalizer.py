from typing import Any

from beartype import beartype
import torch


class RunningMoments(object):

    EPS: float = 1e-8

    @beartype
    def __init__(self, shape: tuple[int, ...], device: torch.device):
        """Maintain running statistics across workers leveraging Chan's method"""
        self.count: float = 1e-4  # haxx to avoid any division by zero
        # initialize mean and var with float64 precision (objectively more accurate)
        dim = shape[-1]  # TODO(lionel): best we can do?
        self.mean = torch.zeros((dim,), dtype=torch.float32, device=device)
        self.var = torch.ones((dim,), dtype=torch.float32, device=device)
        self.device = device

    @beartype
    def update(self, x: torch.Tensor):
        """Update running statistics using the new batch's statistics"""
        assert x.device == self.device, "must: same device"
        self.update_moments(x.mean(dim=0), x.var(dim=0), x.size(0))

    @beartype
    def update_moments(self,
                       batch_mean: torch.Tensor,
                       batch_var: torch.Tensor,
                       batch_count: int):
        """ Implementation of Chan's method to compute and maintain mean and variance estimates
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        # at this stage, all tensors are on the same device (batch and self stats)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        # compute new mean
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        # compute new var
        new_var = m_2 / tot_count
        # compute new count
        new_count = tot_count
        # update moments
        self.mean = new_mean
        self.var = new_var
        assert self.mean.device == self.device and self.var.device == self.device, "device issue"
        self.count = new_count

    @beartype
    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return (x - self.mean) / torch.sqrt(self.var + self.EPS)

    @beartype
    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return (x * self.var.sqrt()) + self.mean

    @beartype
    def divide_by_std(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor) and x.device == self.device, "must: same device"
        return x / torch.sqrt(self.var + self.EPS)

    @beartype
    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

    @beartype
    def state_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
