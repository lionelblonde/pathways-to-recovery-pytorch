from beartype import beartype
import torch


class ActionNoise(object):

    @beartype
    def reset(self):  # exists even if useless for non-temporally correlated noise
        pass


class NormalActionNoise(ActionNoise):

    @beartype
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, generator: torch.Generator):
        """Additive action space Gaussian noise"""
        assert isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor)
        self.mu = mu
        self.sigma = sigma
        self.device = self.mu.device  # grap the device we are on (assumed sigma and mu on same)
        self.rng = generator

    @beartype
    def generate(self):
        return torch.normal(self.mu, self.sigma, generator=self.rng).to(self.device)

    @beartype
    def __repr__(self):
        return f"NormalAcNoise(mu={self.mu}, sigma={self.sigma})"
