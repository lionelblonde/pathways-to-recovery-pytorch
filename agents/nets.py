import math
from contextlib import nullcontext
from collections import OrderedDict
from typing import Callable

from beartype import beartype
from einops import rearrange, pack
import torch
from torch import nn
from torch.nn import functional as ff
from torch.nn.utils.parametrizations import spectral_norm

from helpers import logger
from helpers.normalizer import RunningMoments


STANDARDIZED_OB_CLAMPS = [-5., 5.]
ARCTANH_EPS = 1e-8
SAC_LOG_STD_BOUNDS = [-20., 2.]


@beartype
def log_module_info(model: nn.Module):

    def _fmt(n) -> str:
        if n // 10 ** 6 > 0:
            out = str(round(n / 10 ** 6, 2)) + " M"
        elif n // 10 ** 3:
            out = str(round(n / 10 ** 3, 2)) + " k"
        else:
            out = str(n)
        return out

    logger.info("logging model specs")
    logger.info(model)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f"total trainable params: {_fmt(num_params)}.")


@beartype
def init(constant_bias: float = 0.) -> Callable[[nn.Module], None]:
    """Perform orthogonal initialization"""

    def _init(m: nn.Module) -> None:

        if (isinstance(m, (nn.Conv2d, nn.Linear))):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


@beartype
def snwrap(*, use_sn: bool = False) -> Callable[[nn.Module], nn.Module]:
    """Spectral normalization wrapper"""

    def _snwrap(m: nn.Module) -> nn.Module:
        assert isinstance(m, nn.Linear)
        if use_sn:
            return spectral_norm(m)
        return m

    return _snwrap


@beartype
def arctanh(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the arctanh function.
    Can be very numerically unstable, hence the clamping.
    """
    out = torch.atanh(x)
    if out.sum().isfinite():
        # note: a sum() is often faster than a any() or all()
        # there might be edge cases but at worst we use the clamped version and get notified
        return out
    logger.warn("using a numerically stable (and clamped) arctanh")
    one_plus_x = (1 + x).clamp(
        min=ARCTANH_EPS)
    one_minus_x = (1 - x).clamp(
        min=ARCTANH_EPS)
    return 0.5 * torch.log(one_plus_x / one_minus_x)
    # equivalent to 0.5 * (x.log1p() - (-x).log1p()) but with NaN-proof clamping
    # torch.atanh(x) is numerically unstable here
    # note: with both of the methods above, we get NaN at the first iteration


class NormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @beartype
    @staticmethod
    def logp(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @beartype
    @staticmethod
    def sample(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        # re-parametrization trick
        eps = torch.empty(mean.size()).to(mean.device).normal_(generator=generator)
        eps.requires_grad = False
        return mean + (std * eps)

    @beartype
    @staticmethod
    def mode(mean: torch.Tensor) -> torch.Tensor:
        return mean


class TanhNormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @beartype
    @staticmethod
    def logp(x: torch.Tensor,
             mean: torch.Tensor,
             std: torch.Tensor,
             *,
             x_scale: float) -> torch.Tensor:
        # we need to assemble the logp of a sample which comes from a Gaussian sample
        # after being mapped through a tanh. This needs a change of variable.
        # See appendix C of the SAC paper for an explanation of this change of variable.
        x_ = arctanh(x / x_scale)
        logp1 = NormalToolkit.logp(x_, mean, std)
        logp2 = 2. * (math.log(2.) - x_ - ff.softplus(-2. * x_))
        logp2 = logp2.sum(dim=-1, keepdim=True)
        # trick for numerical stability from:
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return logp1 - logp2

    @beartype
    @staticmethod
    def sample(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        sample = NormalToolkit.sample(mean, std, generator)
        return torch.tanh(sample)

    @beartype
    @staticmethod
    def mode(mean: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mean)


# models

class Discriminator(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_size: int,
                 rms_obs: RunningMoments,
                 *,
                 wrap_absorb: bool,
                 d_batch_norm: bool,
                 spectral_norm: bool,
                 state_only: bool):
        super().__init__()
        self.wrap_absorb = wrap_absorb
        self.d_batch_norm = d_batch_norm
        self.spectral_norm = spectral_norm
        self.state_only = state_only
        if self.d_batch_norm:
            self.rms_obs = rms_obs
        else:
            logger.info("rms_obs not used by discriminator")

        # wrap absorbing ajustments
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        if self.wrap_absorb:
            ob_dim += 1
            ac_dim += 1

        apply_sn = snwrap(use_sn=self.spectral_norm)  # spectral normalization

        # define the input dimension
        in_dim = ob_dim
        if self.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim

        # assemble the layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(in_dim, hid_size))),
                ("nl", nn.LeakyReLU(negative_slope=0.1)),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", apply_sn(nn.Linear(hid_size, hid_size))),
                ("nl", nn.LeakyReLU(negative_slope=0.1)),
            ]))),
        ]))
        self.d_head = nn.Linear(hid_size, 1)
        # perform initialization
        self.fc_stack.apply(init())
        self.d_head.apply(init())

    @beartype
    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        if self.d_batch_norm:
            # apply normalization
            if self.wrap_absorb:
                # normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(*STANDARDIZED_OB_CLAMPS)
                input_a, _ = pack([input_a_, rearrange(input_a[:, -1], "b -> b 1")], "b *")
                if self.state_only:
                    # normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(*STANDARDIZED_OB_CLAMPS)
                    input_b, _ = pack([input_b_, rearrange(input_b[:, -1], "b -> b 1")], "b *")
            else:
                # normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(*STANDARDIZED_OB_CLAMPS)
                if self.state_only:
                    # normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.state_only:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)
        x, _ = pack([input_a, input_b], "b *")  # concatenate along last dim
        x = self.fc_stack(x)
        return self.d_head(x)  # no sigmoid here


# TD3 (diffs: original uses no layer norm, no obs normalizer)

class Actor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: RunningMoments,
                 max_ac: float,
                 *,
                 layer_norm: bool):
        super().__init__()
        self.ob_dim = ob_shape[-1]  # needed in child class
        self.ac_dim = ac_shape[-1]  # needed in child class
        self.rms_obs = rms_obs
        self.max_ac = max_ac
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.ob_dim, hid_dims[0])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0])),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1])),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], self.ac_dim)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def act(self, ob: torch.Tensor) -> torch.Tensor:
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        return float(self.max_ac) * torch.tanh(self.head(x))


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: RunningMoments,
                 *,
                 layer_norm: bool,
                 use_c51: bool,
                 c51_num_atoms: int,
                 use_qr: bool,
                 num_tau: int):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.layer_norm = layer_norm
        self.use_c51 = use_c51
        self.c51_num_atoms = c51_num_atoms
        self.use_qr = use_qr
        self.num_tau = num_tau

        # number of head
        if self.use_c51:
            num_heads = self.c51_num_atoms
        elif self.use_qr:
            num_heads = self.num_tau
        else:
            num_heads = 1

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim + ac_dim, hid_dims[0])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0])),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1])),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], num_heads)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def forward(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x, _ = pack([ob, ac], "b *")
        x = self.fc_stack(x)
        x = self.head(x)
        if self.use_c51:
            # return a categorical distribution
            x = ff.log_softmax(x, dim=1).exp()
        return x


# SAC

class TanhGaussActor(Actor):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: RunningMoments,
                 max_ac: float,
                 *,
                 generator: torch.Generator,
                 state_dependent_std: bool,
                 layer_norm: bool):
        super().__init__(ob_shape, ac_shape, hid_dims, rms_obs, max_ac, layer_norm=layer_norm)
        self.rng = generator
        self.state_dependent_std = state_dependent_std
        # overwrite head
        if self.state_dependent_std:
            self.head = nn.Linear(hid_dims[1], 2 * self.ac_dim)
        else:
            self.head = nn.Linear(hid_dims[1], self.ac_dim)
            self.ac_logstd_head = nn.Parameter(torch.full((self.ac_dim,), math.log(0.6)))
        # perform initialization (since head written over)
        self.head.apply(init())
        # no need to init the Parameter type object

    @beartype
    def logp(self, ob: torch.Tensor, ac: torch.Tensor, max_ac: float) -> torch.Tensor:
        out = self.mean_std(ob)
        return TanhNormalToolkit.logp(ac, *out, x_scale=max_ac)  # mean, std

    @beartype
    def sample(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            out = self.mean_std(ob)
            return float(self.max_ac) * TanhNormalToolkit.sample(*out, generator=self.rng)

    @beartype
    def mode(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            mean, _ = self.mean_std(ob)
            return float(self.max_ac) * TanhNormalToolkit.mode(mean)

    @beartype
    @staticmethod
    def bound_log_std(log_std: torch.Tensor) -> torch.Tensor:
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = SAC_LOG_STD_BOUNDS
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    @beartype
    def mean_std(self, ob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        if self.state_dependent_std:
            ac_mean, ac_log_std = self.head(x).chunk(2, dim=-1)
        else:
            ac_mean = self.head(x)
            ac_log_std = self.ac_logstd_head.expand_as(ac_mean)
        ac_log_std = self.bound_log_std(ac_log_std)
        ac_std = ac_log_std.exp()
        return ac_mean, ac_std
