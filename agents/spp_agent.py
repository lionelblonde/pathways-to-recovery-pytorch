from pathlib import Path
from typing import Optional, Union
from collections import defaultdict

from beartype import beartype
from omegaconf import DictConfig
from einops import repeat, rearrange, pack
import wandb
import numpy as np
import torch
from torch.nn.utils import clip_grad as cg

import torch.nn.functional as ff
from torch import autograd

from helpers import logger
from helpers.normalizer import RunningMoments
from helpers.dataset import DemoDataset
from helpers.math_util import huber_quant_reg_loss
from agents.nets import log_module_info, Actor, TanhGaussActor, Critic, Discriminator
from agents.ac_noise import NormalActionNoise
from agents.memory import ReplayBuffer


class SPPAgent(object):

    MAGIC_FACTOR: float = 0.1
    TRAIN_METRICS_WANDB_LOG_FREQ: int = 100

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 max_ac: float,
                 device: torch.device,
                 hps: DictConfig,
                 actr_noise_rng: torch.Generator,
                 expert_dataset: Optional[DemoDataset],
                 replay_buffers: Optional[list[ReplayBuffer]]):

        self.ob_shape, self.ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]
        self.max_ac = max_ac

        self.device = device

        assert isinstance(hps, DictConfig)
        self.hps = hps

        # define the out grads used by grad pen here not to always redefine them
        self.go = rearrange(torch.ones(
            self.hps.batch_size * self.hps.num_env), "b -> b 1").to(self.device)

        self.timesteps_so_far = 0
        self.actr_updates_so_far = 0
        self.crit_updates_so_far = 0
        self.disc_updates_so_far = 0

        assert self.hps.lookahead > 1 or not self.hps.n_step_returns
        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")
        assert 0. <= float(self.hps.d_label_smooth) <= 1.

        # demo dataset
        self.expert_dataset = expert_dataset

        # replay buffer
        self.replay_buffers = replay_buffers

        # critic
        # either C51 xor QR xor clipped double (for either TD3 or SAC)
        assert sum([self.hps.use_c51, self.hps.use_qr, self.hps.clipped_double]) == 1
        # if not there will be an error thrown when computing losses
        if self.hps.use_c51:
            assert not self.hps.clipped_double
            c51_supp_range = (self.hps.c51_vmin,
                              self.hps.c51_vmax,
                              self.hps.c51_num_atoms)
            self.c51_supp = torch.linspace(*c51_supp_range).to(self.device)
            self.c51_delta = ((self.hps.c51_vmax - self.hps.c51_vmin) /
                              (self.hps.c51_num_atoms - 1))
        elif self.hps.use_qr:
            assert not self.hps.clipped_double
            qr_cum_density = np.array([((2 * i) + 1) / (2.0 * self.hps.num_tau)
                                       for i in range(self.hps.num_tau)])
            qr_cum_density = torch.Tensor(qr_cum_density).to(self.device)
            self.qr_cum_density = qr_cum_density.view(1, 1, -1, 1).expand(self.hps.batch_size,
                                                                          self.hps.num_tau,
                                                                          self.hps.num_tau,
                                                                          -1).to(self.device)

        # setup action noise
        self.ac_noise = None
        if self.hps.prefer_td3_over_sac:
            self.ac_noise = NormalActionNoise(
                mu=torch.zeros(self.ac_shape).to(self.device),
                sigma=float(self.hps.normal_noise_std) * torch.ones(self.ac_shape).to(self.device),
                generator=actr_noise_rng,
            )  # spherical/isotropic additive Normal(0., 0.1) action noise (we set the std via cfg)
            logger.info(f"{self.ac_noise} configured")

        # create observation normalizer that maintains running statistics
        self.rms_obs = RunningMoments(shape=self.ob_shape, device=self.device)

        assert not (self.hps.use_c51 and self.hps.ret_norm)
        assert not (self.hps.use_qr and self.hps.ret_norm)
        if self.hps.ret_norm:
            # create return normalizer that maintains running statistics
            self.rms_ret = RunningMoments(shape=(1,), device=self.device)

        # create online and target nets

        self.hid_dims = (400, 300) if self.hps.prefer_td3_over_sac else (256, 256)

        actr_net_args = [self.ob_shape, self.ac_shape, self.hid_dims, self.rms_obs, self.max_ac]
        actr_net_kwargs = {"layer_norm": self.hps.layer_norm}
        if not self.hps.prefer_td3_over_sac:
            actr_net_kwargs.update({
                "generator": actr_noise_rng, "state_dependent_std": self.hps.state_dependent_std})
            actr_module = TanhGaussActor
        else:
            actr_module = Actor
        self.actr = actr_module(*actr_net_args, **actr_net_kwargs).to(self.device)
        if self.hps.prefer_td3_over_sac:
            # using TD3 (SAC does not use a target actor)
            self.targ_actr = Actor(*actr_net_args, **actr_net_kwargs).to(self.device)

        crit_net_args = [self.ob_shape, self.ac_shape, self.hid_dims, self.rms_obs]
        crit_net_kwargs_keys = ["layer_norm", "use_c51", "c51_num_atoms", "use_qr", "num_tau"]
        crit_net_kwargs = {k: getattr(self.hps, k) for k in crit_net_kwargs_keys}
        self.crit = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
        self.targ_crit = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)

        # initilize the target nets
        if self.hps.prefer_td3_over_sac:
            # using TD3 (SAC does not use a target actor)
            self.targ_actr.load_state_dict(self.actr.state_dict())
        self.targ_crit.load_state_dict(self.crit.state_dict())

        if self.hps.clipped_double:
            # create second ("twin") critic and target critic
            # ref: TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
            self.targ_twin = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        # set up the optimizers
        self.actr_opt = torch.optim.Adam(self.actr.parameters(), lr=self.hps.actor_lr)
        self.crit_opt = torch.optim.Adam(
            self.crit.parameters(),
            lr=self.hps.critic_lr,
            weight_decay=self.hps.wd_scale,
        )
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(
                self.twin.parameters(),
                lr=self.hps.critic_lr,
                weight_decay=self.hps.wd_scale,
            )

        self.log_alpha = torch.tensor(self.hps.alpha_init).log().to(self.device)
        # the previous line is here for the alpha property to always exist
        if not self.hps.prefer_td3_over_sac:
            # create learnable Lagrangian multiplier
            # common trick: learn log(alpha) instead of alpha directly
            self.log_alpha.requires_grad = True
            self.targ_ent = -self.ac_shape[-1]  # set target entropy to -|A|
            self.loga_opt = torch.optim.Adam(
                [self.log_alpha],
                lr=self.hps.log_alpha_lr,
            )

        # set up lr scheduler
        self.actr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actr_opt,
            (t_max := ((self.MAGIC_FACTOR * self.hps.num_timesteps * self.hps.actor_update_delay) /
                       (self.hps.training_steps_per_iter * self.hps.g_steps))),
        )
        logger.info(f"{t_max = }")

        if self.expert_dataset is not None:
            # create discriminator and its optimizer
            disc_net_args = [self.ob_shape, self.ac_shape, self.hps.d_hid_size, self.rms_obs]
            disc_net_kwargs_keys = ["wrap_absorb", "d_batch_norm", "spectral_norm", "state_only"]
            disc_net_kwargs = {k: getattr(self.hps, k) for k in disc_net_kwargs_keys}
            self.disc = Discriminator(*disc_net_args, **disc_net_kwargs).to(self.device)
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.hps.d_lr)

        log_module_info(self.actr)
        log_module_info(self.crit)
        if self.hps.clipped_double:
            log_module_info(self.twin)
        log_module_info(self.disc)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @beartype
    def norm_rets(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize if return normalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.standardize(x)
        return x

    @beartype
    def denorm_rets(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize if return denormalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.destandardize(x)
        return x

    @beartype
    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of transitions from the replay buffer"""
        assert self.replay_buffers is not None

        # create patcher if needed
        @beartype
        def _patcher(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return self.get_syn_rew(x, y, z)

        patcher = _patcher if self.hps.historical_patching else None

        # get a batch of transitions from replay buffer
        batches = defaultdict(list)
        for rb in self.replay_buffers:
            batch = rb.sample(
                self.hps.batch_size,
                patcher=patcher,
                n_step_returns=self.hps.n_step_returns,
                lookahead=self.hps.lookahead,
                gamma=self.hps.gamma,
            )
            for k, v in batch.items():
                batches[k].append(v)
        out = {}
        for k, v in batches.items():
            out[k], _ = pack(v, "* d")  # equiv to: rearrange(v, "n b d -> (n b) d")
        return out

    @beartype
    def predict(self, ob: np.ndarray, *, apply_noise: bool) -> np.ndarray:
        """Predict an action, with or without perturbation"""
        # create tensor from the state (`require_grad=False` by default)
        ob_tensor = torch.Tensor(ob).to(self.device)
        if self.ac_noise is not None:
            # using TD3
            assert self.hps.prefer_td3_over_sac
            # predict an action
            ac_tensor = self.actr.act(ob_tensor)
            # if desired, add noise to the predicted action
            if apply_noise:
                # apply additive action noise once the action has been predicted,
                # in combination with parameter noise, or not.
                ac_tensor += self.ac_noise.generate()
                logger.info("applied noise")
        else:
            # using SAC
            assert not self.hps.prefer_td3_over_sac
            if apply_noise:
                ac_tensor = self.actr.sample(
                    ob_tensor, stop_grad=True)
                logger.info("applied noise")
            else:
                ac_tensor = self.actr.mode(
                    ob_tensor, stop_grad=True)
        # place on cpu as a numpy array
        ac = ac_tensor.numpy(force=True)
        # clip the action to fit within the range from the environment
        ac.clip(-self.max_ac, self.max_ac)
        return ac

    @beartype
    def compute_losses(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       next_state: torch.Tensor,
                       next_action: torch.Tensor,
                       reward: torch.Tensor,
                       done: torch.Tensor,
                       td_len: torch.Tensor) -> tuple[torch.Tensor,
                                                      torch.Tensor,
                                                      Optional[torch.Tensor],
                                                      Optional[torch.Tensor]]:
        """Compute the critic and actor losses"""

        twin_loss = None
        loga_loss = None

        if self.hps.prefer_td3_over_sac:
            # using TD3
            action_from_actr = self.actr.act(state)
            log_prob = None  # quiets down the type checker
        else:
            # using SAC
            action_from_actr = self.actr.sample(state, stop_grad=False)
            log_prob = self.actr.logp(state, action_from_actr, self.max_ac)
            # here, there are two gradient pathways: the reparam trick makes the sampling process
            # differentiable (pathwise derivative), and logp is a score function gradient estimator
            # intuition: aren't they competing and therefore messing up with each other's compute
            # graphs? to understand what happens, write down the closed form of the Normal's logp
            # (or see this formula in nets.py) and replace x by mean + eps * std
            # it shows that with both of these gradient pathways, the mean receives no gradient
            # only the std receives some (they cancel out)
            # moreover, if only the std receives gradient, we can expect subpar results if this std
            # is state independent
            # this can be observed here, and has been noted in openai/spinningup
            # in native PyTorch, it is equivalent to using `log_prob` on a sample from `rsample`
            # note also that detaching the action in the logp (using `sample`, and not `rsample`)
            # yields to poor results, showing how allowing for non-zero gradients for the mean
            # can have a destructive effect, and that is why SAC does not allow them to flow.

        if self.hps.use_c51:

            # compute qz estimate
            z = self.crit(state, action)  # shape: [batch_size, c51_num_atoms]
            z = rearrange(z, "b n -> b n 1")  # equivalent to unsqueeze(-1)
            z.clamp(0.01, 0.99)

            # compute target qz estimate
            z_prime = self.targ_crit(next_state, next_action)
            # `z_prime` is shape [batch_size, c51_num_atoms]
            z_prime.clamp(0.01, 0.99)

            reward = repeat(reward, "b 1 -> b n", n=self.hps.c51_num_atoms)
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done))
            c51_supp = rearrange(self.c51_supp, "n -> 1 n", n=self.hps.c51_num_atoms)
            # `reward` has shape [batch_size, c51_num_atoms]
            # `gamma_mask` is shape [batch_size, 1]
            # `c51_supp` is shape [1, c51_num_atoms]
            # the prod of the two latter broadcast into [batch_size, c51_num_atoms]
            # `tz` is shape [batch_size, c51_num_atoms]
            tz = reward + (gamma_mask * c51_supp)
            tz = tz.clamp(self.hps.c51_vmin, self.hps.c51_vmax)

            b = (tz - self.hps.c51_vmin) / self.c51_delta
            l = b.floor().long()  # noqa
            u = b.ceil().long()
            targ_z = torch.zeros_like(z_prime)  # shape: [batch_size, c51_num_atoms]
            z_prime_l = z_prime * (u + (l == u).float() - b)
            z_prime_u = z_prime * (b - l.float())
            for i in range(targ_z.size(0)):
                targ_z[i].index_add_(0, l[i], z_prime_l[i])
                targ_z[i].index_add_(0, u[i], z_prime_u[i])

            # reshape target to be of shape [batch_size, c51_num_atoms, 1]
            targ_z = rearrange(targ_z, "b n -> b n 1")  # equivalent to unsqueeze(-1)
            # z and targ_z now have the same shape

            # critic loss
            ce_losses = -(targ_z.detach() * torch.log(z)).sum(dim=1)
            crit_loss = ce_losses.mean()

            # actor loss
            actr_loss = -self.crit(state, action_from_actr)  # [batch_size, num_atoms]
            # we matmul by the transpose of rearranged `c51_supp` of shape [1, c51_num_atoms]
            actr_loss = actr_loss.matmul(c51_supp.t())  # resulting shape: [batch_size, 1]

        elif self.hps.use_qr:

            # compute qz estimate, shape: [batch_size, num_tau]
            z = self.crit(state, action)
            z = rearrange(z, "b n -> b n 1")  # equivalent to unsqueeze(-1)

            # compute target qz estimate, shape: [batch_size, num_tau]
            z_prime = self.targ_crit(next_state, next_action)

            # `reward` has shape [batch_size, 1]
            # reshape rewards to be of shape [batch_size x num_tau, 1]
            reward = repeat(reward, "b 1 -> (b n) 1", n=self.hps.num_tau)
            # reshape product of gamma and mask to be of shape [batch_size x num_tau, 1]
            gamma_mask = repeat(
                (self.hps.gamma ** td_len) * (1 - done), "b 1 -> (b n) 1", n=self.hps.num_tau)
            # like mask and reward, make `z_prime` of shape [batch_size x num_tau, 1]
            z_prime = rearrange(z_prime, "b n -> (b n) 1")
            # assemble the 3 elements of shape [batch_size x num_tau, 1]
            targ_z = reward + (gamma_mask * z_prime)
            # reshape target to be of shape [batch_size, num_tau, 1]
            targ_z = rearrange(targ_z, "(b n) 1 -> b n 1",
                               b=self.hps.batch_size, n=self.hps.num_tau)

            # critic loss
            # compute the TD error loss
            # note: online version has shape [batch_size, num_tau, 1],
            # while the target version has shape [batch_size, num_tau, 1].
            td_errors = targ_z[:, :, None, :].detach() - z[:, None, :, :]  # broadcasting
            # the resulting shape is [batch_size, num_tau, num_tau, 1]

            # assemble the Huber quantile regression loss
            huber_td_errors = huber_quant_reg_loss(td_errors, self.qr_cum_density)
            # the resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            # sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N" in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # resulting shape is [batch_size, 1]
            # average across the minibatch
            crit_loss = crit_loss.mean()

            # actor loss
            actr_loss = -self.crit(state, action_from_actr)

        else:

            # compute qz estimates
            q = self.denorm_rets(self.crit(state, action))
            twin_q = self.denorm_rets(self.twin(state, action))

            # compute target qz estimate and same for twin
            q_prime = self.targ_crit(next_state, next_action)
            twin_q_prime = self.targ_twin(next_state, next_action)
            if self.hps.bcq_style_targ_mix:
                # use BCQ style of target mixing: soft minimum
                q_prime = (0.75 * torch.min(q_prime, twin_q_prime) +
                           0.25 * torch.max(q_prime, twin_q_prime))
            else:
                # use TD3 style of target mixing: hard minimum
                q_prime = torch.min(q_prime, twin_q_prime)

            q_prime = self.denorm_rets(q_prime)

            if not self.hps.prefer_td3_over_sac:  # only for SAC
                # add the causal entropy regularization term
                next_log_prob = self.actr.logp(next_state, next_action, self.max_ac)
                q_prime -= self.alpha.detach() * next_log_prob

            # assemble the Bellman target
            targ_q = (reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime)

            targ_q = targ_q.detach()

            targ_q = self.norm_rets(targ_q)
            if self.hps.ret_norm:
                # update the running stats
                self.rms_ret.update(targ_q)

            # critic and twin losses
            crit_loss = ff.smooth_l1_loss(q, targ_q)  # Huber loss for both here and below
            twin_loss = ff.smooth_l1_loss(twin_q, targ_q)  # overwrites the None initially set

            # actor loss
            if self.hps.prefer_td3_over_sac:
                actr_loss = -self.crit(state, action_from_actr)
            else:
                actr_loss = (self.alpha.detach() * log_prob) - torch.min(
                    self.crit(state, action_from_actr), self.twin(state, action_from_actr))
                if not actr_loss.mean().isfinite():
                    raise ValueError("NaNs: numerically unstable arctanh func")

            if (not self.hps.prefer_td3_over_sac) and self.hps.learnable_alpha:
                assert log_prob is not None
                loga_loss = (self.log_alpha * (-log_prob - self.targ_ent).detach()).mean()
                # we use log(alpha) here but would work with alpha as well

        actr_loss = actr_loss.mean()

        return actr_loss, crit_loss, twin_loss, loga_loss

    @beartype
    @staticmethod
    def send_to_dash(metrics: dict[str, Union[np.float64, np.int64, np.ndarray]],
                     *,
                     step_metric: int,
                     glob: str):
        """Send the metrics to the wandb dashboard"""

        wandb_dict = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                assert v.ndim == 0
            assert hasattr(v, "item"), "in case of API changes"
            wandb_dict[f"{glob}/{k}"] = v.item()

        wandb_dict[f"{glob}/step"] = step_metric

        wandb.log(wandb_dict)
        logger.info(f"logged this to wandb: {wandb_dict}")

    @beartype
    def update_actr_crit(self, batch: dict[str, torch.Tensor], *, update_actr: bool):
        """Update the critic and the actor"""

        with torch.no_grad():
            # define inputs
            if self.hps.wrap_absorb:
                state = batch["obs0_orig"]
                action = batch["acs_orig"]
                next_state = batch["obs1_orig"]
            else:
                state = batch["obs0"]
                action = batch["acs"]
                next_state = batch["obs1"]
            reward = batch["rews"]
            done = batch["dones1"].float()
            td_len = batch["td_len"] if self.hps.n_step_returns else torch.ones_like(done)

        # update the observation normalizer
        self.rms_obs.update(state)

        # compute target action
        if self.hps.prefer_td3_over_sac:
            # using TD3
            if self.hps.targ_actor_smoothing:
                n_ = action.clone().detach().normal_(0., self.hps.td3_std).to(self.device)
                n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
                next_action = (
                    self.targ_actr.act(next_state) + n_).clamp(-self.max_ac, self.max_ac)
            else:
                next_action = self.targ_actr.act(next_state)
        else:
            # using SAC
            next_action = self.actr.sample(next_state, stop_grad=True)

        # compute critic and actor losses
        actr_loss, crit_loss, twin_loss, loga_loss = self.compute_losses(
            state, action, next_state, next_action, reward, done, td_len,
        )  # if `twin_loss` is None at this point, it means we are using C51 or QR

        if update_actr or (not self.hps.prefer_td3_over_sac):
            # choice: for SAC, always update the actor and log(alpha)

            # update actor
            self.actr_opt.zero_grad()
            actr_loss.backward()
            if self.hps.clip_norm > 0:
                cg.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
            self.actr_opt.step()

            self.actr_updates_so_far += 1

            # update lr
            self.actr_sched.step()

            if loga_loss is not None:
                # update log(alpha), and therefore alpha
                assert (not self.hps.prefer_td3_over_sac) and self.hps.learnable_alpha
                self.loga_opt.zero_grad()
                loga_loss.backward()
                self.loga_opt.step()

            if self.actr_updates_so_far % self.TRAIN_METRICS_WANDB_LOG_FREQ == 0:
                wandb_dict = {
                    "actr_loss": actr_loss.numpy(force=True),
                    "actr_lr": np.array(self.actr_sched.get_last_lr()[0]).astype(np.float64),
                }
                if not self.hps.prefer_td3_over_sac:
                    wandb_dict.update({"alpha": self.alpha.numpy(force=True)})
                self.send_to_dash(
                    wandb_dict,
                    step_metric=self.actr_updates_so_far,
                    glob="train_actr",
                )

        # update critic
        self.crit_opt.zero_grad()
        crit_loss.backward()
        self.crit_opt.step()
        if twin_loss is not None:
            # update twin
            self.twin_opt.zero_grad()
            twin_loss.backward()
            self.twin_opt.step()

        self.crit_updates_so_far += 1

        if self.crit_updates_so_far % self.TRAIN_METRICS_WANDB_LOG_FREQ == 0:
            wandb_dict = {"crit_loss": crit_loss.numpy(force=True)}
            if twin_loss is not None:
                wandb_dict.update({"twin_loss": twin_loss.numpy(force=True)})
            self.send_to_dash(
                wandb_dict,
                step_metric=self.crit_updates_so_far,
                glob="train_crit",
            )

        # update target nets
        if (self.hps.prefer_td3_over_sac or
            self.crit_updates_so_far % self.hps.crit_targ_update_freq == 0):
            self.update_target_net()

    @beartype
    def update_disc(self, batch: dict[str, torch.Tensor]):

        assert self.expert_dataset is not None

        # filter out the keys we want
        d_keys = ["obs0"]
        if self.hps.state_only:
            if self.hps.n_step_returns:
                d_keys.append("obs1_td1")
            else:
                d_keys.append("obs1")
        else:
            d_keys.append("acs")

        with torch.no_grad():

            # filter out unwanted keys and tensor-ify
            p_batch = {k: batch[k] for k in d_keys}

            # get a batch of samples from the expert dataset
            e_batches = defaultdict(list)
            for _ in range(self.hps.num_env):
                e_batch = self.expert_dataset.sample(self.hps.batch_size, keys=d_keys)
                for k, v in e_batch.items():
                    e_batches[k].append(v)
            e_batch = {}
            for k, v in e_batches.items():
                e_batch[k], _ = pack(v, "* d")  # equiv to: rearrange(v, "n b d -> (n b) d")

            # define inputs
            p_input_a = p_batch["obs0"]
            e_input_a = e_batch["obs0"]
            if self.hps.state_only:
                p_input_b = p_batch["obs1_td1"] if self.hps.n_step_returns else p_batch["obs1"]
                e_input_b = e_batch["obs1"]
            else:
                p_input_b = p_batch["acs"]
                e_input_b = e_batch["acs"]

        # compute scores
        p_scores = self.disc(p_input_a, p_input_b)
        e_scores = self.disc(e_input_a, e_input_b)

        # entropy loss
        scores, _ = pack([p_scores, e_scores], "* d")  # concat along the batch dim, d is 1
        entropy = ff.binary_cross_entropy_with_logits(
            input=scores,
            target=torch.sigmoid(scores),
        )
        entropy_loss = -self.hps.ent_reg_scale * entropy

        # create labels
        fake_labels = 0. * torch.ones_like(p_scores)
        real_labels = 1. * torch.ones_like(e_scores)

        # apply label smoothing to real labels (one-sided label smoothing)
        if (offset := self.hps.d_label_smooth) != 0:
            real_labels.uniform_(1. - offset, 1. + offset)
            logger.info("applied one-sided label smoothing")

        # binary classification
        p_loss = ff.binary_cross_entropy_with_logits(
            input=p_scores,
            target=fake_labels,
        )
        e_loss = ff.binary_cross_entropy_with_logits(
            input=e_scores,
            target=real_labels,
        )
        p_e_loss = p_loss + e_loss

        # sum losses
        disc_loss = p_e_loss + entropy_loss

        grad_pen = None
        if self.hps.grad_pen:
            # add gradient penalty to loss
            grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
            disc_loss += (self.hps.grad_pen_scale * grad_pen)

        # update parameters
        self.disc_opt.zero_grad()
        disc_loss.backward()
        self.disc_opt.step()

        self.disc_updates_so_far += 1  # count this as one update

        if self.disc_updates_so_far % self.TRAIN_METRICS_WANDB_LOG_FREQ == 0:
            wandb_dict = {
                "entropy_loss": entropy_loss.numpy(force=True),
                "p_loss": p_loss.numpy(force=True),
                "e_loss": e_loss.numpy(force=True),
                "p_e_loss": p_e_loss.numpy(force=True),
            }
            if self.hps.grad_pen and grad_pen is not None:
                wandb_dict.update({"grad_pen": grad_pen.numpy(force=True)})
            self.send_to_dash(wandb_dict, step_metric=self.disc_updates_so_far, glob="train_disc")

    @beartype
    def grad_pen(self,
                 p_input_a: torch.Tensor,
                 p_input_b: torch.Tensor,
                 e_input_a: torch.Tensor,
                 e_input_b: torch.Tensor) -> torch.Tensor:
        """Compute the gradient penalty"""

        # assemble interpolated inputs
        eps_a = torch.rand(p_input_a.size(0), 1).to(self.device)
        eps_b = torch.rand(p_input_b.size(0), 1).to(self.device)
        input_a_i = eps_a * p_input_a + ((1. - eps_a) * e_input_a)
        input_b_i = eps_b * p_input_b + ((1. - eps_b) * e_input_b)

        input_a_i.requires_grad = True
        input_b_i.requires_grad = True

        score = self.disc(input_a_i, input_b_i)

        # get the gradient of this operation w.r.t. its inputs
        grads = autograd.grad(
            inputs=[input_a_i, input_b_i],
            outputs=score,
            grad_outputs=self.go,
            retain_graph=True,
            create_graph=True,
        )
        packed_grads, _ = pack(list(grads), "b *")
        grads_norm = packed_grads.norm(2, dim=-1)

        if self.hps.one_sided_pen:
            # penalize the gradient for having a norm GREATER than k
            grad_pen = torch.max(
                torch.zeros_like(grads_norm),
                grads_norm - self.hps.grad_pen_targ,
            ).pow(2)
        else:
            # penalize the gradient for having a norm LOWER OR GREATER than k
            grad_pen = (grads_norm - self.hps.grad_pen_targ).pow(2)

        return grad_pen.mean()  # average over batch

    @beartype
    def get_syn_rew(self,
                    state: torch.Tensor,
                    action: torch.Tensor,
                    next_state: torch.Tensor) -> torch.Tensor:
        # beartype should hangle the type asserts
        # define the discriminator inputs
        input_a = state
        input_b = next_state if self.hps.state_only else action

        # compure score
        score = self.disc(input_a, input_b).detach()
        # counterpart of GAN's minimax (also called "saturating") loss
        # numerics: 0 for non-expert-like states, goes to +inf for expert-like states
        # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
        # e.g. walking simulations that get cut off when the robot falls over
        minimax_reward = -torch.log(1. - torch.sigmoid(score) + 1e-8)
        if self.hps.minimax_only:
            reward = minimax_reward
        else:
            # counterpart of GAN"s non-saturating loss
            # recommended in the original GAN paper and later in (Fedus et al. 2017)
            # numerics: 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            non_satur_reward = ff.logsigmoid(score)
            # return the sum the two previous reward functions (as in AIRL, Fu et al. 2018)
            # numerics: might be better might be way worse
            reward = non_satur_reward + minimax_reward
        return reward

    @beartype
    def update_target_net(self):
        """Update the target networks"""

        if sum([self.hps.use_c51, self.hps.use_qr]) == 0:
            # if non-distributional, targets slowly track their non-target counterparts
            with torch.no_grad():
                if self.hps.prefer_td3_over_sac:
                    # using TD3 (SAC does not use a target actor)
                    for param, targ_param in zip(self.actr.parameters(),
                                                 self.targ_actr.parameters()):
                        new_param = self.hps.polyak * param
                        new_param += (1. - self.hps.polyak) * targ_param
                        targ_param.copy_(new_param)
                for param, targ_param in zip(self.crit.parameters(),
                                             self.targ_crit.parameters()):
                    new_param = self.hps.polyak * param
                    new_param += (1. - self.hps.polyak) * targ_param
                    targ_param.copy_(new_param)
                if self.hps.clipped_double:
                    for param, targ_param in zip(self.twin.parameters(),
                                                 self.targ_twin.parameters()):
                        new_param = self.hps.polyak * param
                        new_param += (1. - self.hps.polyak) * targ_param
                        targ_param.copy_(new_param)
        elif self.crit_updates_so_far % self.hps.targ_up_freq == 0:
            # distributional case: periodically set target weights with online models
            with torch.no_grad():
                actr_state_dict = self.actr.state_dict()
                crit_state_dict = self.crit.state_dict()
                if self.hps.prefer_td3_over_sac:
                    # using TD3 (SAC does not use a target actor)
                    self.targ_actr.load_state_dict(actr_state_dict)
                self.targ_crit.load_state_dict(crit_state_dict)
                if self.hps.clipped_double:
                    twin_state_dict = self.twin.state_dict()
                    self.targ_twin.load_state_dict(twin_state_dict)

    @beartype
    def save_to_path(self, path: Path, xtra: Optional[str] = None):
        """Save the agent to disk"""
        # prep checkpoint
        suffix = f"checkpoint_{self.timesteps_so_far}"
        if xtra is not None:
            suffix += f"_{xtra}"
        suffix += ".tar"
        path = path / suffix
        checkpoint = {
            "hps": self.hps,  # handy for archeology
            "timesteps_so_far": self.timesteps_so_far,
            # and now the state_dict objects
            "rms_obs": self.rms_obs.state_dict(),
            "actr": self.actr.state_dict(),
            "crit": self.crit.state_dict(),
            "disc": self.disc.state_dict(),
            "actr_opt": self.actr_opt.state_dict(),
            "crit_opt": self.crit_opt.state_dict(),
            "disc_opt": self.disc_opt.state_dict(),
            "actr_sched": self.actr_sched.state_dict(),
        }
        if self.hps.clipped_double:
            checkpoint.update({
                "twin": self.twin.state_dict(),
                "twin_opt": self.twin_opt.state_dict(),
            })
        # save checkpoint to filesystem
        torch.save(checkpoint, path)

    @beartype
    def load_from_path(self, path: Path):
        """Load an agent from disk into this one"""
        checkpoint = torch.load(path)
        if "timesteps_so_far" in checkpoint:
            self.timesteps_so_far = checkpoint["timesteps_so_far"]
        # the "strict" argument of `load_state_dict` is True by default
        self.rms_obs.load_state_dict(checkpoint["rms_obs"])
        self.actr.load_state_dict(checkpoint["actr"])
        self.crit.load_state_dict(checkpoint["crit"])
        self.disc.load_state_dict(checkpoint["disc"])
        self.actr_opt.load_state_dict(checkpoint["actr_opt"])
        self.crit_opt.load_state_dict(checkpoint["crit_opt"])
        self.disc_opt.load_state_dict(checkpoint["disc_opt"])
        self.actr_sched.load_state_dict(checkpoint["actr_sched"])
        if self.hps.clipped_double:
            if "twin" in checkpoint:
                self.twin.load_state_dict(checkpoint["twin"])
                if "twin_opt" in checkpoint:
                    self.twin_opt.load_state_dict(checkpoint["twin_opt"])
                else:
                    logger.warn("twin opt is missing from the loaded tar!")
                    logger.warn("we move on nonetheless, from a fresh opt")
            else:
                raise IOError("no twin found in checkpoint tar file")
        elif "twin" in checkpoint:  # in the case where clipped double is off
            logger.warn("there is a twin the loaded tar, but you want none")
