import os
import subprocess
from pathlib import Path
from typing import Optional

from beartype import beartype
import fire
from omegaconf import OmegaConf, DictConfig
import random
import numpy as np
import torch

from gymnasium.core import Env

import orchestrator
from helpers import logger
from helpers.env_makers import make_env
from helpers.dataset import DemoDataset
from agents.memory import ReplayBuffer
from agents.spp_agent import SPPAgent


@beartype
def make_uuid(num_syllables: int = 2, num_parts: int = 3) -> str:
    """Randomly create a semi-pronounceable uuid"""
    part1 = ["s", "t", "r", "ch", "b", "c", "w", "z", "h", "k", "p", "ph", "sh", "f", "fr"]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ "-", "_", "."]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


@beartype
def get_name(uuid: str, env_id: str, seed: int) -> str:
    """Assemble long experiment name"""
    name = uuid
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        sha = out.strip().decode("ascii")
        name += f".gitSHA_{sha}"
    except OSError:
        pass
    name += f".{env_id}"
    name += f".seed{str(seed).zfill(2)}"
    return name


class MagicRunner(object):

    DISABLE_LOGGER: bool = False

    @beartype
    def __init__(self, cfg: str,  # give the relative path to cfg here
                 env_id: str,  # never in cfg: always give one in arg
                 seed: int,  # never in cfg: always give one in arg
                 num_demos: int,  # never in cfg: always give one arg
                 expert_path: str,  # never in cfg: always give one arg
                 wandb_project: Optional[str] = None,  # is either given in arg (prio) or in cfg
                 uuid: Optional[str] = None,  # never in cfg, but not forced to give in arg either
                 load_ckpt: Optional[str] = None):  # same as uuid: from arg or nothing

        logger.configure_default_logger()

        # retrieve config from filesystem
        proj_root = Path(__file__).resolve().parent
        _cfg = OmegaConf.load(proj_root / Path(cfg))
        assert isinstance(_cfg, DictConfig)
        self._cfg: DictConfig = _cfg  # for the type-checker

        logger.info("the config loaded:")
        logger.info(OmegaConf.to_yaml(self._cfg))

        self._cfg.root = str(proj_root)  # in config: used by wandb
        for k in ("checkpoints", "logs", "videos"):
            new_k = f"{k[:-1]}_dir"
            self._cfg[new_k] = str(proj_root / k)  # for yml saving

        # set only if nonexistant key in cfg
        self._cfg.seed = seed
        self._cfg.env_id = env_id
        self._cfg.num_demos = num_demos
        self._cfg.expert_path = expert_path

        assert "wandb_project" in self._cfg  # if not in cfg from fs, abort
        if wandb_project is not None:
            self._cfg.wandb_project = wandb_project  # overwrite cfg

        assert "uuid" not in self._cfg  # uuid should never be in the cfg file
        self._cfg.uuid = uuid if uuid is not None else make_uuid()

        assert "load_ckpt" not in self._cfg  # load_ckpt should never be in the cfg file
        if load_ckpt is not None:
            self._cfg.load_ckpt = load_ckpt  # add in cfg
        else:
            logger.info("no ckpt to load: key will not exist in cfg")

        self.name = get_name(self._cfg.uuid, self._cfg.env_id, self._cfg.seed)

        # slight overwrite for consistency, before setting to read-only
        self._cfg.num_env = self._cfg.numenv if self._cfg.vecenv else 1

        # set the cfg to read-only for safety
        OmegaConf.set_readonly(self._cfg, value=True)

    @beartype
    def train(self):

        # mlsys
        torch.set_num_threads(self._cfg.num_env)
        # TODO(lionel): keep an eye on this

        # set printing options
        np.set_printoptions(precision=3)

        # name
        name = f"{self.name}.train_demos{str(self._cfg.num_demos).zfill(3)}"
        # logger
        if self.DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            log_path = Path(self._cfg.log_dir) / name
            log_path.mkdir(parents=True, exist_ok=True)
            logger.configure(directory=log_path, format_strs=["stdout", "log", "json", "csv"])
            # config dump
            OmegaConf.save(config=self._cfg, f=(log_path / "cfg.yml"))

        # device
        assert not self._cfg.fp16 or self._cfg.cuda, "fp16 => cuda"  # TODO(lionel): fp16 not done
        if self._cfg.cuda:
            # use cuda
            assert torch.cuda.is_available()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0")
        else:
            # default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        logger.info(f"device in use: {device}")

        # seed
        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed_all(self._cfg.seed)

        # env
        env, net_shapes, erb_shapes, max_ac, max_episode_steps = make_env(
            self._cfg.env_id,
            vectorized=self._cfg.vecenv,
            num_envs=self._cfg.numenv,
            wrap_absorb=self._cfg.wrap_absorb,
            record=False,
            render=self._cfg.render,
        )

        # create an agent wrapper

        expert_dataset = DemoDataset(
            generator=torch.Generator(device).manual_seed(self._cfg.seed),
            np_rng=np.random.default_rng(self._cfg.seed),
            device=device,
            expert_path=self._cfg.expert_path,
            num_demos=self._cfg.num_demos,
            max_ep_steps=max_episode_steps,
            wrap_absorb=self._cfg.wrap_absorb,
        )
        logger.info(f"dd#0 [{expert_dataset}] is set")

        replay_buffers = [ReplayBuffer(
            generator=torch.Generator(device).manual_seed(self._cfg.seed),
            capacity=self._cfg.mem_size,
            erb_shapes=erb_shapes,
            device=device,
        ) for _ in range(self._cfg.num_env)]
        for i, rb in enumerate(replay_buffers):
            logger.info(f"rb#{i} [{rb}] is set")

        def agent_wrapper():
            return SPPAgent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                actr_noise_rng=torch.Generator(device).manual_seed(self._cfg.seed),
                expert_dataset=expert_dataset,
                replay_buffers=replay_buffers,
            )

        # create an evaluation environment not to mess up with training rollouts
        eval_env, _, _, _, _ = make_env(
            self._cfg.env_id,
            vectorized=False,
            wrap_absorb=self._cfg.wrap_absorb,
            record=self._cfg.record,
            render=self._cfg.render,
        )
        assert isinstance(eval_env, Env), "no vecenv allowed here"

        # train
        orchestrator.learn(
            cfg=self._cfg,
            env=env,
            eval_env=eval_env,
            agent_wrapper=agent_wrapper,
            name=name,
        )

        # cleanup
        env.close()
        eval_env.close()

    @beartype
    def evaluate(self):

        # mlsys
        torch.set_num_threads(1)  # TODO(lionel): keep an eye on this

        # set printing options
        np.set_printoptions(precision=3)

        # name
        name = f"{self.name}.eval_trajs{str(self._cfg['num_trajs']).zfill(2)}"
        # logger
        if self.DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            logger.configure(directory=None, format_strs=["stdout"])

        # device
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage

        # seed
        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed_all(self._cfg.seed)

        # env
        env, net_shapes, _, max_ac, _ = make_env(
            self._cfg.env_id,
            vectorized=False,
            wrap_absorb=self._cfg.wrap_absorb,
            record=self._cfg.record,
            render=self._cfg.render,
        )
        assert isinstance(env, Env), "no vecenv allowed here"

        # create an agent wrapper
        def agent_wrapper():
            return SPPAgent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                actr_noise_rng=torch.Generator(device).manual_seed(self._cfg.seed),
                expert_dataset=None,
                replay_buffers=None,
            )

        # evaluate
        orchestrator.evaluate(
            cfg=self._cfg,
            env=env,
            agent_wrapper=agent_wrapper,
            name=name,
        )

        # cleanup
        env.close()


if __name__ == "__main__":
    fire.Fire(MagicRunner)
