import time
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Union, Callable

from beartype import beartype
from einops import rearrange
from termcolor import colored
from omegaconf import OmegaConf, DictConfig
import wandb
from wandb.errors import CommError
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.experimental.vector.async_vector_env import AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv

from helpers import logger
from helpers.misc_util import log_iter_info, prettify_numb
from helpers.opencv_util import record_video
from agents.spp_agent import SPPAgent


DEBUG = False


@beartype
def segment(env: Union[Env, AsyncVectorEnv, SyncVectorEnv],
            agent: SPPAgent,
            seed: int,
            segment_len: int,
            *,
            wrap_absorb: bool):

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    assert agent.replay_buffers is not None

    t = 0

    ob, _ = env.reset(seed=seed)  # seed is a keyword argument, not positional

    while True:

        # predict action
        assert isinstance(ob, np.ndarray)
        ac = agent.predict(ob, apply_noise=True)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, ac_low, ac_high)

        if t > 0 and t % segment_len == 0:
            yield

        # interact with env
        new_ob, _, terminated, truncated, _ = env.step(ac)  # reward and info ignored
        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            done, terminated = np.array([terminated or truncated]), np.array([terminated])
            if truncated:
                logger.warn("termination caused by something like time limit or out of bounds?")
        else:
            done = np.logical_or(terminated, truncated)  # might not be used but diagnostics
            done, terminated = rearrange(done, "b -> b 1"), rearrange(terminated, "b -> b 1")
        # read about what truncation means at the link below:
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#truncation

        tr_or_vtr = [ob, ac, new_ob, done, terminated]
        if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            pp_func = partial(postproc_vtr, env.num_envs)
        else:
            assert isinstance(env, Env)
            pp_func = postproc_tr
        outss = pp_func(tr_or_vtr, agent.ob_shape, agent.ac_shape, wrap_absorb=wrap_absorb)
        assert outss is not None
        for i, outs in enumerate(outss):
            for out in outs:
                # add transition to the i-th replay buffer
                agent.replay_buffers[i].append(out, rew_func=agent.get_syn_rew)

        # set current state with the next
        ob = deepcopy(new_ob)

        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            if done:
                ob, _ = env.reset(seed=seed)

        t += 1


@beartype
def postproc_vtr(num_envs: int,
                 vtr: list[np.ndarray],
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 *,
                 wrap_absorb: bool) -> list[tuple[dict[str, np.ndarray], ...]]:
    # N.B.: for the num of envs and the workloads, serial treatment is faster than parallel
    # time it takes for the main process to spawn X threads is too much overhead
    # it starts becoming interesting if the post-processing is heavier though
    vouts = []
    for i in range(num_envs):
        tr = [e[i] for e in vtr]
        outs = postproc_tr(tr, ob_shape, ac_shape, wrap_absorb=wrap_absorb)
        vouts.extend(outs)
    return vouts


@beartype
def postproc_tr(tr: list[np.ndarray],
                ob_shape: tuple[int, ...],
                ac_shape: tuple[int, ...],
                *,
                wrap_absorb: bool) -> list[tuple[dict[str, np.ndarray], ...]]:

    ob, ac, new_ob, done, terminated = tr

    if wrap_absorb:

        ob_0 = np.append(ob, 0)
        ac_0 = np.append(ac, 0)

        # previously this was the cond: `done and env._elapsed_steps != env._max_episode_steps`
        if terminated:
            # wrap with an absorbing state
            new_ob_zeros_1 = np.append(np.zeros(ob_shape[-1]), 1)
            transition = {
                "obs0": ob_0,
                "acs": ac_0,
                "obs1": new_ob_zeros_1,
                "dones1": done,
                "obs0_orig": ob,
                "acs_orig": ac,
                "obs1_orig": new_ob,
            }
            # add absorbing transition
            ob_zeros_1 = np.append(np.zeros(ob_shape[-1]), 1)
            ac_zeros_1 = np.append(np.zeros(ac_shape[-1]), 1)
            new_ob_zeros_1 = np.append(np.zeros(ob_shape[-1]), 1)
            transition_a = {
                "obs0": ob_zeros_1,
                "acs": ac_zeros_1,
                "obs1": new_ob_zeros_1,
                "dones1": done,
                "obs0_orig": ob,  # from previous transition, with reward eval on absorbing
                "acs_orig": ac,  # from previous transition, with reward eval on absorbing
                "obs1_orig": new_ob,  # from previous transition, with reward eval on absorbing
            }
            return [(transition, transition_a)]

        new_ob_0 = np.append(new_ob, 0)
        transition = {
            "obs0": ob_0,
            "acs": ac_0,
            "obs1": new_ob_0,
            "dones1": done,
            "obs0_orig": ob,
            "acs_orig": ac,
            "obs1_orig": new_ob,
        }
        return [(transition,)]

    transition = {
        "obs0": ob,
        "acs": ac,
        "obs1": new_ob,
        "dones1": done,
    }
    return [(transition,)]


@beartype
def episode(env: Env,
            agent: SPPAgent,
            seed: int):
    # generator that spits out a trajectory collected during a single episode
    # `append` operation is also significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible
    logger.warn("remember: in episode generator, we generate a seed randomly")
    logger.warn("i.e. not using 'ob, _ = env.reset(seed=seed)' with same seed")
    # note that despite sampling a new seed, it is using a seeded rng: reproducible
    ob, _ = env.reset(seed=seed + rng.integers(100000, size=1).item())

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    env_rews = []

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=False)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, ac_low, ac_high)

        obs.append(ob)
        acs.append(ac)
        new_ob, env_rew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated

        env_rews.append(env_rew)
        cur_ep_len += 1
        assert isinstance(env_rew, float)  # quiets the type-checker
        cur_ep_env_ret += env_rew
        ob = deepcopy(new_ob)

        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {
                "obs": obs,
                "acs": acs,
                "env_rews": env_rews,
                "ep_len": cur_ep_len,
                "ep_env_ret": cur_ep_env_ret,
            }
            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            env_rews = []
            logger.warn("remember: in episode generator, we generate a seed randomly")
            logger.warn("i.e. not using 'ob, _ = env.reset(seed=seed)' with same seed")
            ob, _ = env.reset(seed=seed + rng.integers(100000, size=1).item())


@beartype
def evaluate(cfg: DictConfig,
             env: Env,
             agent_wrapper: Callable[[], SPPAgent],
             name: str):

    assert isinstance(cfg, DictConfig)

    vid_dir = Path(cfg.video_dir) / name
    if cfg.record:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # create an agent
    agent = agent_wrapper()

    # create episode generator
    ep_gen = episode(env, agent, cfg.seed)

    # load the model
    model_path = cfg.model_path
    agent.load_from_path(model_path)
    logger.info(f"model loaded from path:\n {model_path}")

    # collect trajectories

    num_trajs = cfg.num_trajs
    len_buff, env_ret_buff = [], []

    for i in range(num_trajs):

        logger.info(f"evaluating [{i + 1}/{num_trajs}]")
        traj = next(ep_gen)
        ep_len, ep_env_ret = traj["ep_len"], traj["ep_env_ret"]

        # aggregate to the history data structures
        len_buff.append(ep_len)
        env_ret_buff.append(ep_env_ret)

        if cfg.record:
            # record a video of the episode
            frame_collection = env.render()  # ref: https://younis.dev/blog/render-api/
            record_video(vid_dir, str(i), np.array(frame_collection))

    eval_metrics = {"ep_len": len_buff, "ep_env_ret": env_ret_buff}

    # log stats in csv
    logger.record_tabular("timestep", agent.timesteps_so_far)
    for k, v in eval_metrics.items():
        logger.record_tabular(f"{k}-mean", np.mean(v))
    logger.info("dumping stats in .csv file")
    logger.dump_tabular()


@beartype
def learn(cfg: DictConfig,
          env: Union[Env, AsyncVectorEnv, SyncVectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], SPPAgent],
          name: str):

    assert isinstance(cfg, DictConfig)

    # create an agent
    agent = agent_wrapper()

    # start clock
    tstart = time.time()

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vid_dir = Path(cfg.video_dir) / name
    if cfg.record:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # save the model as a dry run, to avoid bad surprises at the end
    agent.save_to_path(ckpt_dir, xtra="dryrun")
    logger.info(f"dry run. saving model @:\n{ckpt_dir}")

    # group by everything except the seed, which is last, hence index -1
    # it groups by uuid + gitSHA + env_id + num_demos
    group = ".".join(name.split(".")[:-1])
    # set up wandb
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    for glob in ["train_actr", "train_crit", "train_disc", "eval"]:  # wandb categories
        # define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    # create segment generator for training the agent
    roll_gen = segment(env, agent, cfg.seed, cfg.segment_len, wrap_absorb=cfg.wrap_absorb)
    # create episode generator for evaluating the agent
    eval_seed = cfg.seed + 123456  # arbitrary choice
    ep_gen = episode(eval_env, agent, eval_seed)

    i = 0

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if i % 100 == 0 or DEBUG:
            log_iter_info(i, cfg.num_timesteps // cfg.segment_len, tstart)

        logger.info(("interact").upper())
        its = time.time()
        next(roll_gen)  # no need to get the returned segment, stored in buffer
        agent.timesteps_so_far += cfg.segment_len
        logger.info(f"so far {prettify_numb(agent.timesteps_so_far)} steps made")
        logger.info(colored(
            f"interaction time: {time.time() - its}secs",
            "green"))

        logger.info(("train").upper())

        tts = time.time()
        ttl = []
        gtl = []
        dtl = []
        gs, ds = 0, 0
        for _ in range(tot := cfg.training_steps_per_iter):

            gts = time.time()
            for _ in range(gs := cfg.g_steps):
                # sample a batch of transitions from the replay buffer
                batch = agent.sample_batch()
                # determine if updating the actr
                update_actr = not bool(
                    agent.crit_updates_so_far % cfg.actor_update_delay)
                # update the actor and critic
                agent.update_actr_crit(
                    batch=batch,
                    update_actr=update_actr,
                )  # counters for actr and crit updates are incremented internally!
                gtl.append(time.time() - gts)
                gts = time.time()

            dts = time.time()
            for _ in range(ds := cfg.d_steps):
                # sample a batch of transitions from the replay buffer
                batch = agent.sample_batch()
                # update the discriminator
                agent.update_disc(batch)  # update counter incremented internally too
                dtl.append(time.time() - dts)
                dts = time.time()

            ttl.append(time.time() - tts)
            tts = time.time()

        logger.info(colored(
            f"avg tt over {tot}steps: {(avg_tt_per_iter := np.mean(ttl))}secs",  # logged in eval
            "green", attrs=["reverse"]))
        logger.info(colored(
            f"avg gt over {tot}steps X {gs} g-steps: {np.mean(gtl)}secs",
            "green"))
        logger.info(colored(
            f"avg dt over {tot}steps X {ds} d-steps: {np.mean(dtl)}secs",
            "green"))
        logger.info(colored(
            f"tot tt over {tot}steps: {np.sum(ttl)}secs",
            "magenta", attrs=["reverse"]))

        i += 1

        if i % cfg.eval_every == 0:

            logger.info(("eval").upper())

            len_buff, env_ret_buff = [], []

            for j in range(cfg.eval_steps_per_iter):

                # sample an episode with non-perturbed actor
                ep = next(ep_gen)
                # none of it is collected in the replay buffer

                len_buff.append(ep["ep_len"])
                env_ret_buff.append(ep["ep_env_ret"])

                if cfg.record:
                    # record a video of the episode
                    # ref: https://younis.dev/blog/render-api/
                    frame_collection = eval_env.render()
                    record_video(vid_dir, f"iter{i}-ep{j}", np.array(frame_collection))

            eval_metrics: dict[str, np.ndarray] = {
                "ep_len": np.array(len_buff), "ep_env_ret": np.array(env_ret_buff)}

            # log stats in csv
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for k, v in eval_metrics.items():
                logger.record_tabular(f"{k}-mean", v.mean())
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            # log stats in dashboard
            assert agent.replay_buffers is not None
            agent.send_to_dash(
                {**{f"{k}-mean": v.mean() for k, v in eval_metrics.items()},
                 "rbx-num-entries": np.array(agent.replay_buffers[0].num_entries),
                 # taking the first because this one will always exist whatever the numenv
                 "avg-tt-per-iter": avg_tt_per_iter},
                step_metric=agent.timesteps_so_far,
                glob="eval",
            )

        logger.info()

    # save once we are done
    agent.save_to_path(ckpt_dir, xtra="done")
    logger.info(f"we are done. saving model @:\n{ckpt_dir}\nbye.")
