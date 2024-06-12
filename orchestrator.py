import os
import time
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Union, Callable, ContextManager
from collections import deque
from contextlib import contextmanager, nullcontext

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import rearrange
from termcolor import colored
import wandb
from wandb.errors import CommError
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.experimental.vector.async_vector_env import AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv

from helpers import logger
from helpers.opencv_util import record_video
from agents.eve_agent import EveAgent


DEBUG = False


@beartype
def prettify_numb(n: int) -> str:
    """Display an integer number of millions, ks, etc."""
    m, k = divmod(n, 1_000_000)
    k, u = divmod(k, 1_000)
    return colored(f"{m}M {k}K {u}U", "red", attrs=["reverse"])


@beartype
@contextmanager
def timed(op: str, timer: Callable[[], float]):
    logger.info(colored(
        f"starting timer | op: {op}",
        "magenta", attrs=["underline", "bold"]))
    tstart = timer()
    yield
    tot_time = timer() - tstart
    logger.info(colored(
        f"stopping timer | op took {tot_time}secs",
        "magenta"))


@beartype
def segment(env: Union[Env, AsyncVectorEnv, SyncVectorEnv],
            agent: EveAgent,
            seed: int,
            segment_len: int,
            *,
            wrap_absorb: bool,
            lstm_mode: bool,
            enable_sr: bool):

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    t = 0

    assert agent.replay_buffers is not None
    ongoing_trajs, length = None, None  # quiets down the type-checker
    if lstm_mode or enable_sr:
        assert agent.traject_stores is not None
        ongoing_trajs = [
            deque([], maxlen=(length := agent.traject_stores[0].em_mxlen))
            for _ in range(
                env.num_envs if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)) else 1)]
        # as usual, index 0 chosen because it always exists whether vecencs are used or not
        logger.warn(f"the ongoing trajects are stored in deques of {length=}")

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
        new_ob, _, terminated, truncated, info = env.step(ac)  # reward ignored

        if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            logger.debug(f"{terminated=} | {truncated=}")
            assert isinstance(terminated, np.ndarray)
            assert isinstance(truncated, np.ndarray)
            assert terminated.shape == truncated.shape

        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            done, terminated = np.array([terminated or truncated]), np.array([terminated])
            if truncated:
                logger.debug("termination caused by something like time limit or out of bounds?")
        else:
            done = np.logical_or(terminated, truncated)  # might not be used but diagnostics
            done, terminated = rearrange(done, "b -> b 1"), rearrange(terminated, "b -> b 1")
        # read about what truncation means at the link below:
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#truncation

        tr_or_vtr = [ob, ac, new_ob, terminated]
        # note: we use terminated as a done replacement, but keep the key "dones1"
        # because it is the key used in the demo files

        if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            pp_func = partial(postproc_vtr, env.num_envs, info)
        else:
            assert isinstance(env, Env)
            pp_func = postproc_tr
        outss = pp_func(tr_or_vtr, agent.ob_shape, agent.ac_shape, wrap_absorb=wrap_absorb)
        assert outss is not None
        for i, outs in enumerate(outss):  # iterate over env (although maybe only one non-vec)
            for j, out in enumerate(outs):  # iterate over transitions
                # add transition to the i-th replay buffer
                pp_out = agent.replay_buffers[i].append(out, rew_func=agent.get_syn_rew)

                if lstm_mode or enable_sr:
                    # add transition to the currently ongoing trajectory in the i-th env
                    assert ongoing_trajs is not None
                    ongoing_trajs[i].append(pp_out)

                    if bool(pp_out["dones1"]) and (j + 1) == len(outs):
                        # second cond: if absorbing, there are two dones in a row -> stop at last

                        # the env time limit set by TimeLimit wrapper was once not respected
                        # and all reproducibility effort did not conclude with an conclusive answer
                        # TODO(lionel): find out why this is happening (Gymnasium bug?)

                        # since end of the trajectory, add the trajectory to the i-th traject store
                        assert agent.traject_stores is not None  # quiets down the type-checker
                        agent.traject_stores[i].append(list(ongoing_trajs[i]))
                        # reset the ongoing_trajs to an empty one
                        ongoing_trajs[i] = deque([], maxlen=length)

                # log how filled the i-th replay buffer and i-th trajectory store are
                logger.debug(
                    f"rb#{i} (#entries)/capacity: {agent.replay_buffers[i].how_filled}")
                if lstm_mode or enable_sr:
                    assert agent.traject_stores is not None  # quiets down the type-checker
                    logger.debug(
                        f"ts#{i} (#entries)/capacity: {agent.traject_stores[i].how_filled}")

        # set current state with the next
        ob = deepcopy(new_ob)

        if not isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
            assert isinstance(env, Env)
            if done:
                ob, _ = env.reset(seed=seed)

        t += 1


@beartype
def postproc_vtr(num_envs: int,
                 info: dict[str, np.ndarray],
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
        ob, ac, _, terminated = tr
        if "final_observation" in info:
            if bool(info["_final_observation"][i]):
                logger.debug("writing over new_ob with info[final_observation]")
                tr = [ob, ac, info["final_observation"][i], terminated]
        outs = postproc_tr(tr, ob_shape, ac_shape, wrap_absorb=wrap_absorb)
        vouts.extend(outs)
    return vouts


@beartype
def postproc_tr(tr: list[np.ndarray],
                ob_shape: tuple[int, ...],
                ac_shape: tuple[int, ...],
                *,
                wrap_absorb: bool) -> list[tuple[dict[str, np.ndarray], ...]]:

    ob, ac, new_ob, terminated = tr

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
                "dones1": terminated,
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
                "dones1": terminated,
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
            "dones1": terminated,
            "obs0_orig": ob,
            "acs_orig": ac,
            "obs1_orig": new_ob,
        }
        return [(transition,)]

    transition = {
        "obs0": ob,
        "acs": ac,
        "obs1": new_ob,
        "dones1": terminated,
    }
    return [(transition,)]


@beartype
def episode(env: Env,
            agent: EveAgent,
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
             agent_wrapper: Callable[[], EveAgent],
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
          agent_wrapper: Callable[[], EveAgent],
          timer_wrapper: Callable[[], Callable[[], float]],
          name: str):

    assert isinstance(cfg, DictConfig)

    # create an agent
    agent = agent_wrapper()

    # create a timer
    timer = timer_wrapper()

    # create context manager
    @beartype
    def ctx(op: str) -> ContextManager:
        return timed(op, timer) if DEBUG else nullcontext()

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
    group = ".".join(name.split(".")[:2])
    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
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

    for glob in ["train_actr", "train_crit", "train_disc", "train_sr", "eval"]:  # wandb categories
        # define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    # create segment generator for training the agent
    roll_gen = segment(
        env, agent, cfg.seed, cfg.segment_len,
        wrap_absorb=cfg.wrap_absorb, lstm_mode=cfg.lstm_mode, enable_sr=cfg.enable_sr)
    # create episode generator for evaluating the agent
    eval_seed = cfg.seed + 123456  # arbitrary choice
    ep_gen = episode(eval_env, agent, eval_seed)

    i = 0

    while agent.timesteps_so_far <= cfg.num_timesteps:

        logger.info((f"iter#{i}").upper())
        if i % cfg.eval_every == 0:
            logger.warn((f"iter#{i}").upper())
            # so that when logger level is WARN, we see the iter number before the the eval metrics

        logger.info(("interact").upper())
        its = timer()
        next(roll_gen)  # no need to get the returned segment, stored in buffer
        agent.timesteps_so_far += cfg.segment_len
        logger.info(f"so far {prettify_numb(agent.timesteps_so_far)} steps made")
        logger.info(colored(
            f"interaction time: {timer() - its}secs",
            "green"))

        logger.info(("train").upper())

        tts = timer()
        ttl = []
        gtl = []
        dtl = []
        gs, ds = 0, 0
        for _ in range(tot := cfg.training_steps_per_iter):

            gts = timer()
            for _ in range(gs := cfg.g_steps):

                # sample a batch of transitions and trajectories
                trns_batch = None
                if not cfg.lstm_mode:
                    trns_batch = agent.sample_trns_batch()
                trjs_batch = None
                if cfg.lstm_mode or cfg.enable_sr:
                    trjs_batch = agent.sample_trjs_batch()

                lstm_precomp_hstate = None
                if (there_is_at_least_one_trj := trjs_batch is not None):
                    with ctx("sr training"):
                        lstm_precomp_hstate = agent.update_sr(
                            trjs_batch, just_relay_hstate=(cfg.lstm_mode and not cfg.enable_sr))

                trxs_batch = trjs_batch if cfg.lstm_mode else trns_batch

                if (cfg.lstm_mode and there_is_at_least_one_trj) or not cfg.lstm_mode:
                    assert trxs_batch is not None  # to quiet down the type-checker
                    # determine if updating the actr
                    update_actr = not bool(agent.crit_updates_so_far % cfg.actor_update_delay)
                    with ctx("actor-critic training"):
                        # update the actor and critic
                        agent.update_actr_crit(trxs_batch, lstm_precomp_hstate,
                            update_actr=update_actr, use_sr=there_is_at_least_one_trj)

                gtl.append(timer() - gts)
                gts = timer()

            dts = timer()
            for _ in range(ds := cfg.d_steps):
                # sample a batch of transitions from the replay buffer
                trns_batch = agent.sample_trns_batch()
                with ctx("discriminator training"):
                    # update the discriminator
                    agent.update_disc(trns_batch)

                dtl.append(timer() - dts)
                dts = timer()

            ttl.append(timer() - tts)
            tts = timer()

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
            wandb_dict = {
                **{f"{k}-mean": v.mean() for k, v in eval_metrics.items()},
                "rbx-num-entries": np.array(agent.replay_buffers[0].num_entries),
                # taking the first because this one will always exist whatever the numenv
                "avg-tt-per-iter": avg_tt_per_iter}
            if cfg.lstm_mode or cfg.enable_sr:
                assert agent.traject_stores is not None
                wandb_dict.update({
                    "tsx-num-entries": np.array(agent.traject_stores[0].num_entries)})
            agent.send_to_dash(
                wandb_dict,
                step_metric=agent.timesteps_so_far,
                glob="eval",
            )

        logger.info()

    # save once we are done
    agent.save_to_path(ckpt_dir, xtra="done")
    logger.info(f"we are done. saving model @:\n{ckpt_dir}\nbye.")
