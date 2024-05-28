import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from beartype import beartype

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.experimental.vector.async_vector_env import AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv

from helpers import logger
import environments


@beartype
def get_benchmark(env_id: str):
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    mes = "this benchmark has been flagged as deprecated"
    assert benchmark is not None, "unsupported environment"
    assert not environments.DEPRECATION_FLAGS[benchmark], mes  # order of asserts matter here (T-C)
    return benchmark


@beartype
def make_env(
    env_id: str,
    *,
    vectorized: bool,
    num_envs: Optional[int] = None,
    wrap_absorb: bool,
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv, SyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], float, int]:

    # create an environment
    bench = get_benchmark(env_id)  # at this point benchmark is valid

    if bench == "farama_mujoco":
        # remove the lockfile if it exists (maybe not a thing in Farama's Gymnasium anymore?)
        lockfile = (Path(os.environ["CONDA_PREFIX"]) / "lib" / "python3.7" / "site-packages" /
                    "mujoco_py" / "generated" / "mujocopy-buildlock.lock")
        try:
            Path(lockfile).unlink()
            logger.info("[WARN] removed mujoco lockfile")
        except OSError:
            pass

        return make_farama_mujoco_env(
            env_id,
            vectorized=vectorized,
            num_envs=num_envs,
            wrap_absorb=wrap_absorb,
            record=record,
            render=render,
        )
    raise ValueError(f"invalid benchmark: {bench}")


@beartype
def make_farama_mujoco_env(
    env_id: str,
    *,
    vectorized: bool,
    num_envs: Optional[int],
    wrap_absorb: bool,
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv, SyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], float, int]:

    # not ideal for code golf, but clearer for debug

    assert sum([record, vectorized]) <= 1, "not both same time"
    assert sum([render, vectorized]) <= 1, "not both same time"
    assert (not vectorized) or (num_envs is not None), "must give num_envs when vectorized"

    # create env
    # normally the windowed one is "human" .other option for later: "rgb_array", but prefer:
    # the following: `from gymnasium.wrappers.pixel_observation import PixelObservationWrapper`
    if record:  # overwrites render
        env = gym.make(env_id, render_mode="rgb_array_list")
    elif render:
        env = gym.make(env_id, render_mode="human")
        # reference: https://younis.dev/blog/render-api/
    elif vectorized:
        assert num_envs is not None
        env = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="async")
        assert isinstance(env, (AsyncVectorEnv, SyncVectorEnv))
        logger.info("using vectorized envs")
    else:
        env = gym.make(env_id)

    # build shapes for nets and replay buffer
    net_shapes = {}
    erb_shapes = {}

    # for the nets
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box)  # for due diligence
    ob_shape = ob_space.shape
    assert ob_shape is not None
    ac_space = env.action_space  # used now and later to get max action
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError(f"env ({env}) is discrete: out of scope here")
    assert isinstance(ac_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_shape = ac_space.shape
    assert ac_shape is not None
    net_shapes.update({"ob_shape": ob_shape, "ac_shape": ac_shape})

    # for the replay buffer
    ob_dim = ob_dim_orig = ob_shape[-1]
    ac_dim = ac_dim_orig = ac_shape[-1]  # for both: num dims
    if wrap_absorb:
        ob_dim += 1
        ac_dim += 1
        erb_shapes.update({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
            "obs0_orig": (ob_dim_orig,),
            "obs1_orig": (ob_dim_orig,),
            "acs_orig": (ac_dim_orig,),
        })
    else:
        erb_shapes.update({
            "obs0": (ob_dim,),
            "obs1": (ob_dim,),
            "acs": (ac_dim,),
            "rews": (1,),
            "dones1": (1,),
        })

    # max value for action
    max_ac = max(
        np.abs(np.amax(ac_space.high.astype("float32"))),
        np.abs(np.amin(ac_space.low.astype("float32"))),
    ).item()  # return it not as an ndarray but a standard Python scalar
    # despite the fact that we use the max, the actions are clipped with min and max
    # during interaction in the orchestrator

    # max episode length
    # use to be needed to determine the nature of termination in env
    # (Farama's Gymnasium make the distinction now in the `step` output)
    # now it is just needed to wrap the absorbing states in the demo dataset
    assert env.spec is not None
    max_ep_steps = env.spec.max_episode_steps
    assert max_ep_steps is not None
    if wrap_absorb:
        # when wrapping with absorbing states, count the extra transition
        max_ep_steps += 1

    return env, net_shapes, erb_shapes, max_ac, max_ep_steps
