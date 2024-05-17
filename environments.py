# OpenAI MuJoCo (TL;DR: deprecated)
# these are the old python environments from OpenAI mujoco-py and gym
# to be used via Farama Foundation Gymnasium, one need to install mujoco-py
# which is officially deprecated: we discourage that here
OPENAI_MUJOCO = []
OPENAI_MUJOCO_ = [
    "InvertedPendulum",
    "InvertedDoublePendulum",
    "Reacher",
    "Hopper",
    "HalfCheetah",
    "Walker2d",
    "Ant",
    "Humanoid",
]
OPENAI_MUJOCO.extend([f"{name}-v2" for name in OPENAI_MUJOCO_])
OPENAI_MUJOCO.extend([f"{name}-v3" for name in OPENAI_MUJOCO_])

# Farama Foundation Gymnasium MuJoCo
FARAMA_MUJOCO = []
FARAMA_MUJOCO_ = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "HumanoidStandup",
    "Humanoid",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "Swimmer",
    "Walker2d",
]
FARAMA_MUJOCO.extend([f"{name}-v4" for name in FARAMA_MUJOCO_])

# DeepMind Control Suite (DMC) MuJoCo
DEEPMIND_MUJOCO = []
DEEPMIND_MUJOCO_ = [
    "Hopper-Hop",
    "Cheetah-Run",
    "Walker-Walk",
    "Walker-Run",
    "Stacker-Stack_2",
    "Stacker-Stack_4",
    "Humanoid-Walk",
    "Humanoid-Run",
    "Humanoid-Run_Pure_State",
    "Humanoid_CMU-Stand",
    "Humanoid_CMU-Run",
    "Quadruped-Walk",
    "Quadruped-Run",
    "Quadruped-Escape",
    "Quadruped-Fetch",
    "Dog-Run",
    "Dog-Fetch",
]
DEEPMIND_MUJOCO.extend([f"{name}-Feat-v0" for name in DEEPMIND_MUJOCO_])
DEEPMIND_MUJOCO.extend([f"{name}-Pix-v0" for name in DEEPMIND_MUJOCO_])


BENCHMARKS = {
    "openai_mujoco": OPENAI_MUJOCO,
    "farama_mujoco": FARAMA_MUJOCO,
    "deepmind_mojoco": DEEPMIND_MUJOCO,
}
DEPRECATION_FLAGS = dict.fromkeys(BENCHMARKS, False)
# flag the environements that should not be used
DEPRECATION_FLAGS["openai_mujoco"] = True  # officially deprecated
DEPRECATION_FLAGS["deepmind_mujoco"] = True  # TODO(lionel): once gymnasium integrated, do dmc
