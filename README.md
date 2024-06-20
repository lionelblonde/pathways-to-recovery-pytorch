# Pathways to Recovery (P2R)

The goal of this repository is the investigate the extent to which off-policy AIL methods are
suffering from __credit misassignment__, and to propose a solution consisting to providing a way to
learn a _recovery behavior_ explicitly, helping the policy to get back on the expert track.
This auxiliairy support behavior is distilled in the agent via the augmentation of the imitation
reward with _synthetic returns_ (official impl. from DeepMind in Jax available at [this link](
https://github.com/google-deepmind/deepmind-research/tree/master/synthetic_returns)).

The agents are set to run in the [Gymnasium](
https://gymnasium.farama.org/index.html) suite of environments.
Extensions to other suites should be straighforward considering how common the API is.

Download the expert demonstrations at
(this link (Google Drive))
[https://drive.google.com/drive/folders/1dGw-O6ZT_WWTuqDayIA9xat1jZgeiXoE?usp=drive_link].

## Future Work

Extension to environments with keypoints/keyframes like the ALE suite (Atari), or the cool
"Box-World" environment introduced in [this paper](https://arxiv.org/abs/1806.01830)
(DeepMind's _Relational Deep Reinforcement Learning_).
I have previously played with the latter ("Box-World") recovering its creation code from
DeepMind's [`pyColab`](https://github.com/google-deepmind/pycolab) repo. The wrapped code is
available here:
`https://github.com/lionelblonde/ppo-gail-pytorch/blob/master/helpers/pycolab_envs.py`.
