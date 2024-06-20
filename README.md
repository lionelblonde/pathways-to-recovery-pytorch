# Pathways to Recovery (P2R) (wip/retired for now)

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

## Future ideas of extension

Extension to environments with keypoints/keyframes like the ALE suite (Atari), or the cool
"Box-World" environment introduced in [this paper](https://arxiv.org/abs/1806.01830)
(DeepMind's _Relational Deep Reinforcement Learning_).
I have previously played with the latter ("Box-World") recovering its creation code from
DeepMind's [`pyColab`](https://github.com/google-deepmind/pycolab) repo. The wrapped code is
available here: [
`https://github.com/lionelblonde/ppo-gail-pytorch/blob/master/helpers/pycolab_envs.py`](
https://github.com/lionelblonde/ppo-gail-pytorch/blob/master/helpers/pycolab_envs.py).

As in the original paper, it is very likely that synthetic returns can only be beneficial in
situations where the agent is provided with extremely sparse reward signal. Allowing for such
densification of reward in scenarios where the reward already is treated toward that density goal
is likely to introduce a survival bias in the agent's behavior (i.e. staying alive has become
enough for the agent who is not seeing solving the imitation task as necessary in comparison).
While this problem can be solved by reward shaping (adjusting the reward components' coefficients
by hand), such engineering endeavour arguably defeats the purpose of learning-based imitation.
Environments where agents are to imitate an expert whose behavior is very rare and hard to discover
by chance/by mistake by exploring the MDP are likely to produce sparser imitation rewards.
Similarly, environments involving keypoints like picking up a key for a locked door to unlock later
in the episode, or finding the color associated with the color previously stepped on ("Box-World")
are examples of scenarios that could give an edge to add-ons like synthetic returns in imitation
learning.
