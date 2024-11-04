

import gymnasium as gym
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import Agent, SyncVecEnv


def test_vec_env_states():

    # Testing states shape upon vec env reset for a vector state space
    def env_fn():
        return gym.make("CartPole-v1")

    test_env = env_fn()
    agent = Agent(test_env.observation_space, test_env.action_space, conv_net=0, joint_net=1, device="cpu")

    vec_env = SyncVecEnv(env_fn, num_envs=6, steps_per_env=125, render_every=0, render_fps=0.0, agent=agent)
    states = vec_env.vec_reset()

    assert states.size() == torch.Size([6, 4])
    assert states.dtype == torch.float32

    # Testing states shape upon vec env reset for an image state space
    def env_fn():
        return gym.make("ALE/Breakout-v5")

    test_env = env_fn()
    agent = Agent(test_env.observation_space, test_env.action_space, conv_net=0, joint_net=1, device="cpu")

    vec_env = SyncVecEnv(env_fn, num_envs=4, steps_per_env=125, render_every=0, render_fps=0.0, agent=agent)
    states = vec_env.vec_reset()

    assert states.size() == torch.Size([4, 210, 160, 3])
    assert states.dtype == torch.float32

def test_vec_env_rollout():

    def matching_shape_dtype(x: torch.Tensor, shape: tuple, dtype: torch.dtype):
        assert x.size() == torch.Size(shape)
        assert x.dtype == dtype

    # Testing rollout shapes of vec env rollout for a vector state space and disrete action space
    def env_fn():
        return gym.make("CartPole-v1")

    test_env = env_fn()
    agent = Agent(test_env.observation_space, test_env.action_space, conv_net=0, joint_net=1, device="cpu")

    vec_env = SyncVecEnv(env_fn, num_envs=2, steps_per_env=125, render_every=0, render_fps=0.0, agent=agent)
    vec_env.rollout()

    assert vec_env.states.requires_grad
    matching_shape_dtype(vec_env.states, (125, 2, 4), torch.float32)
    matching_shape_dtype(vec_env.actions, (125, 2), torch.int32)
    matching_shape_dtype(vec_env.rewards, (125, 2), torch.float32)
    matching_shape_dtype(vec_env.done_flags, (125, 2), torch.int32)
    matching_shape_dtype(vec_env.trunc_flags, (125, 2), torch.int32)
    matching_shape_dtype(vec_env.values, (125, 2), torch.float32)
    matching_shape_dtype(vec_env.log_probs, (125, 2), torch.float32)

    # Testing rollout shapes of vec env rollout for a vector state space and continuous action space
    def env_fn():
        return gym.make("Ant-v4")

    test_env = env_fn()
    agent = Agent(test_env.observation_space, test_env.action_space, conv_net=0, joint_net=1, device="cpu")

    vec_env = SyncVecEnv(env_fn, num_envs=3, steps_per_env=250, render_every=0, render_fps=0.0, agent=agent)
    vec_env.rollout()

    assert vec_env.states.requires_grad
    matching_shape_dtype(vec_env.states, (250, 3, 27), torch.float32)
    matching_shape_dtype(vec_env.actions, (250, 3, 8), torch.float32)
    matching_shape_dtype(vec_env.rewards, (250, 3), torch.float32)
    matching_shape_dtype(vec_env.done_flags, (250, 3), torch.int32)
    matching_shape_dtype(vec_env.trunc_flags, (250, 3), torch.int32)
    matching_shape_dtype(vec_env.values, (250, 3), torch.float32)
    matching_shape_dtype(vec_env.log_probs, (250, 3), torch.float32)

