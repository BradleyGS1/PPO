

import gymnasium as gym
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import Agent, VecEnv


def assert_shape_and_dtype(x: torch.Tensor, shape: list, dtype: torch.dtype):
    assert x.size() == torch.Size(shape)
    assert x.dtype == dtype

def test_vec_env_states():
    agent = Agent(state_shape=(4,), action_shape=(2,), conv_net=False, joint_net=True)

    def env_fn():
        return gym.make("CartPole-v1")

    vec_env = VecEnv(env_fn=env_fn, num_envs=2, steps_per_env=100, agent=agent)
    states = vec_env.vec_env_states()

    assert_shape_and_dtype(states, shape=(2, 4), dtype=torch.float32)


    agent = Agent(state_shape=(6,), action_shape=(4,), conv_net=False, joint_net=True)

    def env_fn():
        return gym.make("Acrobot-v1")

    vec_env = VecEnv(env_fn=env_fn, num_envs=8, steps_per_env=100, agent=agent)
    states = vec_env.vec_env_states()

    assert_shape_and_dtype(states, shape=(8, 6), dtype=torch.float32)

def test_vec_env_step():
    agent = Agent(state_shape=(4,), action_shape=(2,), conv_net=False, joint_net=True)

    def env_fn():
        return gym.make("CartPole-v1")

    vec_env = VecEnv(env_fn=env_fn, num_envs=2, steps_per_env=100, agent=agent)
    states = vec_env.vec_env_states()
    actions = agent.get_actions_and_values(states, actions=None)[0]
    rewards, done_flags, trunc_flags = vec_env.vec_env_step(actions)

    assert_shape_and_dtype(actions, shape=(2, ), dtype=torch.int32)
    assert torch.equal(rewards, torch.tensor([1.0, 1.0], dtype=torch.float32, device="cuda"))
    assert torch.equal(done_flags, torch.tensor([0, 0], dtype=torch.int32, device="cuda"))
    assert torch.equal(trunc_flags, torch.tensor([0, 0], dtype=torch.int32, device="cuda"))

def test_vec_env_rollout():
    agent = Agent(state_shape=(4,), action_shape=(2,), conv_net=False, joint_net=True)

    def env_fn():
        return gym.make("CartPole-v1")

    vec_env = VecEnv(env_fn=env_fn, num_envs=2, steps_per_env=100, agent=agent)
    vec_env.rollout()

    assert vec_env.states.requires_grad
    assert_shape_and_dtype(vec_env.states, shape=(200, 4), dtype=torch.float32)
    assert_shape_and_dtype(vec_env.actions, shape=(200, ), dtype=torch.int32)
    assert_shape_and_dtype(vec_env.rewards, shape=(200, ), dtype=torch.float32)
    assert_shape_and_dtype(vec_env.done_flags, shape=(200, ), dtype=torch.int32)
    assert_shape_and_dtype(vec_env.trunc_flags, shape=(200, ), dtype=torch.int32)
    assert_shape_and_dtype(vec_env.values, shape=(200, ), dtype=torch.float32)
    assert_shape_and_dtype(vec_env.log_probs, shape=(200, ), dtype=torch.float32)
