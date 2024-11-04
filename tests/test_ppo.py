

import gymnasium as gym
import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import PPO


def test_ep_advantages():
    ppo = PPO(
        discount_factor=0.99,
        gae_factor=0.95,
        norm_adv=1,
        clip_va_loss=0,
        conv_net=0,
        joint_network=1,
        use_gpu=False
    )

    # Test that it works for small examples with a vectorised env
    rewards = torch.tensor([[1.0, 1.0], [1.0, 0.5], [1.0, 2.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.5], [1.0, 2.0]], dtype=torch.float32)
    values = torch.tensor([[2.0, 0.6], [2.2, 1.2], [1.8, 1.0], [1.8, 1.0], [1.5, 0.8], [2.2, 1.0], [2.4, 0.6], [1.4, 1.4]], dtype=torch.float32)
    end_values = torch.tensor([1.6, 1.8, 0.6], dtype=torch.float32)
    done_flags = torch.tensor([[0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.int32)
    trunc_flags = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], dtype=torch.int32)

    advantages = ppo.compute_advantages(rewards, values, end_values, done_flags, trunc_flags)
    targets = torch.tensor(
        [[ 1.9285,  3.6539],
        [ 0.7979,  2.1966],
        [ 0.2296,  2.0272],
        [-0.8000,  0.0396],
        [ 2.4389, -0.8000],
        [ 0.8090,  2.8596],
        [-0.3902,  2.4090],
        [-0.4000,  1.1940]],
        dtype=torch.float32
    )

    time_steps = 8
    num_agents = 2
    assert advantages.size(0) == time_steps
    assert advantages.size(1) == num_agents
    assert advantages.dtype == torch.float32
    assert (advantages - targets).abs().mean() < 1e-4

    # Test that it works for a larger example with a single env
    rewards = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0], dtype=torch.float32).unsqueeze(dim=1)
    values = torch.tensor([2.0, 1.8, 1.6, 2.0, 1.8, 1.8, 1.8, 2.2, 2.0, 2.6, 1.6, 1.6, 2.4, 2.4, 2.4, 2.0, 1.8, 2.0, 2.0, 2.4], dtype=torch.float32).unsqueeze(dim=1)
    end_values = torch.tensor([2.4, 1.2, 1.8, 1.4, 2.2], dtype=torch.float32)
    done_flags = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32).unsqueeze(dim=1)
    trunc_flags = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.int32).unsqueeze(dim=1)

    advantages = ppo.compute_advantages(rewards, values, end_values, done_flags, trunc_flags)
    targets = torch.tensor(
        [[ 3.6492],
        [ 3.0486],
        [ 2.4078],
        [ 0.0296],
        [-0.8000],
        [ 4.0158],
        [ 2.1624],
        [ 1.8973],
        [ 1.1880],
        [ 0.3157],
        [ 1.4160],
        [ 0.4593],
        [-1.4000],
        [ 2.9224],
        [-0.0570],
        [ 0.3860],
        [ 6.7316],
        [ 5.9028],
        [ 3.1077],
        [ 0.7780]],
        dtype=torch.float32
    )

    time_steps = 20
    num_agents = 1
    assert advantages.size(0) == time_steps
    assert advantages.size(1) == num_agents
    assert advantages.dtype == torch.float32
    assert (advantages - targets).abs().mean() < 1e-3

def test_compute_losses():
    ppo = PPO(
        discount_factor=0.99,
        gae_factor=0.95,
        norm_adv=1,
        clip_va_loss=0,
        conv_net=0,
        joint_network=1,
        use_gpu=False
    )

    prob_ratios = torch.tensor([1.21, 0.99, 1.01, 1.05, 0.75], dtype=torch.float32)
    curr_values = torch.tensor([2.0, 1.8, 2.4, 0.8, 2.4], dtype=torch.float32)
    prev_values = torch.tensor([1.8, 1.8, 2.0, 1.0, 2.2], dtype=torch.float32)
    advantages = torch.tensor([1.9, 1.9, 2.3, 0.8, 2.3], dtype=torch.float32)
    clip_ratio = 0.2
    policy_loss, value_loss, clip_frac, kl_div = ppo.compute_losses(prob_ratios, curr_values, prev_values, advantages, clip_ratio)

    assert policy_loss.dtype == torch.float32
    assert policy_loss.size() == torch.Size([])
    assert (policy_loss - 0.0491).abs().mean() < 1e-3

    assert value_loss.dtype == torch.float32
    assert value_loss.size() == torch.Size([])
    assert (value_loss - 1.5520).abs().mean() < 1e-3

    assert clip_frac.dtype == torch.float32
    assert clip_frac.size() == torch.Size([])
    assert (clip_frac - 0.4000).abs().mean() < 1e-3

    assert kl_div.dtype == torch.float32
    assert kl_div.size() == torch.Size([])
    assert (kl_div - 0.0117).abs().mean() < 1e-3

def test_cartpole():
    ppo = PPO(
        discount_factor=0.99,
        gae_factor=0.95,
        norm_adv=1,
        clip_va_loss=1,
        conv_net=0,
        joint_network=0,
        use_gpu=False
    )

    def env_fn():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
        return env

    time_start =  time.time()
    ppo.train(
        env_fn=env_fn,
        num_updates=200,
        num_envs=4,
        steps_per_env=125,
        num_epochs=4,
        batch_size=128,
        critic_coef=0.5,
        entropy_coef=0.01,
        clip_ratio=0.2,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        target_div=0.01
    )

    train_time_limit = 60
    ep_return_threshold = 250
    assert time.time() - time_start < train_time_limit
    assert ppo.vec_env.max_ep_return > ep_return_threshold

def test_ant():
    ppo = PPO(
        discount_factor=0.99,
        gae_factor=0.95,
        norm_adv=1,
        clip_va_loss=1,
        conv_net=0,
        joint_network=0,
        use_gpu=False
    )

    def env_fn():
        env = gym.make("Ant-v4")
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
        return env

    time_start =  time.time()
    ppo.train(
        env_fn=env_fn,
        num_updates=200,
        num_envs=4,
        steps_per_env=125,
        num_epochs=4,
        batch_size=128,
        critic_coef=0.5,
        entropy_coef=0.01,
        clip_ratio=0.2,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        target_div=0.01
    )

    train_time_limit = 120
    ep_return_threshold = 200
    assert time.time() - time_start < train_time_limit
    assert ppo.vec_env.max_ep_return > ep_return_threshold

