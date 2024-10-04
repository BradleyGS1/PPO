

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import PPO


def test_cuda_available():
    assert torch.cuda.is_available()

def test_ep_returns():
    ppo = PPO(
        discount_factor=0.99,
        gae_factor=0.95,
        norm_adv=1,
        clip_va_loss=0,
        conv_net=0,
        joint_network=1
    )

    rewards = torch.tensor([1.0] * 6, dtype=torch.float32)
    values = torch.tensor([0.5, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2], dtype=torch.float32)
    ep_advantages = ppo.episode_advantages(rewards, values)
    targets = torch.tensor([4.9518, 3.8914, 3.7144, 2.2566, 1.9762, 0.1980], dtype=torch.float32, device="cuda")

    assert ep_advantages.get_device() >= 0
    assert ep_advantages.size() == torch.Size([6])
    assert (ep_advantages - targets).abs().mean() < 1e-3
