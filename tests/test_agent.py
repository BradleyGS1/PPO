

import gymnasium as gym
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import ScaledBeta, Agent


def all_in_range(tensor: torch.Tensor, low: float, high: float) -> bool:
    num_out_of_range = torch.sum((tensor < low).int() + (tensor > high).int()).item()
    return num_out_of_range == 0

def test_init_dense():

    BATCH_SIZE = 1000

    # Test discrete action space model has the right input and output shape
    state_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_space = gym.spaces.Discrete(n=4)

    agent = Agent(state_space, action_space, conv_net=0, joint_net=1, device="cpu")

    states = torch.tensor(np.array([state_space.sample() for _ in range(BATCH_SIZE)], dtype=np.float32))

    actions, log_probs, values, entropy = agent.get_actions_and_values(states, actions=None)

    assert actions.size(0) == BATCH_SIZE
    assert actions.dtype == torch.int32
    assert all_in_range(actions, low=0, high=3)

    assert log_probs.size(0) == BATCH_SIZE
    assert log_probs.dtype == torch.float32

    assert values.size(0) == BATCH_SIZE
    assert values.dtype == torch.float32

    assert entropy.size() == torch.Size([])
    assert entropy.dtype == torch.float32

    # Test the continuous action space model has the right input and output shape
    state_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,))

    agent = Agent(state_space, action_space, conv_net=0, joint_net=1, device="cpu")

    states = torch.tensor(np.array([state_space.sample() for _ in range(BATCH_SIZE)], dtype=np.float32))

    actions, log_probs, values, entropy = agent.get_actions_and_values(states, actions=None)

    assert actions.size(0) == BATCH_SIZE
    assert actions.size(1) == 4
    assert actions.dtype == torch.float32
    assert all_in_range(actions, low=-10.0, high=10.0)

    assert log_probs.size(0) == BATCH_SIZE
    assert log_probs.dtype == torch.float32

    assert values.size(0) == BATCH_SIZE
    assert values.dtype == torch.float32

    assert entropy.size() == torch.Size([])
    assert entropy.dtype == torch.float32

def test_init_convolutional():

    BATCH_SIZE = 1000

    # Test discrete action space model has the right input and output shape
    state_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(n=3)

    agent = Agent(state_space, action_space, conv_net=1, joint_net=1, device="cpu")

    states = torch.tensor(np.array([state_space.sample() for _ in range(BATCH_SIZE)], dtype=np.float32))

    actions, log_probs, values, entropy = agent.get_actions_and_values(states, actions=None)

    assert actions.size(0) == BATCH_SIZE
    assert actions.dtype == torch.int32
    assert all_in_range(actions, low=0, high=2)

    assert log_probs.size(0) == BATCH_SIZE
    assert log_probs.dtype == torch.float32

    assert values.size(0) == BATCH_SIZE
    assert values.dtype == torch.float32

    assert entropy.size() == torch.Size([])
    assert entropy.dtype == torch.float32

    # Test the continuous action space model has the right input and output shape
    state_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(low=-2.0, high=4.0, shape=(6,))

    agent = Agent(state_space, action_space, conv_net=1, joint_net=1, device="cpu")

    states = torch.tensor(np.array([state_space.sample() for _ in range(BATCH_SIZE)], dtype=np.float32))

    actions, log_probs, values, entropy = agent.get_actions_and_values(states, actions=None)

    assert actions.size(0) == BATCH_SIZE
    assert actions.size(1) == 6
    assert actions.dtype == torch.float32
    assert all_in_range(actions, low=-2.0, high=4.0)

    assert log_probs.size(0) == BATCH_SIZE
    assert log_probs.dtype == torch.float32

    assert values.size(0) == BATCH_SIZE
    assert values.dtype == torch.float32

    assert entropy.size() == torch.Size([])
    assert entropy.dtype == torch.float32

