

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ppo import Agent


def test_joint_dense():
    agent = Agent(state_shape=(6,), action_shape=(2,), conv_net=False, joint_net=True)
    states = torch.normal(mean=0.5, std=1.0, size=(6,))
    values = agent.get_values(states)

    assert values.size() == torch.Size([1])


    agent = Agent(state_shape=(4,), action_shape=(4,), conv_net=False, joint_net=True)
    states = torch.normal(mean=1.5, std=0.2, size=(2, 4))
    values = agent.get_values(states)

    assert values.size() == torch.Size([2, 1])


    agent = Agent(state_shape=(8,), action_shape=(6,), conv_net=False, joint_net=True)
    states = torch.normal(mean=0.0, std=1.0, size=(4, 8))
    actions, log_probs, values, entropy = agent.get_actions_and_values(states, actions=None)

    assert actions.size() == torch.Size([4])
    assert log_probs.size() == torch.Size([4])
    assert values.size() == torch.Size([4])
    assert entropy.size() == torch.Size([])


    agent = Agent(state_shape=(2,), action_shape=(2,), conv_net=False, joint_net=True)
    states = torch.normal(mean=0.0, std=1.0, size=(25, 2))
    actions = torch.randint(low=0, high=2, size=(25,))
    actions_result, log_probs, values, entropy = agent.get_actions_and_values(states, actions=actions)

    assert actions.size() == torch.Size([25])
    assert torch.max(actions) == 1
    assert torch.equal(actions, actions_result)
    assert log_probs.size() == torch.Size([25])
    assert values.size() == torch.Size([25])
    assert entropy.size() == torch.Size([])
    assert (entropy - 0.6930).abs() < 0.01

def main():
    test_joint_dense()

main()