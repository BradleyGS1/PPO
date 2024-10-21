

import gymnasium as gym


def train_fn():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    return env

