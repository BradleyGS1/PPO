

import gymnasium as gym


def train_fn():
    env = gym.make("Acrobot-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    return env

