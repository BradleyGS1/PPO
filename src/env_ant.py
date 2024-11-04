

import gymnasium as gym


def train_fn():
    env = gym.make("Ant-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    return env

