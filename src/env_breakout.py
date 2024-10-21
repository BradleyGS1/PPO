

import gymnasium as gym
import numpy as np
from collections import deque


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        gym.Wrapper.__init__(self, env)

        self.num_stack = num_stack
        self.frame_stack = None

    def reset(self):
        state, info = self.env.reset()
        self.frame_stack = deque([state for _ in range(self.num_stack)], maxlen=self.num_stack)
        return np.concatenate(self.frame_stack, axis=-1), info

    def step(self, action):
        new_state, reward, done, trunc, info = self.env.step(action)
        reward = np.sign(reward)
        self.frame_stack.popleft()
        self.frame_stack.append(new_state)
        return np.concatenate(self.frame_stack, axis=-1), reward, done, trunc, info

def train_fn():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=10)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, grayscale_newaxis=True, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

