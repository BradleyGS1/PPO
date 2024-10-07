

import gymnasium as gym
import numpy as np
import torch
import ray

from ppo import Agent


@ray.remote(num_cpus=1, num_gpus=0)
class ActorEnv:
    def __init__(
        self,
        env_fn: callable,
        actor_id: int
    ):
        self.env: gym.Env = env_fn()
        self.actor_id = actor_id
        self.reset()

    def get_spaces(self):
        return self.state.shape, self.env.action_space

    def get_state(self):
        return self.state

    def reset(self):
        self.state, _ = self.env.reset()
        self.state = np.float32(self.state)
        if len(self.state.shape) == 3:
            self.state = np.permute_dims(self.state, axes=(2, 0, 1))
        return 1

    def step(self, action: np.ndarray):
        action = np.squeeze(action)
        self.state, reward, done, trunc, info = self.env.step(action)
        self.state = np.float32(self.state)
        if len(self.state.shape) == 3:
            self.state = np.permute_dims(self.state, axes=(2, 0, 1))
        reward = np.float32(reward)
        done = np.int32(done)
        trunc = np.int32(trunc)
        return reward, done, trunc, info

class RayVecEnv:
    def __init__(
        self,
        env_fn: callable,
        num_envs: int,
        steps_per_env: int,
        agent: Agent
    ):
        self.num_envs = num_envs
        self.envs = [ActorEnv.remote(env_fn, agent_id) for agent_id in range(num_envs)]
        self.steps_per_env = steps_per_env
        self.agent = agent

        self.rolling_ep_returns = [[] for _ in range(num_envs)]
        self.mean_ep_return = np.float32(np.nan)
        self.lower_ep_return = np.float32(np.nan)
        self.upper_ep_return = np.float32(np.nan)

        self.state_shape, self.action_space = ray.get(self.envs[0].get_spaces.remote())
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dtype = torch.int32
        else:
            self.action_dtype = torch.float32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def vec_env_states(self):
        states = ray.get([env.get_state.remote() for env in self.envs])
        states = torch.tensor(np.stack(states, axis=0), dtype=torch.float32, device=self.device)
        return states

    def vec_env_step(self, actions: torch.Tensor):
        rewards = np.zeros(shape=(self.num_envs,), dtype=np.float32)
        dones = np.zeros(shape=(self.num_envs,), dtype=np.int32)
        truncs = np.zeros(shape=(self.num_envs,), dtype=np.int32)

        step_processes = []
        for env, action in zip(self.envs, actions.cpu()):
            step_processes.append(env.step.remote(action.numpy()))

        step_data = ray.get(step_processes)
        for i, actor_data in enumerate(step_data):
            reward, done, trunc, info = actor_data
            rewards[i] = reward
            dones[i] = done
            truncs[i] = trunc

            if "episode" in info:
                ep_return = info["episode"]["r"].item()
                roll_ep_returns = self.rolling_ep_returns[i]

                if len(roll_ep_returns) == 10:
                    roll_ep_returns.pop(0)
                roll_ep_returns.append(ep_return)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.int32, device=self.device)
        truncs = torch.tensor(truncs, dtype=torch.int32, device=self.device)

        return rewards, dones, truncs

    def rollout(self):
        num_steps = self.steps_per_env
        num_envs = self.num_envs

        self.states = torch.zeros(size=(num_steps, num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(size=(num_steps, num_envs, ), dtype=self.action_dtype, device=self.device)
        self.rewards = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
        self.done_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
        self.trunc_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
        self.values = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)

        end_states = [[] for _ in range(num_envs)]
        self.ep_counts = torch.zeros(size=(self.num_envs,), dtype=torch.int32, device=self.device)
        self.total_return = np.float32(0.0)

        for t_step in range(self.steps_per_env):
            # Get states from all envs
            t_states = self.vec_env_states()

            # Compute actions, log_probs and critic values for each env using their states
            with torch.no_grad():
                t_actions, t_log_probs, t_values, _  = self.agent.get_actions_and_values(t_states, actions=None)

            # Perform a vector env step using the sampled actions
            t_rewards, t_dones, t_truncs = self.vec_env_step(t_actions)

            for actor in range(self.num_envs):
                env = self.envs[actor]

                reward = t_rewards[actor]
                done = t_dones[actor]
                trunc = t_truncs[actor]
                terminated = done + trunc

                self.total_return += reward.cpu().numpy()

                can_reset = True
                if terminated == 0 and t_step == self.steps_per_env - 1:
                    terminated += 1
                    t_truncs[actor] += 1
                    can_reset = False

                if terminated > 0:
                    end_state = torch.tensor(ray.get(env.get_state.remote()), dtype=torch.float32, device="cpu")
                    end_states[actor].append(end_state)

                    if can_reset:
                        ray.get(env.reset.remote())

                    self.ep_counts[actor] += 1

            self.states[t_step] = t_states
            self.actions[t_step] = t_actions
            self.rewards[t_step] = t_rewards
            self.done_flags[t_step] = t_dones
            self.trunc_flags[t_step] = t_truncs
            self.values[t_step] = t_values
            self.log_probs[t_step] = t_log_probs

        self.states.requires_grad = True
        end_states_tensors = [torch.stack(actor_end_states, dim=0) for actor_end_states in end_states]
        self.end_states = torch.concatenate(end_states_tensors, dim=0).to(device=self.device)

        ep_returns_stack = np.concatenate(self.rolling_ep_returns)
        if len(ep_returns_stack) > 0:
            self.mean_ep_return = np.mean(ep_returns_stack, dtype=np.float32)
            self.lower_ep_return, self.upper_ep_return = np.percentile(
                ep_returns_stack, [5.0, 95.0])