

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ray
import wandb
import time

class Agent(nn.Module):
    def __init__(
        self,
        state_shape: tuple,
        action_shape: tuple,
        conv_net: bool,
        joint_net: bool,
    ):
        super(Agent, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.conv_net = conv_net
        self.joint_net = joint_net

        if conv_net:
            self.init_conv_net()
        else:
            self.init_dense_net()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

    def init_layer(self, layer, std: float=np.sqrt(2)):
        nn.init.orthogonal_(layer.weight, std)
        return layer

    def init_conv_net(self):
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Conv2d(self.state_shape[-1], 32, 8, stride=4)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        if self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
        self.policy = self.init_layer(nn.Linear(512, self.action_shape[0]), std=0.01)
        self.critic = self.init_layer(nn.Linear(512, 1), std=1)

    def init_dense_net(self):
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Linear(self.state_shape[0], 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh()
        )
        if self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Linear(4, 64)),
                nn.Tanh(),
                self.init_layer(nn.Linear(64, 64)),
                nn.Tanh()
            )
        self.policy = self.init_layer(nn.Linear(64, self.action_shape[0]), std=0.01)
        self.critic = self.init_layer(nn.Linear(64, 1), std=1)

    def get_values(
        self,
        states: torch.Tensor
    ) -> torch.Tensor:

        if self.joint_net:
            hidden = self.pi_backbone(states)
            values = self.critic(hidden)
        else:
            hidden = self.va_backbone(states)
            values = self.critic(hidden)

        return values

    def get_actions_and_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.joint_net:
            hidden = self.pi_backbone(states)
            logits = self.policy(hidden)
            values = self.critic(hidden)
        else:
            pi_hidden = self.pi_backbone(states)
            va_hidden = self.va_backbone(states)
            logits = self.policy(pi_hidden)
            values = self.critic(va_hidden)

        action_dists = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = action_dists.sample().to(dtype=torch.int32)

        log_probs = action_dists.log_prob(actions)
        values = values.flatten()
        entropy = action_dists.entropy().mean()

        return actions, log_probs, values, entropy

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
        return 1

    def step(self, action: np.ndarray):
        self.state, reward, done, trunc, info = self.env.step(action)
        self.state = np.float32(self.state)
        reward = np.float32(reward)
        done = np.int32(done)
        trunc = np.int32(trunc)
        return reward, done, trunc, info

class VecEnv:
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
                ep_return = info["episode"]["r"]
                roll_ep_returns = self.rolling_ep_returns[i]

                if len(roll_ep_returns) == 10:
                    roll_ep_returns.pop(0)
                roll_ep_returns.append(ep_return)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.int32, device=self.device)
        truncs = torch.tensor(truncs, dtype=torch.int32, device=self.device)

        return rewards, dones, truncs

    def rollout(self):
        data_size = self.steps_per_env * self.num_envs
        self.states = torch.zeros(size=(data_size, *self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(size=(data_size, ), dtype=self.action_dtype, device=self.device)
        self.rewards = torch.zeros(size=(data_size, ), dtype=torch.float32, device=self.device)
        self.done_flags = torch.zeros(size=(data_size, ), dtype=torch.int32, device=self.device)
        self.trunc_flags = torch.zeros(size=(data_size, ), dtype=torch.int32, device=self.device)
        self.values = torch.zeros(size=(data_size, ), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(size=(data_size, ), dtype=torch.float32, device=self.device)

        end_states_unordered = []
        end_states_ordering = []

        ep_counts = np.zeros(shape=(self.num_envs,), dtype=np.int32)
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

                state = t_states[actor]
                action = t_actions[actor]
                reward = t_rewards[actor]
                done = t_dones[actor]
                trunc = t_truncs[actor]
                value = t_values[actor]
                log_prob = t_log_probs[actor]

                self.total_return += reward.cpu().numpy()

                can_reset = True
                if t_step == self.steps_per_env - 1:
                    can_reset = False
                    trunc += 1

                if done + trunc > 0:
                    end_state = torch.tensor(ray.get(env.get_state.remote()), dtype=torch.float32, device="cpu")
                    end_states_unordered.append(end_state)

                    ep_count = ep_counts[actor]
                    end_state_order = np.array([actor, ep_count], dtype=np.int32)
                    end_states_ordering.append(end_state_order)

                    if can_reset:
                        ray.get(env.reset.remote())

                    ep_counts[actor] += 1

                data_idx = t_step + self.steps_per_env * actor
                self.states[data_idx] = state
                self.actions[data_idx] = action
                self.rewards[data_idx] = reward
                self.done_flags[data_idx] = done
                self.trunc_flags[data_idx] = trunc
                self.values[data_idx] = value
                self.log_probs[data_idx] = log_prob

        self.states.requires_grad = True

        num_ends = np.sum(ep_counts).item()
        cum_counts = np.cumsum(ep_counts) - ep_counts
        self.end_states = torch.zeros(size=(num_ends, *self.state_shape), dtype=torch.float32, device=self.device)

        for i in range(num_ends):
            end_state = end_states_unordered[i]
            actor = end_states_ordering[i][0]
            order = end_states_ordering[i][1]

            end_idx = cum_counts[actor] + order
            self.end_states[end_idx] = end_state

        ep_returns_stack = np.hstack(self.rolling_ep_returns)
        if len(ep_returns_stack) > 0:
            self.mean_ep_return = np.mean(ep_returns_stack, dtype=np.float32)
            self.lower_ep_return, self.upper_ep_return = np.percentile(
                ep_returns_stack, [5.0, 95.0])

class PPO:
    def __init__(
        self,
        discount_factor: float,
        gae_factor: float,
        norm_adv: bool,
        clip_va_loss: bool,
        conv_net: bool,
        joint_network: bool,
        **kwargs
    ):
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.norm_adv = norm_adv
        self.clip_va_loss = clip_va_loss
        self.conv_net = conv_net
        self.joint_network = joint_network

        self.agent: Agent = None
        self.pi_optimizer: optim.optimizer.Optimizer = None
        self.va_optimizer: optim.optimizer.Optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def episode_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ):
        ep_size = rewards.size(0)

        td_residuals = rewards + self.discount_factor * values[1:] - values[:-1]
        td_residuals = td_residuals.cpu()

        ep_advantages = torch.zeros(size=(ep_size,), dtype=torch.float32, device="cpu")
        ep_advantages[-1] = td_residuals[-1]

        gae_factor = self.gae_factor
        discount_factor = self.discount_factor

        for i in range(ep_size-2, -1, -1):
            ep_advantages[i] = td_residuals[i] + gae_factor * discount_factor * ep_advantages[i+1]

        ep_advantages = ep_advantages.to(self.device)
        return ep_advantages

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        end_values: torch.Tensor,
        done_flags: torch.Tensor,
        trunc_flags: torch.Tensor
    ):
        num_rewards = rewards.size(0)
        advantages = torch.zeros(size=(num_rewards,), dtype=torch.float32, device=self.device)

        num_advantages = 0
        ep_start_idx = 0
        ep_count = 0
        for i in range(num_rewards):
            done = done_flags[i]
            trunc = trunc_flags[i]

            if done + trunc > 0:
                end_value = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                if trunc > 0:
                    end_value += end_values[ep_count]

                ep_rewards = rewards[ep_start_idx:i+1]
                ep_values = torch.concat([values[ep_start_idx:i+1], end_value])

                ep_size = ep_rewards.size(0)
                ep_advantages = self.episode_advantages(ep_rewards, ep_values)

                for j in range(ep_size):
                    advantages[num_advantages] = ep_advantages[j]
                    num_advantages += 1

                ep_start_idx = i + 1
                ep_count += 1

        return advantages

    def compute_losses(
        self,
        prob_ratios: torch.Tensor,
        curr_values: torch.Tensor,
        prev_values: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float
    ):
        returns = advantages + prev_values

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clipped_ratios = torch.clamp(prob_ratios, 1-clip_ratio, 1+clip_ratio)
        weighted_advantages = prob_ratios * advantages
        clipped_advantages = clipped_ratios * advantages

        # Policy loss, note the negative sign infront for gradient ascent
        policy_loss = -1.0 * torch.minimum(weighted_advantages, clipped_advantages).mean()

        # Critic loss, with or without value function loss clipping. Andrychowicz, et al. (2021) suggests value
        # function loss clipping actually hurts performance.
        if self.clip_va_loss:
            squared_error = (returns - curr_values) ** 0.5
            clipped_values = torch.clamp(curr_values, prev_values-clip_ratio, prev_values+clip_ratio)
            clipped_error = (returns - clipped_values) ** 0.5
            value_loss = 0.5 * torch.maximum(squared_error, clipped_error).mean()
        else:
            value_loss = 0.5 * ((returns - curr_values) ** 0.5).mean()

        with torch.no_grad():
            # Compute proportion of data that gets clipped
            clip_frac = ((prob_ratios - 1).abs() > clip_ratio).mean()

            # Compute KL Divergence approximation http://joschu.net/blog/kl-approx.html
            kl_div = (prob_ratios - 1 - prob_ratios.log()).mean()

        return policy_loss, value_loss, clip_frac, kl_div

    def train_step(
        self,
        vec_env: VecEnv,
        num_epochs: int,
        batch_size: int,
        critic_coef: float,
        entropy_coef: float,
        clip_ratio: float,
        target_div: float,
        max_grad_norm: float,
        lr_anneal: float
    ):
        # Run the model to get training data
        num_envs = vec_env.num_envs
        steps_per_env = vec_env.steps_per_env

        rollout_start = time.time()
        vec_env.rollout()

        rollout_time = time.time() - rollout_start
        env_steps_per_sec = steps_per_env * num_envs / rollout_time

        # Calculate critic values for termination states
        end_states = vec_env.end_states
        with torch.no_grad():
            end_values = self.agent.get_values(end_states)

        # Calculate the advantages
        rewards = vec_env.rewards
        values = vec_env.values
        done_flags = vec_env.done_flags
        trunc_flags = vec_env.trunc_flags
        advantages = self.compute_advantages(
            rewards, values, end_values, done_flags, trunc_flags)

        # Perform multiple epochs of updates using the gathered data
        updates_start = time.time()
        clip_fracs = []
        grad_steps_done = 0
        for epoch in range(num_epochs):
            data_size = steps_per_env * num_envs
            shuffled_indices = torch.randperm(data_size)

            num_batches = torch.ceil(data_size / batch_size).item()
            for start in range(0, data_size, num_batches):

                # Gather a shuffled minibatch from the total data batch
                end = torch.minimum(data_size, start + batch_size).item()
                mb_indices = shuffled_indices[start:end]

                mb_states = vec_env.states[mb_indices]
                mb_actions = vec_env.actions[mb_indices]
                mb_log_probs = vec_env.log_probs[mb_indices]
                mb_values = vec_env.values[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Compute the new log probs, critic values and entropy using minibatch states and actions
                _, new_log_probs, new_values, entropy = self.agent.get_actions_and_values(
                    mb_states, mb_actions)

                # Compute the minibatch probability ratios
                mb_prob_ratios = torch.exp(new_log_probs - mb_log_probs)

                # Compute the losses required for agent updates
                policy_loss, critic_loss, clip_frac, kl_div = self.compute_losses(
                    mb_prob_ratios, new_values, mb_values, mb_advantages, clip_ratio)

                # Perform gradient update
                if self.joint_network:
                    total_loss = policy_loss + critic_coef * critic_loss - entropy_coef * entropy
                    total_loss = total_loss * lr_anneal

                    self.pi_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.pi_optimizer.step()
                else:
                    policy_loss = policy_loss - entropy_coef * entropy
                    policy_loss = policy_loss * lr_anneal

                    self.pi_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.pi_optimizer.step()

                    critic_loss = critic_coef * critic_loss
                    critic_loss = critic_loss * lr_anneal

                    self.va_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.va_optimizer.step()

                clip_fracs.append(clip_frac.item())
                grad_steps_done += 1

            # Early stop the updates if target_kl threshold exceeded
            if kl_div > target_div:
                break

        updates_time = time.time() - updates_start
        grad_steps_per_sec = grad_steps_done / updates_time

        policy_loss = policy_loss.item()
        critic_loss = critic_loss.item()
        entropy = entropy.item()
        clip_frac = np.mean(clip_fracs, dtype=np.float32)
        kl_div = kl_div.item()

        rollout_return = vec_env.total_return.item() / num_envs
        ep_return_mean = vec_env.mean_ep_return.item()
        ep_return_lower = vec_env.lower_ep_return.item()
        ep_return_upper = vec_env.upper_ep_return.item()

        if wandb.run is not None:
            wandb.log({
                "utils/ep_return_mean": ep_return_mean,
                "utils/ep_return_lower": ep_return_lower,
                "utils/ep_return_upper": ep_return_upper,
                "utils/env_steps_per_sec": env_steps_per_sec,
                "utils/grad_steps_per_sec": grad_steps_per_sec,
                "metrics/policy_loss": policy_loss,
                "metrics/critic_loss": critic_loss,
                "metrics/entropy": entropy,
                "metrics/clip_frac": clip_frac,
                "metrics/kl_div": kl_div,
                "metrics/ep_return": wandb.Table(
                    data=[[ep_return_mean, ep_return_lower, ep_return_upper]],
                    columns=["episode_return", "ep_return_0.05", "ep_return_0.95"]
                )
            }, step=steps_per_env*num_envs)

        return policy_loss, critic_loss, entropy, clip_frac, kl_div, rollout_return, ep_return_mean
