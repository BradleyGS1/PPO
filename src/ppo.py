

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb
import time
import tqdm
import os

from typing import Callable
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


class ScaledBeta(torch.distributions.Distribution):
    def __init__(
        self,
        modes: torch.Tensor,
        precisions: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor
    ):
        super(ScaledBeta, self).__init__(validate_args=False)

        self.modes = modes
        self.precisions = precisions
        self.lows = lows
        self.highs = highs

        self.alphas = modes * precisions + 1.0
        self.betas = precisions + 2.0 - self.alphas

        self.diffs = highs - lows
        self.log_diffs = torch.log(highs - lows)
        self.epsilon = 1e-6

        self.unit_beta_dist = torch.distributions.Beta(self.alphas, self.betas)

    def sample(self):
        unit_variates = self.unit_beta_dist.sample()
        scaled_variates = unit_variates * self.diffs + self.lows
        return scaled_variates

    def log_prob(self, variates: torch.Tensor):
        unit_variates = (variates - self.lows) / (self.diffs + self.epsilon)
        unit_log_probs = self.unit_beta_dist.log_prob(unit_variates)
        scaled_log_probs = unit_log_probs - self.log_diffs
        return torch.sum(scaled_log_probs, dim=1)

    def entropy(self):
        unit_entropy = self.unit_beta_dist.entropy()
        scaled_entropy = unit_entropy + self.log_diffs
        return torch.sum(scaled_entropy, dim=1)

class Agent(nn.Module):
    def __init__(
        self,
        state_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        conv_net: bool,
        joint_net: bool,
        device: str
    ):
        super(Agent, self).__init__()

        self.action_space = action_space
        self.state_space = state_space
        self.conv_net = conv_net
        self.joint_net = joint_net

        if len(state_space.shape) < 3:
            self.permute_states_fn = lambda x: x
        else:
            self.permute_states_fn = self.permute_states

        if conv_net:
            self.init_conv_net()
        else:
            self.init_dense_net()

        self.device = device
        self.to(self.device, dtype=torch.float32)

    def init_layer(self, layer, std: float=np.sqrt(2)):
        nn.init.orthogonal_(layer.weight, std)
        return layer

    def init_conv_net(self):
        state_shape = self.state_space.shape
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Conv2d(state_shape[-1], 32, 8, stride=4)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        if not self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Conv2d(state_shape[-1], 32, 8, stride=4)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.policy = self.init_layer(nn.Linear(512, self.action_space.n), std=0.01)

        elif isinstance(self.action_space, gym.spaces.Box):
            self.policy = self.init_layer(nn.Linear(512, 2 * self.action_space.shape[0]), std=0.01)

        self.critic = self.init_layer(nn.Linear(512, 1), std=1.0)

    def init_dense_net(self):
        state_shape = self.state_space.shape
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Linear(state_shape[0], 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh()
        )
        if not self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Linear(state_shape[0], 64)),
                nn.Tanh(),
                self.init_layer(nn.Linear(64, 64)),
                nn.Tanh()
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.policy = self.init_layer(nn.Linear(64, self.action_space.n), std=0.01)

        elif isinstance(self.action_space, gym.spaces.Box):
            self.policy = self.init_layer(nn.Linear(64, 2 * self.action_space.shape[0]), std=0.01)

        self.critic = self.init_layer(nn.Linear(64, 1), std=1.0)

    def permute_states(self, states: torch.Tensor):
        return torch.permute(states, (0, 3, 1, 2))

    def get_values(
        self,
        states: torch.Tensor
    ) -> torch.Tensor:

        states = self.permute_states_fn(states)

        if self.joint_net:
            hidden = self.pi_backbone(states)
            values = self.critic(hidden)
        else:
            hidden = self.va_backbone(states)
            values = self.critic(hidden)

        return torch.flatten(values)

    def get_actions_and_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        states = self.permute_states_fn(states)

        if self.joint_net:
            hidden = self.pi_backbone(states)
            policy_output = self.policy(hidden)
            values = self.critic(hidden)
        else:
            pi_hidden = self.pi_backbone(states)
            va_hidden = self.va_backbone(states)
            policy_output = self.policy(pi_hidden)
            values = self.critic(va_hidden)

        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = policy_output
            action_dist = torch.distributions.Categorical(logits=logits)

            if actions is None:
                actions = action_dist.sample().to(dtype=torch.int32)

        elif isinstance(self.action_space, gym.spaces.Box):
            n = self.action_space.shape[0]

            lows = self.action_space.low
            highs = self.action_space.high
            if not isinstance(lows, np.ndarray):
                lows = lows * np.ones((n,), dtype=np.float32)
                highs = highs * np.ones((n,), dtype=np.float32)

            lows = torch.tensor(lows)
            highs = torch.tensor(highs)

            # Using a beta distribution with a, b >= 1 so it is unimodal
            modes = 0.5 * (torch.clip(policy_output[:, :n], min=-1.0, max=1.0) + 1) # Modes of beta distribution: (a - 1) / (a + b - 2)
            precisions = torch.exp(policy_output[:, n:])                            # Precisions of beta distribution: a + b - 2

            action_dist = ScaledBeta(modes, precisions, lows, highs)

            if actions is None:
                actions = action_dist.sample().to(dtype=torch.float32)

        log_probs = action_dist.log_prob(actions)
        values = values.flatten()
        entropy = action_dist.entropy().mean()

        return actions, log_probs, values, entropy

class SyncVecEnv:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        steps_per_env: int,
        render_every: int,
        render_fps: float,
        agent: Agent
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.global_steps = 0

        self.render_every = (render_every if render_every > 0 else 1)
        self.can_record = render_every > 0
        self.ready_to_record = False
        self.is_recording = self.can_record
        self.record_episode = 0
        self.record_buffer = []
        self.record_total_reward = 0.0
        self.render_fps = render_fps
        self.render_folder = "./renders/misc"
        if self.can_record and wandb.run is not None:
            project_name = wandb.run.project
            run_name = wandb.run.name
            self.render_folder = f"./renders/{project_name}/{run_name}"
        os.makedirs(self.render_folder, exist_ok=True)

        self.agent = agent

        self.max_ep_return = np.float32(np.nan)
        self.lower_ep_return = np.float32(np.nan)
        self.median_ep_return = np.float32(np.nan)
        self.upper_ep_return = np.float32(np.nan)
        self.median_ep_length = np.float32(np.nan)

        self.state_space = agent.state_space
        self.action_space = agent.action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dtype = torch.int32
        else:
            self.action_dtype = torch.float32

        self.device = self.agent.device
        self.t_states = self.vec_reset()

    def close(self):
        self.envs = None

    def vec_reset(self) -> torch.Tensor:
        states = torch.zeros(size=(self.num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
        for i, env in enumerate(self.envs):
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            states[i] = state
        return states

    def env_reset(self, env_id: int):
        env = self.envs[env_id]
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state

    def vec_step(
        self,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        actions = actions.cpu().numpy()

        states = torch.zeros(size=(self.num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
        rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device=self.device)
        done_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        trunc_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        for i, env in enumerate(self.envs):
            action = np.squeeze(actions[i])
            state, reward, done, trunc, info = env.step(action)
            states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
            rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_flags[i] = torch.tensor(done, dtype=torch.int32, device=self.device)
            trunc_flags[i] = torch.tensor(trunc, dtype=torch.int32, device=self.device)

        return states, rewards, done_flags, trunc_flags

    def rollout(self):
        with torch.no_grad():
            num_steps = self.steps_per_env
            num_envs = self.num_envs

            self.states = torch.zeros(size=(num_steps, num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros(size=(num_steps, num_envs, *self.action_space.shape), dtype=self.action_dtype, device=self.device)
            self.rewards = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.done_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.trunc_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.values = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.log_probs = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)

            end_states = [[] for _ in range(num_envs)]
            self.total_return = np.float32(0.0)

            for t_step in range(num_steps):

                # Record rendered observation from the first environment if recording
                if self.is_recording:
                    obs_render = self.envs[0].render().astype(np.uint8)
                    obs_image = Image.fromarray(obs_render)

                    draw = ImageDraw.Draw(obs_image)
                    font = ImageFont.load_default()
                    text = f"Total Reward: {self.record_total_reward}"
                    position = (50, 40)
                    text_color = (0, 204, 102)
                    draw.text(position, text, text_color, font)

                    self.record_buffer.append(obs_image) 

                # Compute actions, log_probs and critic values for each env using their states
                t_actions, t_log_probs, t_values, _  = self.agent.get_actions_and_values(self.t_states, actions=None)

                # Perform a vector env step using the sampled actions
                t_new_states, t_rewards, t_dones, t_truncs = self.vec_step(t_actions)

                for actor in range(self.num_envs):
                    reward = t_rewards[actor]
                    done = t_dones[actor]
                    trunc = t_truncs[actor]
                    terminated = done + trunc

                    self.total_return += reward.cpu().numpy()
                    if actor == 0:
                        self.record_total_reward += reward.cpu().numpy()

                    can_reset = True
                    if terminated == 0 and t_step == num_steps - 1:
                        terminated += 1
                        t_truncs[actor] += 1
                        can_reset = False

                    elif terminated > 0 and actor == 0:
                        if self.is_recording:
                            self.is_recording = False
                            if len(self.record_buffer) > 1:
                                self.record_buffer[0].save(
                                        f"{self.render_folder}/render_{self.record_episode}.gif", 
                                        save_all=True, 
                                        append_images=self.record_buffer[1:], 
                                        duration=1000/self.render_fps, 
                                        loop=0)

                            self.record_buffer = []
                            self.record_episode += 1

                        elif self.ready_to_record:
                            self.ready_to_record = False
                            self.is_recording = True
                            self.record_total_reward = 0.0

                    if terminated > 0:
                        end_state = t_new_states[actor]
                        end_states[actor].append(end_state.cpu())

                        if can_reset:
                            t_new_states[actor] = self.env_reset(actor)

                    self.global_steps += 1

                    ready_to_record = self.global_steps % self.render_every == self.render_every - 1
                    if self.can_record and ready_to_record:
                        self.ready_to_record = True

                self.states[t_step] = self.t_states
                self.actions[t_step] = t_actions
                self.rewards[t_step] = t_rewards
                self.done_flags[t_step] = t_dones
                self.trunc_flags[t_step] = t_truncs
                self.values[t_step] = t_values
                self.log_probs[t_step] = t_log_probs

                self.t_states = t_new_states

        self.states.requires_grad = True
        end_states_tensors = [torch.stack(actor_end_states, dim=0) for actor_end_states in end_states]
        self.end_states = torch.concatenate(end_states_tensors, dim=0).to(device=self.device)

        if not hasattr(self.envs[0], "return_queue"):
            return

        ep_returns_stack = np.concatenate([np.array(env.return_queue).reshape(-1) for env in self.envs])
        if len(ep_returns_stack) > 0:
            if np.isnan(self.max_ep_return) or np.max(ep_returns_stack) > self.max_ep_return:
                self.max_ep_return = np.max(ep_returns_stack)

            self.lower_ep_return, self.median_ep_return, self.upper_ep_return = np.percentile(
                ep_returns_stack, [5.0, 50.0, 95.0])

        ep_lengths_stack = np.concatenate([np.array(env.length_queue).reshape(-1) for env in self.envs])
        if len(ep_lengths_stack) > 0:
            self.median_ep_length = np.percentile(ep_lengths_stack, 50.0)


class PPO:
    def __init__(
        self,
        discount_factor: float,
        gae_factor: float,
        norm_adv: bool,
        clip_va_loss: bool,
        conv_net: bool,
        joint_network: bool,
        use_gpu: bool,
        **kwargs
    ):
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.norm_adv = norm_adv
        self.clip_va_loss = clip_va_loss
        self.conv_net = conv_net
        self.joint_network = joint_network
        self.use_gpu = use_gpu

        self.project_name = kwargs.get("project_name", None)

        self.agent: Agent = None
        self.pi_optimizer: optim.optimizer.Optimizer = None
        self.va_optimizer: optim.optimizer.Optimizer = None

        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"

        self.updates = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        end_values: torch.Tensor,
        done_flags: torch.Tensor,
        trunc_flags: torch.Tensor
    ):
        num_steps = rewards.size(0)
        num_agents = rewards.size(1)

        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)

        discount_factor = self.discount_factor
        gae_factor = self.gae_factor

        ep_counts = torch.sum(done_flags + trunc_flags - done_flags * trunc_flags, dim=0, dtype=torch.int32)
        end_indices = torch.cumsum(ep_counts, dim=0) - 1
        next_values = torch.zeros(size=(num_agents,), dtype=torch.float32, device=self.device)
        next_advantages = torch.zeros(size=(num_agents,), dtype=torch.float32, device=self.device)

        for t in reversed(range(num_steps)):
            dones = done_flags[t]
            truncs = trunc_flags[t]
            terminations = dones + truncs

            next_values = (1 - terminations) * next_values + truncs * end_values[end_indices]
            next_advantages = (1 - terminations) * next_advantages
            end_indices = end_indices - terminations

            td_residuals = rewards[t] + discount_factor * next_values - values[t]
            advantages[t] = td_residuals + discount_factor * gae_factor * next_advantages

            next_values = values[t]
            next_advantages = advantages[t]

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
            squared_error = (returns - curr_values) ** 2
            clipped_values = torch.clamp(curr_values, prev_values-clip_ratio, prev_values+clip_ratio)
            clipped_error = (returns - clipped_values) ** 2
            value_loss = 0.5 * torch.maximum(squared_error, clipped_error).mean()
        else:
            value_loss = 0.5 * ((returns - curr_values) ** 2).mean()

        with torch.no_grad():
            # Compute proportion of data that gets clipped
            clip_frac = ((prob_ratios - 1).abs() > clip_ratio).float().mean()

            # Compute KL Divergence approximation http://joschu.net/blog/kl-approx.html
            kl_div = (prob_ratios - 1 - prob_ratios.log()).mean()

        return policy_loss, value_loss, clip_frac, kl_div

    def train_step(
        self,
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
        num_envs = self.vec_env.num_envs
        steps_per_env = self.vec_env.steps_per_env

        rollout_start = time.time()
        self.vec_env.rollout()

        # Calculate the advantages
        rewards = self.vec_env.rewards
        values = self.vec_env.values
        end_states = self.vec_env.end_states
        done_flags = self.vec_env.done_flags
        trunc_flags = self.vec_env.trunc_flags

        with torch.no_grad():
            end_values = self.agent.get_values(end_states)
            advantages = self.compute_advantages(
                rewards, values, end_values, done_flags, trunc_flags)

        rollout_time = time.time() - rollout_start
        env_steps_per_sec = steps_per_env * num_envs / rollout_time

        states = self.vec_env.states.flatten(0, 1)
        actions = self.vec_env.actions.flatten(0, 1)
        log_probs = self.vec_env.log_probs.flatten(0, 1)
        values = self.vec_env.values.flatten(0, 1)
        advantages = advantages.flatten(0, 1)

        # Perform multiple epochs of updates using the gathered data
        updates_start = time.time()
        early_stop_updates = False
        clip_fracs = []
        grad_steps_done = 0
        for epoch in range(num_epochs):
            data_size = steps_per_env * num_envs
            batch_indices = np.arange(data_size)
            np.random.shuffle(batch_indices)

            for start in range(0, data_size, batch_size):

                # Gather a shuffled minibatch from the total data batch
                end = np.minimum(data_size, start + batch_size).item()
                mb_indices = batch_indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs = log_probs[mb_indices]
                mb_values = values[mb_indices]
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
                    full_policy_loss = policy_loss - entropy_coef * entropy
                    full_policy_loss = full_policy_loss * lr_anneal

                    self.pi_optimizer.zero_grad()
                    full_policy_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.pi_optimizer.step()

                    critic_loss = critic_coef * critic_loss
                    full_critic_loss = critic_loss * lr_anneal

                    self.va_optimizer.zero_grad()
                    full_critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.va_optimizer.step()

                # Set the early stop updates flag to true if target_kl threshold exceeded
                if target_div is not None and kl_div > target_div:
                    early_stop_updates = True

                clip_fracs.append(clip_frac.item())
                grad_steps_done += 1

            if early_stop_updates:
                break

        updates_time = time.time() - updates_start
        grad_steps_per_sec = grad_steps_done / updates_time
        global_steps_per_sec = steps_per_env * num_envs / (rollout_time + updates_time)

        policy_loss = policy_loss.item()
        critic_loss = critic_loss.item()
        entropy = entropy.item()
        clip_frac = np.mean(clip_fracs, dtype=np.float32).item()
        kl_div = kl_div.item()

        rollout_return = self.vec_env.total_return.item() / num_envs
        ep_return_max = self.vec_env.max_ep_return.item()
        ep_return_lower = self.vec_env.lower_ep_return.item()
        ep_return_median = self.vec_env.median_ep_return.item()
        ep_return_upper = self.vec_env.upper_ep_return.item()

        ep_length_median = self.vec_env.median_ep_length.item()

        if wandb.run is not None:
            wandb.log({
                "utils/ep_return_0.50": ep_return_median,
                "utils/ep_return_0.05": ep_return_lower,
                "utils/ep_return_0.95": ep_return_upper,
                "utils/env_steps_per_sec": env_steps_per_sec,
                "utils/grad_steps_per_sec": grad_steps_per_sec,
                "utils/global_steps_per_sec": global_steps_per_sec,
                "losses/policy_loss": policy_loss,
                "losses/critic_loss": critic_loss,
                "losses/entropy": entropy,
                "metrics/clip_frac": clip_frac,
                "metrics/kl_div": kl_div,
                "metrics/roll_return": rollout_return,
                "metrics/ep_return_max": ep_return_max,
                "metrics/ep_length_0.50": ep_length_median,
            }, step=self.updates*steps_per_env*num_envs)

        return policy_loss, critic_loss, entropy, clip_frac, kl_div, rollout_return, ep_return_max

    def train(
        self,
        env_fn: Callable[[], gym.Env],
        num_updates: int,
        num_envs: int,
        steps_per_env: int,
        num_epochs: int,
        batch_size: int,
        critic_coef: float,
        entropy_coef: float,
        clip_ratio: float,
        max_grad_norm: float,
        learning_rate: float,
        target_div: float = None,
        render_every: int = 0,
        render_fps: float = 0.0,
        early_stop_reward: float = None
    ): 
        curr_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        if self.project_name is not None:
            wandb.init(project=self.project_name, name=f"run-{curr_datetime}", reinit=True, config={
                "discount_factor": self.discount_factor,
                "gae_factor": self.gae_factor,
                "norm_adv": self.norm_adv,
                "clip_va_loss": self.clip_va_loss,
                "joint_network": self.joint_network,
                "use_gpu": self.use_gpu,
                "num_updates": num_updates,
                "num_envs": num_envs,
                "steps_per_env": steps_per_env,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "critic_coef": critic_coef,
                "entropy_coef": entropy_coef,
                "clip_ratio": clip_ratio,
                "max_grad_norm": max_grad_norm,
                "learning_rate": learning_rate,
                "target_div": target_div,
                "render_every": render_every,
                "render_fps": render_fps,
                "early_stop_reward": early_stop_reward
            })

        test_env = env_fn()
        test_env_obs_shape = test_env.reset()[0].shape

        state_space = test_env.observation_space
        if hasattr(state_space, "_shape") and test_env_obs_shape != state_space.shape:
            state_space._shape = test_env_obs_shape
        action_space = test_env.action_space

        self.agent = Agent(state_space, action_space, conv_net=self.conv_net, joint_net=self.joint_network, device=self.device)

        del test_env

        self.vec_env = SyncVecEnv(env_fn, num_envs, steps_per_env, render_every, render_fps, self.agent)

        self.pi_optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.va_optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        lr_anneal = 1.0

        pbar = tqdm.trange(num_updates, leave=True)
        for update in pbar:
            pi_loss, va_loss, entropy, clip_frac, kl_div, roll_return, ep_return_max = self.train_step(
                num_epochs, batch_size, critic_coef, entropy_coef,
                clip_ratio, target_div, max_grad_norm, lr_anneal
            )

            if wandb.run is not None:
                wandb.log({
                    "params/learning_rate": learning_rate * lr_anneal,
                }, step=self.updates*steps_per_env*num_envs, commit=True)

            lr_anneal -= 0.999 / (num_updates - 1)
            self.updates += 1

            pbar.set_postfix({
                "pi_loss": f"{pi_loss:.3f}", "va_loss": f"{va_loss:.3f}",
                "entropy": f"{entropy:.3f}", "return": f"{roll_return:.3f}",
                "max_ep_ret": f"{ep_return_max:.3f}", "kl_div": f"{kl_div:.4f}",
                "clip_frac": f"{clip_frac:.3f}"
            })

            if early_stop_reward is not None and ep_return_max >= early_stop_reward:
                pbar.close()
                print("Early stop reward reached.")
                break

        wandb.finish()

