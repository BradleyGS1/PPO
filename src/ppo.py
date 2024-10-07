

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb
import time
import tqdm

from typing import Callable
from datetime import datetime


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
        if not self.joint_net:
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
        if not self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Linear(self.state_shape[0], 64)),
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

        return torch.flatten(values)

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

class SyncVecEnv:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        steps_per_env: int,
        agent: Agent
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.agent = agent

        self.rolling_ep_returns = [[] for _ in range(num_envs)]
        self.mean_ep_return = np.float32(np.nan)
        self.lower_ep_return = np.float32(np.nan)
        self.upper_ep_return = np.float32(np.nan)

        state, _ = self.envs[0].reset()
        self.state_shape = state.shape
        self.action_space = self.envs[0].action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dtype = torch.int32
        else:
            self.action_dtype = torch.float32

        if len(self.state_shape) < 3:
            self.permute_state_fn = lambda x: x
        else:
            self.permute_state_fn = self.permute_state
            new_state_shape = (self.state_shape[2], self.state_shape[0], self.state_shape[1])
            self.state_shape = new_state_shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_states = self.vec_reset()

    def permute_state(self, state: np.ndarray):
        return np.permute_dims(state, axes=(2, 0, 1))

    def close(self):
        self.envs = None

    def vec_reset(self) -> torch.Tensor:
        states = torch.zeros(size=(self.num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
        for i, env in enumerate(self.envs):
            state = env.reset()[0]
            state = self.permute_state_fn(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            states[i] = state
        return states

    def env_reset(self, env_id: int):
        env = self.envs[env_id]
        state = env.reset()[0]
        state = self.permute_state_fn(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state

    def vec_step(
        self, 
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        actions = actions.cpu().numpy()

        states = torch.zeros(size=(self.num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
        rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device=self.device)
        done_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        trunc_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        for i, env in enumerate(self.envs):
            action = np.squeeze(actions[i])
            state, reward, done, trunc, info = env.step(action)
            state = self.permute_state_fn(state)
            states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
            rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_flags[i] = torch.tensor(done, dtype=torch.int32, device=self.device)
            trunc_flags[i] = torch.tensor(trunc, dtype=torch.int32, device=self.device)

            if "episode" in info:
                ep_return = info["episode"]["r"].item()
                roll_ep_returns = self.rolling_ep_returns[i]

                if len(roll_ep_returns) == 10:
                    roll_ep_returns.pop(0)
                roll_ep_returns.append(ep_return)

        return states, rewards, done_flags, trunc_flags

    def rollout(self):
        with torch.no_grad():
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
            self.ep_counts = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
            self.total_return = np.float32(0.0)

            for t_step in range(self.steps_per_env):

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

                    can_reset = True
                    if terminated == 0 and t_step == self.steps_per_env - 1:
                        terminated += 1
                        t_truncs[actor] += 1
                        can_reset = False

                    if terminated > 0:
                        end_state = t_new_states[actor]
                        end_states[actor].append(end_state.cpu())

                        if can_reset:
                            t_new_states[actor] = self.env_reset(actor)

                        self.ep_counts[actor] += 1

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

        ep_returns_stack = np.concatenate(self.rolling_ep_returns)
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

        self.updates = 0

    def episode_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ):
        ep_size = rewards.size(0)
        gae_factor = self.gae_factor
        discount_factor = self.discount_factor

        td_residuals = rewards + discount_factor * values[1:] - values[:-1]

        ep_advantages = torch.zeros(size=(ep_size,), dtype=torch.float32, device=self.device)
        ep_advantages[-1] = td_residuals[-1]

        for i in range(ep_size-2, -1, -1):
            ep_advantages[i] = td_residuals[i] + gae_factor * discount_factor * ep_advantages[i+1]

        return ep_advantages

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        end_states: torch.Tensor,
        ep_counts: torch.Tensor,
        done_flags: torch.Tensor,
        trunc_flags: torch.Tensor
    ):
        num_steps = rewards.size(0)
        num_agents = rewards.size(1)

        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
        end_values = self.agent.get_values(end_states)

        discount_factor = self.discount_factor
        gae_factor = self.gae_factor

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
            squared_error = (returns - curr_values) ** 0.5
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
        vec_env: SyncVecEnv,
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

        # Calculate the advantages
        rewards = vec_env.rewards
        values = vec_env.values
        end_states = vec_env.end_states
        ep_counts = vec_env.ep_counts
        done_flags = vec_env.done_flags
        trunc_flags = vec_env.trunc_flags
        with torch.no_grad():
            advantages = self.compute_advantages(
                rewards, values, end_states, ep_counts, done_flags, trunc_flags)

        rollout_time = time.time() - rollout_start
        env_steps_per_sec = steps_per_env * num_envs / rollout_time

        states = vec_env.states.flatten(0, 1)
        actions = vec_env.actions.flatten(0, 1)
        log_probs = vec_env.log_probs.flatten(0, 1)
        values = vec_env.values.flatten(0, 1)
        advantages = advantages.flatten(0, 1)

        # Perform multiple epochs of updates using the gathered data
        updates_start = time.time()
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

                clip_fracs.append(clip_frac.item())
                grad_steps_done += 1

            # Early stop the updates if target_kl threshold exceeded
            if kl_div > target_div:
                break

        updates_time = time.time() - updates_start
        grad_steps_per_sec = grad_steps_done / updates_time
        global_steps_per_sec = steps_per_env * num_envs / (rollout_time + updates_time)

        policy_loss = policy_loss.item()
        critic_loss = critic_loss.item()
        entropy = entropy.item()
        clip_frac = np.mean(clip_fracs, dtype=np.float32)
        kl_div = kl_div.item()

        rollout_return = vec_env.total_return.item() / num_envs
        ep_return_mean = vec_env.mean_ep_return.item()
        ep_return_lower = vec_env.lower_ep_return.item()
        ep_return_upper = vec_env.upper_ep_return.item()
        ep_return_data = (ep_return_lower, ep_return_mean, ep_return_upper)

        if wandb.run is not None:
            wandb.log({
                "utils/ep_return_mean": ep_return_mean,
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
                "metrics/ep_return": ep_return_mean
            }, step=self.updates*steps_per_env*num_envs)

        return policy_loss, critic_loss, entropy, clip_frac, kl_div, rollout_return, ep_return_data

    def train(
        self,
        env_fn: callable,
        num_updates: int,
        num_envs: int,
        steps_per_env: int,
        num_epochs: int,
        batch_size: int,
        critic_coef: float,
        entropy_coef: float,
        clip_ratio: float,
        target_div: float,
        max_grad_norm: float,
        learning_rate: float,
        early_stop_reward: float = None
    ):
        test_env = env_fn()
        state_shape = test_env.reset()[0].shape
        action_shape = [test_env.action_space.n] # temporary
        self.agent = Agent(state_shape, action_shape, conv_net=self.conv_net, joint_net=self.joint_network)

        del test_env
        vec_env = SyncVecEnv(env_fn, num_envs, steps_per_env, self.agent)

        self.pi_optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.va_optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        lr_anneal = 1.0

        curr_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        wandb.init(project=f"exp-ppo-{curr_datetime}", config={
            "discount_factor": self.discount_factor,
            "gae_factor": self.gae_factor,
            "norm_adv": self.norm_adv,
            "clip_va_loss": self.clip_va_loss,
            "joint_network": self.joint_network,
            "num_updates": num_updates,
            "num_envs": num_envs,
            "steps_per_env": steps_per_env,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "critic_coef": critic_coef,
            "entropy_coef": entropy_coef,
            "clip_ratio": clip_ratio,
            "target_div": target_div,
            "max_grad_norm": max_grad_norm,
            "learning_rate": learning_rate,
            "early_stop_reward": early_stop_reward
        })

        ep_return_data_list = []

        early_stop_count = 0
        pbar = tqdm.trange(num_updates, leave=True)
        for update in pbar:
            pi_loss, va_loss, entropy, clip_frac, kl_div, roll_return, ep_return_data = self.train_step(
                vec_env, num_epochs, batch_size, critic_coef, entropy_coef, 
                clip_ratio, target_div, max_grad_norm, lr_anneal
            )

            wandb.log({
                "params/learning_rate": learning_rate * lr_anneal,
            }, step=self.updates*steps_per_env*num_envs, commit=True)

            ep_return = ep_return_data[1]
            ep_return_data_list.append(ep_return_data)

            lr_anneal -= 0.999 / (num_updates - 1)
            self.updates += 1

            pbar.set_postfix({
                "pi_loss": f"{pi_loss:.3f}", "va_loss": f"{va_loss:.3f}",
                "entropy": f"{entropy:.3f}", "return": f"{roll_return:.3f}",
                "ep_return": f"{ep_return:.3f}", "kl_div": f"{kl_div:.4f}",
                "clip_frac": f"{clip_frac:.3f}"
            })

            if early_stop_reward is not None and ep_return >= early_stop_reward:
                early_stop_count += 1
                if early_stop_count == 3:
                    pbar.close()
                    print("Early stop reward reached.")
                    break
            else:
                early_stop_count = 0

        wandb.log({"utils/ep_return_table": wandb.Table(
            columns=["ep_return_0.05", "ep_return", "ep_return_0.95"],
            data=ep_return_data_list)})

        wandb.finish()

