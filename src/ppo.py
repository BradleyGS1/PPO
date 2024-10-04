

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


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
            self.init_layer(nn.Linear(self.state_shape[0], 256)),
            nn.ReLU(),
            self.init_layer(nn.Linear(256, 256)),
            nn.ReLU()
        )
        if self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Linear(4, 256)),
                nn.ReLU(),
                self.init_layer(nn.Linear(256, 256)),
                nn.ReLU()
            )
        self.policy = self.init_layer(nn.Linear(256, self.action_shape[0]), std=0.01)
        self.critic = self.init_layer(nn.Linear(256, 1), std=1)

    def get_values(
        self,
        states: torch.Tensor
    ):
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
    ):
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
            actions = action_dists.sample()

        log_probs = action_dists.log_prob(actions)
        values = values.flatten()
        entropy = action_dists.entropy().mean()

        return actions, log_probs, values, entropy


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

        self.agent = None

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

        if torch.cuda.is_available():
            ep_advantages = ep_advantages.cuda()

        return ep_advantages

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
            clip_frac = ((prob_ratios - 1).abs() > clip_ratio).sum()

            # Compute KL Divergence approximation http://joschu.net/blog/kl-approx.html
            kl_div = (prob_ratios - 1 - prob_ratios.log()).mean()

        return policy_loss, value_loss, clip_frac, kl_div
