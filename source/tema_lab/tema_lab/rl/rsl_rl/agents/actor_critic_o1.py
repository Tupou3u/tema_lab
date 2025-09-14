# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class ActorCritic_o1(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims,
        critic_hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        enc_dims=[128, 64],
        len_o1=48,
        enc_activation=True,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.len_obs = num_actor_obs
        self.len_o1 = len_o1

        # Policy
        actor_enc_layers = []
        actor_enc_layers.append(nn.Linear(self.len_obs - self.len_o1, enc_dims[0]))
        actor_enc_layers.append(activation)
        for layer_index in range(len(enc_dims) - 1):
            actor_enc_layers.append(nn.Linear(enc_dims[layer_index], enc_dims[layer_index + 1]))
            if layer_index != len(enc_dims) - 2:
                actor_enc_layers.append(activation)
            elif enc_activation:
                actor_enc_layers.append(activation)
                
        self.actor_enc = nn.Sequential(*actor_enc_layers)

        actor_layers = []
        actor_layers.append(nn.Linear(enc_dims[-1] + self.len_o1, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_enc_layers = []
        critic_enc_layers.append(nn.Linear(self.len_obs - self.len_o1, enc_dims[0]))
        critic_enc_layers.append(activation)
        for layer_index in range(len(enc_dims) - 1):
            critic_enc_layers.append(nn.Linear(enc_dims[layer_index], enc_dims[layer_index + 1]))
            if layer_index != len(enc_dims) - 2:
                critic_enc_layers.append(activation)
            elif enc_activation:
                actor_enc_layers.append(activation)

        self.critic_enc = nn.Sequential(*critic_enc_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(enc_dims[-1] + self.len_o1, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Encoder: {self.actor_enc}")
        print(f"Actor MLP: {self.actor}")
        print(f"Actor parameters: {sum([p.numel() for p in self.actor.parameters()]) + sum([p.numel() for p in self.actor_enc.parameters()])}\n")
        print(f"Critic Encoder: {self.critic_enc}")
        print(f"Critic MLP: {self.critic}")
        print(f"Critic parameters: {sum([p.numel() for p in self.critic.parameters()]) + sum([p.numel() for p in self.critic_enc.parameters()])}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        o_t = observations[:, :self.len_o1]
        x_t = observations[:, self.len_o1:]
        l_t = self.actor_enc(x_t)
        mean = self.actor(torch.cat((o_t, l_t), dim=1))
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        o_t = observations[:, :self.len_o1]
        x_t = observations[:, self.len_o1:]
        l_t = self.actor_enc(x_t)
        actions_mean = self.actor(torch.cat((o_t, l_t), dim=1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        o_t = critic_observations[:, :self.len_o1]
        x_t = critic_observations[:, self.len_o1:]
        l_t = self.critic_enc(x_t)
        value = self.critic(torch.cat((o_t, l_t), dim=1))
        return value

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
