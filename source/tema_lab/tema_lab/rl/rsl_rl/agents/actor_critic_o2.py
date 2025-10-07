from __future__ import annotations
from dataclasses import MISSING

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from math import sqrt


class ActorCritic_o2(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims,
        critic_hidden_dims,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        len_o: int = MISSING,
        len_h: int = MISSING,
        len_p: int = MISSING,
        len_l_o: int = MISSING,
        len_l_h: int = MISSING,
        len_l_p: int = MISSING,
        o_enc_dims: list[int] = MISSING,
        h_enc_dims: list[int] = MISSING,
        p_enc_dims: list[int] = MISSING,
        history_len: int = MISSING,
        enc_activation: bool = MISSING,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.len_o = len_o
        self.len_h = len_h
        self.len_p = len_p
        self.history_len = history_len

        # Policy                
        self.actor_o_enc = self.create_mlp(len_o, len_l_o, o_enc_dims, activation, out_activation=False, std=1.0)
        self.actor_h_enc = self.create_mlp(len_h, len_l_h, h_enc_dims, activation, out_activation=False, std=1.0)
        self.actor_p_enc = self.create_mlp(len_p, len_l_p, p_enc_dims, activation, out_activation=False, std=1.0)
        self.actor_backbone = self.create_mlp(
            len_o + len_l_o + len_l_h + len_l_p, num_actions, actor_hidden_dims, activation, out_activation=False, std=1.0
        )

        # Value function
        self.critic_o_enc = self.create_mlp(len_o, len_l_o, o_enc_dims, activation, out_activation=False, std=1.0)
        self.critic_h_enc = self.create_mlp(len_h, len_l_h, h_enc_dims, activation, out_activation=False, std=1.0)
        self.critic_p_enc = self.create_mlp(len_p, len_l_p, p_enc_dims, activation, out_activation=False, std=1.0)
        self.critic_backbone = self.create_mlp(
            len_o + len_l_o + len_l_h + len_l_p, 1, critic_hidden_dims, activation, out_activation=False, std=1.0
        )

        print(f"Actor Encoder: {self.actor_o_enc, self.actor_h_enc, self.actor_p_enc}")
        print(f"Actor MLP: {self.actor_backbone}")
        print(f"Actor parameters: {sum([p.numel() for p in list(self.actor_o_enc.parameters()) + list(self.actor_h_enc.parameters()) + list(self.actor_p_enc.parameters()) + list(self.actor_backbone.parameters())])}\n")
        print(f"Critic Encoder: {self.critic_o_enc, self.critic_h_enc, self.critic_p_enc}")
        print(f"Critic MLP: {self.critic_backbone}")
        print(f"Critic parameters: {sum([p.numel() for p in list(self.critic_o_enc.parameters()) + list(self.critic_h_enc.parameters()) + list(self.critic_p_enc.parameters()) + list(self.critic_backbone.parameters())])}\n")

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
    def layer_init(layer, std=sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def create_mlp(self, input_dim, out_dim, hidden_dims, activation, out_activation=False, std=1.0):
        layers = []
        layers.append(self.layer_init(nn.Linear(input_dim, hidden_dims[0])))
        layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(self.layer_init(nn.Linear(hidden_dims[layer_index], out_dim), std=std))
                if out_activation:
                    layers.append(activation)
            else:
                layers.append(self.layer_init(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1])))
                layers.append(activation)
        return nn.Sequential(*layers)

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