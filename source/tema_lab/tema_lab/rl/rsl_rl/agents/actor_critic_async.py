from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from math import sqrt


def layer_init(layer, std=sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic_async(nn.Module):
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
        actor_cnn_kernel_sizes: list[int] = [3, 3, 3],
        actor_cnn_strides: list[int] = [3, 3, 3],
        actor_cnn_filters: list[int] = [32, 16, 8],
        actor_cnn_paddings: list[int] = [0, 0, 1],
        actor_cnn_dilations: list[int] = [1, 1, 1],
        critic_enc_dims=[128, 64],
        len_o1: int = 48,
        sum_actor_obs: int = 65,
        enc_activation: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.sum_actor_obs = sum_actor_obs
        self.num_critic_obs = num_critic_obs
        self.len_o1 = len_o1

        # Policy        
        s_out_channels = actor_cnn_filters
        s_in_channels = [self.len_o1] + actor_cnn_filters[:-1]

        cnn_actor_layers = []
        s_cnn_out = self.sum_actor_obs - 1
        for in_ch, out_ch, kernel_size, stride, padding, dilation in zip(
            s_in_channels, 
            s_out_channels, 
            actor_cnn_kernel_sizes, 
            actor_cnn_strides, 
            actor_cnn_paddings, 
            actor_cnn_dilations
        ):
            cnn_actor_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ))
            cnn_actor_layers.append(nn.BatchNorm1d(out_ch))
            cnn_actor_layers.append(activation)
            s_cnn_out = (s_cnn_out + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        cnn_actor_layers.append(nn.Flatten())
        cnn_actor_layers.append(nn.Linear(s_cnn_out * s_out_channels[-1], critic_enc_dims[-1]))
        if enc_activation:
            cnn_actor_layers.append(activation)
        self.cnn_actor = nn.Sequential(*cnn_actor_layers)

        actor_layers = []
        actor_layers.append(nn.Linear(critic_enc_dims[-1] + self.len_o1, actor_hidden_dims[0]))
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
        critic_enc_layers.append(nn.Linear(self.num_critic_obs - self.len_o1, critic_enc_dims[0]))
        critic_enc_layers.append(activation)
        for layer_index in range(len(critic_enc_dims) - 1):
            critic_enc_layers.append(nn.Linear(critic_enc_dims[layer_index], critic_enc_dims[layer_index + 1]))
            if layer_index != len(critic_enc_dims) - 2:
                critic_enc_layers.append(activation)
            elif enc_activation:
                critic_enc_layers.append(activation)

        self.critic_enc = nn.Sequential(*critic_enc_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(critic_enc_dims[-1] + self.len_o1, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Encoder: {self.cnn_actor}")
        print(f"Actor MLP: {self.actor}")
        print(f"Actor parameters: {sum([p.numel() for p in self.actor.parameters()]) + sum([p.numel() for p in self.cnn_actor.parameters()])}\n")
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
        # o_t = observations[:, -1, :].reshape(observations.shape[0], -1)
        # h = observations[:, :-1, :].permute(0, 2, 1)
        if observations.dim() == 3:
            o_t = observations[:, -1, :]
            h = observations[:, :-1, :].permute(0, 2, 1)
        else:
            o_t = observations[:, -self.len_o1:]
            h = observations.reshape(observations.shape[0], self.sum_actor_obs, self.len_o1)[:, :-1, :].permute(0, 2, 1)
            
        l_t = self.cnn_actor(h)
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
        if observations.dim() == 3:
            o_t = observations[:, -1, :]
            h = observations[:, :-1, :].permute(0, 2, 1)
        else:
            o_t = observations[:, -self.len_o1:]
            h = observations.reshape(observations.shape[0], self.sum_actor_obs, self.len_o1)[:, :-1, :].permute(0, 2, 1)
        l_t = self.cnn_actor(h)
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
