# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class CNN1d_o1_StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        student_cnn_kernel_sizes: list[int] = [3, 3, 3],
        student_cnn_strides: list[int] = [3, 3, 3],
        student_cnn_filters: list[int] = [32, 16, 8],
        student_cnn_paddings: list[int] = [0, 0, 1],
        student_cnn_dilations: list[int] = [1, 1, 1],
        teacher_enc_dims: list[int] = [128, 64],
        len_o1: int = 48,
        sum_student_obs: int = 65,
        enc_activation: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        self.sum_student_obs = sum_student_obs
        self.num_teacher_obs = num_teacher_obs
        self.len_o1 = len_o1

        # student
        s_out_channels = student_cnn_filters
        s_in_channels = [self.len_o1] + student_cnn_filters[:-1]

        cnn_student_layers = []
        s_cnn_out = self.sum_student_obs - 1
        for in_ch, out_ch, kernel_size, stride, padding, dilation in zip(
            s_in_channels, 
            s_out_channels, 
            student_cnn_kernel_sizes, 
            student_cnn_strides, 
            student_cnn_paddings, 
            student_cnn_dilations
        ):
            cnn_student_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ))
            cnn_student_layers.append(nn.BatchNorm1d(out_ch))
            cnn_student_layers.append(nn.ReLU())
            s_cnn_out = (s_cnn_out + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        cnn_student_layers.append(nn.Flatten())
        cnn_student_layers.append(nn.Linear(s_cnn_out * s_out_channels[-1], teacher_enc_dims[-1]))
        if enc_activation:
            cnn_student_layers.append(activation)
        self.cnn_student = nn.Sequential(*cnn_student_layers)

        student_layers = []
        student_layers.append(nn.Linear(teacher_enc_dims[-1] + self.len_o1, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # teacher
        teacher_enc_layers = []
        teacher_enc_layers.append(nn.Linear(self.num_teacher_obs - self.len_o1, teacher_enc_dims[0]))
        teacher_enc_layers.append(activation)
        for layer_index in range(len(teacher_enc_dims) - 1):
            teacher_enc_layers.append(nn.Linear(teacher_enc_dims[layer_index], teacher_enc_dims[layer_index + 1]))
            if layer_index != len(teacher_enc_dims) - 2:
                teacher_enc_layers.append(activation)
            elif enc_activation:
                teacher_enc_layers.append(activation)
        self.teacher_enc = nn.Sequential(*teacher_enc_layers)
        self.teacher_enc.eval()

        teacher_layers = []
        teacher_layers.append(nn.Linear(teacher_enc_dims[-1] + self.len_o1, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student CNN: {self.cnn_student}")
        print(f"Student MLP: {self.student}")
        print(f"Student parameters: {sum([p.numel() for p in self.cnn_student.parameters()]) + sum([p.numel() for p in self.student.parameters()])}\n")
        print(f"Teacher Encoder: {self.teacher_enc}")
        print(f"Teacher MLP: {self.teacher}")
        print(f"Teacher parameters: {sum([p.numel() for p in self.teacher_enc.parameters()]) + sum([p.numel() for p in self.teacher.parameters()])}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
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
        o_t = observations[:, -1, :].reshape(observations.shape[0], -1)
        h = observations[:, :-1, :].permute(0, 2, 1)
        # h = observations.permute(0, 2, 1)
        l_t = self.cnn_student(h)
        mean = self.student(torch.cat((o_t, l_t), dim=1))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, latent=False):
        if observations.dim() == 3:
            o_t = observations[:, -1, :]
            h = observations[:, :-1, :].permute(0, 2, 1)
            # h = observations.permute(0, 2, 1)
        else:
            o_t = observations[:, -self.len_o1:]
            h = observations.reshape(observations.shape[0], self.sum_student_obs, self.len_o1)[:, :-1, :].permute(0, 2, 1)
            # h = observations.reshape(observations.shape[0], self.num_student_obs, self.len_student_o_t).permute(0, 2, 1)
        l_t = self.cnn_student(h)
        actions_mean = self.student(torch.cat((o_t, l_t), dim=1))
        if latent: 
            return actions_mean, l_t
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            o_t = teacher_observations[:, :self.len_o1]
            h = teacher_observations[:, self.len_o1:]
            z_t = self.teacher_enc(h)
            actions = self.teacher(torch.cat((o_t, z_t), dim=1))
        return actions
    
    def evaluate_enc(self, teacher_observations):
        with torch.no_grad():
            h = teacher_observations[:, self.len_o1:]
            z_t = self.teacher_enc(h)
        return z_t 

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value

            self.teacher.load_state_dict(teacher_state_dict, strict=strict)

            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor_enc." in key:
                    teacher_state_dict[key.replace("actor_enc.", "")] = value

            self.teacher_enc.load_state_dict(teacher_state_dict, strict=strict)
                
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_enc.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_enc.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
