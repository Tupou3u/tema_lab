from __future__ import annotations
import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class Memory(torch.nn.Module):
    def __init__(
        self, 
        input_size, 
        type="gru", 
        num_layers=1, 
        hidden_size=256,
        mlp_dim=64,
        enc_activation=True
    ):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
        
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_size, mlp_dim))
        if enc_activation:
            mlp_layers.append(nn.ELU())
        self.mlp_enc = nn.Sequential(*mlp_layers)

    def forward(self, input, masks=None, hidden_states=None):
        hid_out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        out = self.mlp_enc(hid_out).squeeze(0)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()


class Recurrent_o1_StudentTeacher(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        init_noise_std=0.1,
        teacher_enc_dims=[128, 64],
        enc_activation=True,
        **kwargs,
    ):
        super().__init__()
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        activation = resolve_nn_activation(activation)

        self.num_student_obs = num_student_obs
        self.num_teacher_obs = num_teacher_obs

        # student
        self.student_enc = Memory(
            num_student_obs, 
            type=rnn_type, 
            num_layers=rnn_num_layers, 
            hidden_size=rnn_hidden_dim, 
            mlp_dim=teacher_enc_dims[-1], 
            enc_activation=enc_activation
        )
        student_layers = []
        student_layers.append(nn.Linear(teacher_enc_dims[-1] + self.num_student_obs, student_hidden_dims[0]))
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
        teacher_enc_layers.append(nn.Linear(self.num_teacher_obs - self.num_student_obs, teacher_enc_dims[0]))
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
        teacher_layers.append(nn.Linear(teacher_enc_dims[-1] + self.num_student_obs, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student CNN: {self.student_enc}")
        print(f"Student MLP: {self.student}")
        print(f"Student parameters: {sum([p.numel() for p in self.student_enc.parameters()]) + sum([p.numel() for p in self.student.parameters()])}\n")
        print(f"Teacher Encoder: {self.teacher_enc}")
        print(f"Teacher MLP: {self.teacher}")
        print(f"Teacher parameters: {sum([p.numel() for p in self.teacher_enc.parameters()]) + sum([p.numel() for p in self.teacher.parameters()])}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.student_enc.reset(dones, hidden_states[0])

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
        l_t = self.student_enc(observations)
        mean = self.student(torch.cat((observations, l_t), dim=1))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, latent=False):
        l_t = self.student_enc(observations)
        actions_mean = self.student(torch.cat((observations, l_t), dim=1))
        if latent:
            return actions_mean, l_t
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            o_t = teacher_observations[:, :self.num_student_obs]
            h = teacher_observations[:, self.num_student_obs:]
            z_t = self.teacher_enc(h)
            actions = self.teacher(torch.cat((o_t, z_t), dim=1))
        return actions
    
    def evaluate_enc(self, teacher_observations):
        with torch.no_grad():
            h = teacher_observations[:, self.num_student_obs:]
            z_t = self.teacher_enc(h)
        return z_t 
    
    def get_hidden_states(self):
        return self.student_enc.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.student_enc.detach_hidden_states(dones)

    def load_state_dict(self, state_dict, strict=True):
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

    