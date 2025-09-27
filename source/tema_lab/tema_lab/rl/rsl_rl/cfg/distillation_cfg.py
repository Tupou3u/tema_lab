# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg


#########################
# Policy configurations #
#########################    
    
    
@configclass
class RslRlDistillation_CNN1d_o1_StudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "CNN1d_o1_StudentTeacher"
    student_cnn_kernel_sizes: list[int] | list[tuple[int, int]] = MISSING
    student_cnn_strides: list[int] | list[tuple[int, int]] = MISSING
    student_cnn_filters: list[int] = MISSING
    student_cnn_paddings: list[int] | list[tuple[int, int]] = MISSING
    student_cnn_dilations: list[int] | list[tuple[int, int]] = MISSING
    teacher_enc_dims: list[int] = MISSING
    len_o1: int = MISSING
    sum_student_obs: int = MISSING
    enc_activation: bool = MISSING
    
    
@configclass
class RslRlDistillation_CNN1d_o1_res_StudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "CNN1d_o1_res_StudentTeacher"
    student_cnn_blocks_kernel_sizes: list[list[int]] = MISSING
    student_cnn_blocks_strides: list[list[int]] = MISSING
    student_cnn_blocks_in_channels: list[list[int]] = MISSING
    student_cnn_blocks_out_channels: list[list[int]] = MISSING
    student_cnn_blocks_paddings: list[list[int]] = MISSING
    student_cnn_blocks_dilations: list[list[int]] = MISSING
    teacher_enc_dims: list[int] = MISSING
    len_o1: int = MISSING
    sum_student_obs: int = MISSING
    enc_activation: bool = MISSING
    
    
@configclass
class RslRlDistillation_Recurrent_o1_StudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "Recurrent_o1_StudentTeacher"
    rnn_type: str = MISSING
    rnn_hidden_dim: int = MISSING
    rnn_num_layers: int = MISSING
    teacher_recurrent: bool = MISSING
    teacher_enc_dims: list[int] = MISSING
    enc_activation: bool = MISSING



############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillation_o1_AlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "Distillation_o1"
    loss_type: str = MISSING
    alpha: float = MISSING
