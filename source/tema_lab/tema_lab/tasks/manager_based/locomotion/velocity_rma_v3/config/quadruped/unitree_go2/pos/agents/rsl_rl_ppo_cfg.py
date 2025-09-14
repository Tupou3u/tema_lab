# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,  
    RslRlPpoActorCritic_o1_Cfg,
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationCNNStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg
)


@configclass
class Go2RoughPPORunnerCfg_Teacher(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_v3_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCritic_o1_Cfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[256, 128, 64],  
        activation="elu",
        enc_dims=[256, 128, 64],
        len_o1=48,
        enc_activation=False
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.1,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.01,  
        # entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=8,
        # num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,  
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True
    )


@configclass
class Go2FlatPPORunnerCfg_Teacher(Go2RoughPPORunnerCfg_Teacher):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_v3_flat"


@configclass
class Go2RoughPPORunnerCfg_Policy(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_v3_rough"
    empirical_normalization = False
    policy = RslRlDistillationCNNStudentTeacherCfg(
        class_name="CNN1d_o2_StudentTeacher",
        init_noise_std=0.1,
        student_hidden_dims=[256, 128, 64], 
        teacher_hidden_dims=[256, 128, 64], 
        activation="elu",
        student_cnn_kernel_sizes=[5, 5, 5, 5, 5, 5],
        student_cnn_strides=[1, 2, 1, 2, 1, 2],
        student_cnn_filters=[32] * 6,
        student_cnn_paddings=[2, 2, 4, 2, 8, 2],
        student_cnn_dilations=[1, 1, 2, 1, 4, 1]
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation_o1",
        num_learning_epochs=10,
        learning_rate=1e-4,
        gradient_length=16
    )


@configclass
class Go2FlatPPORunnerCfg_Policy(Go2RoughPPORunnerCfg_Policy):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_v3_flat"
