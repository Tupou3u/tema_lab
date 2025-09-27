import gymnasium as gym

##
# Register Gym environments.
##

# ----------------------------------- Velocity ----------------------------------- #

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Flat-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Rough-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)


# ----------------------------------- Velocity with pose ----------------------------------- #

gym.register(
    id="TemaLab-Go2-Velocity_with_orientation-PositionControl-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel_with_pose.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel_with_pose.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity_with_orientation-PositionControl-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel_with_pose.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel_with_pose.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity_with_orientation-PositionControl-Flat-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel_with_pose.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel_with_pose.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity_with_orientation-PositionControl-Rough-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel_with_pose.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel_with_pose.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)