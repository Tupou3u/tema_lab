import gymnasium as gym

##
# Register Gym environments.
##


# ----------------------------------- Position Control ----------------------------------- #

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Flat-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl-Rough-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)

# ----------------------------------- Position Control 6D ----------------------------------- #

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl6D-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_6d.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_6d.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl6D-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_6d.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_6d.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl6D-Flat-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_6d.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_6d.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="TemaLab-Go2-Velocity-PositionControl6D-Rough-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_6d.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos_6d.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)


