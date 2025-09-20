import gymnasium as gym

# ----------------------------------- Position Control FTG ----------------------------------- #

gym.register(
    id="Go2-Velocity-PositionControlFTG-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="Go2-Velocity-PositionControlFTG-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="Go2-Velocity-PositionControlFTG-Flat-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="Go2-Velocity-PositionControlFTG-Rough-Policy-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.pos.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)