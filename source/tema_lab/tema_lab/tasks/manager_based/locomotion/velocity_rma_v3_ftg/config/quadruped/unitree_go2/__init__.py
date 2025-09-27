import gymnasium as gym

# ----------------------------------- Position Control FTG ----------------------------------- #

gym.register(
    id="TemaLab-Go2-VelocityFTG-Flat-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-VelocityFTG-Rough-Teacher-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Teacher",
    },
)

gym.register(
    id="TemaLab-Go2-VelocityFTG-Flat-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.flat_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg_Policy",
    },
)

gym.register(
    id="TemaLab-Go2-VelocityFTG-Rough-Distillation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vel.rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.vel.agents.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg_Policy",
    },
)