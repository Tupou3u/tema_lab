import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import torch
from isaaclab.envs import ManagerBasedRLEnv
# from tema_lab.tasks.manager_based.locomotion.velocity_rma_v3.config.quadruped.unitree_go2.vel.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from tema_lab.tasks.manager_based.locomotion.velocity_rma_v3.config.quadruped.unitree_go2.vel_with_pose.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from tema_lab.rl.rsl_rl.agents.actor_critic_o1 import ActorCritic_o1
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

env_cfg = Go2FlatEnvCfg()
# env_cfg = Go2RoughEnvCfg()

env_cfg.scene.num_envs = 1
env_cfg.terminations = None
env_cfg.events = None
env_cfg.commands.base_velocity.debug_vis = False
controller_cfg = Se2KeyboardCfg(
    v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
    v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
    omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
)
controller = Se2Keyboard(controller_cfg)
env_cfg.observations.policy.velocity_commands = ObsTerm(
    func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
)

env = ManagerBasedRLEnv(cfg=env_cfg)

model = ActorCritic_o1(
    num_actor_obs=283, 
    num_critic_obs=283,
    num_actions=12,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
    enc_dims=[256, 128, 64],
    len_o1=48,
    enc_activation=False
)
load_state = torch.load('/home/tema/Downloads/loaded_models/msi/2025-09-02_01-05-26_teacher/model_50000.pt', weights_only=True)['model_state_dict']
model.load_state_dict(load_state)

model.eval()
pi = model.act_inference

dt = env.step_dt
obs, _ = env.reset()

while simulation_app.is_running():
    start_time = time.time()
    with torch.inference_mode():
        # action = pi(obs['policy'].cpu())
        action = torch.zeros_like(env.action_manager.action)
        obs, rew, terminated, truncated, info = env.step(action)

    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0:
        time.sleep(sleep_time)

env.close()
simulation_app.close()
