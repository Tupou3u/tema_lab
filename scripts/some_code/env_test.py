import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go2 test")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch
from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab_tasks.manager_based.locomotion.velocity_rma_v2_1.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
# from isaaclab_tasks.manager_based.locomotion.velocity_ik.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
# from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3_ftg.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
# from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3.config.quadruped.unitree_go2.pos_6d.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from isaaclab_rl.rsl_rl.new_modules.actor_critic_o1 import ActorCritic_o1
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm

env_cfg = Go2FlatEnvCfg()
# env_cfg = Go2RoughEnvCfg()

env_cfg.scene.num_envs = 1
env_cfg.terminations = None
env_cfg.events = None
env_cfg.commands.base_velocity.debug_vis = False
controller_cfg = Se2KeyboardCfg(
    v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
    v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
    omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1]
)
controller = Se2Keyboard(controller_cfg)
env_cfg.observations.policy.velocity_commands = ObsTerm(
    func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
)
env = ManagerBasedRLEnv(cfg=env_cfg)

model = ActorCritic_o1(
    num_actor_obs=297,
    num_critic_obs=297,
    num_actions=16, 
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation='elu',
    enc_dims=[256, 128, 64],
    len_o1=48,
    enc_activation=False
)
load_state = torch.load('/home/aivizw/Downloads/model_4000.pt', weights_only=True)['model_state_dict']

# model = ActorCritic_o1(
#     num_actor_obs=283,
#     num_critic_obs=283,
#     num_actions=12, 
#     actor_hidden_dims=[256, 128, 64],
#     critic_hidden_dims=[256, 128, 64],
#     enc_dims=[256, 128, 64],
#     len_o1=48,
#     enc_activation=False
# )
# load_state = torch.load('logs/rsl_rl/go2_velocity_rma_v3_flat_v2/2025-09-09_16-45-38/model_18000.pt', weights_only=True)['model_state_dict']    

# model = ActorCritic_o1(
#     num_actor_obs=258,
#     num_critic_obs=258,
#     num_actions=12, 
#     actor_hidden_dims=[256, 128, 64],
#     critic_hidden_dims=[256, 128, 64],
#     enc_dims=[128, 64],
#     len_o1=48,
#     enc_activation=True
# )
# load_state = torch.load('/home/aivizw/Downloads/model_95000.pt', weights_only=True)['model_state_dict']        

model.load_state_dict(load_state)
model.eval()
pi = model.act_inference
# pi = model.act

default_pos = torch.tensor([
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5
])

dt = env.unwrapped.step_dt
# simulate physics
obs, info = env.reset()

while simulation_app.is_running():
    start_time = time.time()
    with torch.inference_mode():
        action = pi(obs['policy'].cpu())
        # action = torch.zeros_like(env.action_manager.action)
        obs, rew, terminated, truncated, info = env.step(action)
        
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0:
        time.sleep(sleep_time)

env.close()
simulation_app.close()
