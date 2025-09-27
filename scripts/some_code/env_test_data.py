import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

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
from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from isaaclab_rl.rsl_rl.new_modules.actor_critic_o1 import ActorCritic_o1
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
import matplotlib.pyplot as plt
from isaaclab.managers import SceneEntityCfg

env_cfg = Go2FlatEnvCfg()

env_cfg.scene.num_envs = 1
env_cfg.terminations.time_out = None
env_cfg.terminations.terrain_out_of_bounds = None
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

# model = ActorCritic_o1(
#     num_actor_obs=258, 
#     num_critic_obs=258,
#     num_actions=12,
#     actor_hidden_dims=[256, 128, 64],
#     critic_hidden_dims=[256, 128, 64],
#     activation="elu",
#     enc_dims=[128, 64],
#     len_o1=48,
#     enc_activation=True
# )
# load_state = torch.load('/home/aivizw/Downloads/model_62000.pt', weights_only=True)['model_state_dict']

model = ActorCritic_o1(
    num_actor_obs=283,
    num_critic_obs=283,
    num_actions=12, 
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    enc_dims=[256, 128, 64],
    len_o1=48,
    enc_activation=False
)
model_path = 'logs/rsl_rl/go2_velocity_rma_v3_flat_v2/2025-09-04_11-52-13/model_20000.pt'
load_state = torch.load(model_path, weights_only=True)['model_state_dict']

model.load_state_dict(load_state)
model.eval()
pi = model.act_inference

data = []

dt = env.step_dt
obs, _ = env.reset()
count = 0

while simulation_app.is_running():
    start_time = time.time()
    with torch.inference_mode():
        action = pi(obs['policy'].cpu())
        obs, rew, terminated, truncated, info = env.step(action)

    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0:
        time.sleep(sleep_time)

    # data.append(env.scene["robot"].data.body_lin_vel_w[:, [15, 16, 17, 18], 0].squeeze().tolist())
    data.append(env.scene["robot"].data.joint_pos[:, [0, 1, 2, 3]].squeeze().tolist())
    # data.append(env.scene["robot"].data.body_pos_w[:, [16, 15], 2].squeeze().tolist())
    # data.append(env.scene["robot"].data.body_pos_w[:, [18, 17], 2].squeeze().tolist())

    if count == 300:
        plt.plot(data[100:])
        plt.savefig(model_path.replace(".pt", "_") + 'hip_pos.png')

    count += 1

env.close()
simulation_app.close()
