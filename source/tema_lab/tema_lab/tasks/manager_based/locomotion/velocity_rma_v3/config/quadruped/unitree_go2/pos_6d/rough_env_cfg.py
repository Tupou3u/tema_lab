from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3.velocity_env_cfg_6d import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import *  # isort: skip
from math import pi


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }

        # self.scene.robot.actuators = {
        #     "legs": DelayedPDActuatorCfg(
        #         joint_names_expr=[".*"],
        #         effort_limit=23.5,
        #         velocity_limit=30.0,
        #         stiffness=25, 
        #         damping=0.5, 
        #         min_delay=0,  
        #         max_delay=4,  
        #     )
        # }
        
        # self.scene.robot.actuators = {
        #     "legs": DelayedDCMotorCfg(
        #         joint_names_expr=[".*"],
        #         effort_limit=23.5,
        #         saturation_effort=23.5,
        #         velocity_limit=30.0,
        #         stiffness=25, 
        #         damping=0.5, 
        #         min_delay=0,  
        #         max_delay=5,  # in physics timesteps
        #     )
        # }
        
        # self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.4
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.15
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.15
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.075
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.075
        
        self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.15
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.15)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.5)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.5)

        # ------------------------------Observations------------------------------

        if hasattr(self.observations, 'teacher'):
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            self.observations.teacher.phase.params["cycle_time"] = self.episode_length_s
            
        else:
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.phase.params["cycle_time"] = self.episode_length_s

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------

        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # ------------------------------Rewards------------------------------

        self.rewards.base_linear_velocity.weight = 5.0
        self.rewards.base_angular_velocity.weight = 5.0
        self.rewards.base_height.weight = 5.0
        # self.rewards.base_roll.weight = 3.0
        # self.rewards.base_pitch.weight = 3.0
        self.rewards.action_smoothness.weight = -1.0
        self.rewards.joint_acc.weight = -1.0e-4
        self.rewards.joint_pos.weight = -0.1
        self.rewards.joint_torques.weight = -5.0e-4
        self.rewards.joint_vel.weight = -1.0e-2
    
        self.rewards.base_linear_velocity.params["std"] = 1.0
        self.rewards.base_angular_velocity.params["std"] = 2.0
        self.rewards.base_height.params["std"] = 0.2

        # self.rewards.base_linear_velocity.weight = 3.0
        # self.rewards.base_angular_velocity.weight = 3.0
        # self.rewards.base_height.weight = 4.0
        # self.rewards.base_roll.weight = 5.0
        # self.rewards.base_pitch.weight = 5.0
        # # self.rewards.foot_clearance.weight = 5.0
        # # self.rewards.gait.weight = 5.0
        # self.rewards.action_smoothness.weight = -1.0
        # self.rewards.air_time_variance.weight = -1.0
        # self.rewards.foot_slip.weight = -0.5
        # self.rewards.joint_acc.weight = -1.0e-4
        # # self.rewards.joint_pos.weight = -0.1
        # # self.rewards.joint_torques.weight = -5.0e-4
        # self.rewards.joint_vel.weight = -1.0e-2
        # self.rewards.undesired_contacts.weight = -1.0
        # # self.rewards.contact_forces.weight = -1.5e-4
        # self.rewards.feet_contact_without_cmd.weight = 1.0
        # # self.rewards.feet_force_variance_without_cmd.weight = 5.0
        # # self.rewards.joint_pos_limits.weight = -5.0
        # # self.rewards.joint_vel_limits.weight = -1.0
        # # self.rewards.joint_torque_limits.weight = -0.1
        # # self.rewards.lin_vel_z_l2.weight = -2.0
        # # self.rewards.ang_vel_l2.weight = -0.05
        # # self.rewards.lin_vel_w_z_l2.weight = -2.0
        # # self.rewards.ang_vel_w_z_l2.weight = -0.5
        # # self.rewards.joint_mirror.weight = -0.05
        # # self.rewards.action_mirror.weight = -0.005
        # self.rewards.feet_stumble.weight = -0.1
        
        # self.rewards.base_linear_velocity.params["std"] = 0.5
        # self.rewards.base_angular_velocity.params["std"] = 0.2
        # self.rewards.base_height.params["std"] = 0.1
        # self.rewards.base_roll.params["std"] = 0.1
        # self.rewards.base_pitch.params["std"] = 0.05
        # self.rewards.feet_force_variance_without_cmd.params["std"] = 40.0
        
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        mirror_joints = [
            ["FR_hip_joint", "RL_hip_joint"], ["FR_thigh_joint", "RL_thigh_joint"], ["FR_calf_joint", "RL_calf_joint"],
            ["FL_hip_joint", "RR_hip_joint"], ["FL_thigh_joint", "RR_thigh_joint"], ["FL_calf_joint", "RR_calf_joint"],
        ]
        self.rewards.action_mirror.params["mirror_joints"] = [[self.joint_names.index(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints]

        # ------------------------------Terminations------------------------------

        self.terminations.illegal_contact = None
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base", ".*_hip", ".*_thigh", ".*_calf"]

        # ------------------------------Commands------------------------------

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        
        self.commands.base_pose.rel_default_pos_envs = 1.0
        self.commands.base_pose.ranges.height = (0.15, 0.38) 
        self.commands.base_pose.ranges.roll = (-pi/6, pi/6)  
        self.commands.base_pose.ranges.pitch = (-pi/6, pi/6) 
        
        # self.commands.base_pose.ranges.height = (0.15, 0.38) 
        # self.commands.base_pose.ranges.roll = (0.0, 0.0) 
        # self.commands.base_pose.ranges.pitch = (0.0, 0.0) 

        if self.__class__.__name__ == "Go2RoughEnvCfg":
            self.disable_zero_weight_rewards()
        