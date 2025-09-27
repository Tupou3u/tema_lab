from isaaclab.utils import configclass
from tema_lab.tasks.manager_based.locomotion.velocity_rma_v3_ftg.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from tema_lab.lab.terrains.config import HARD_ROUGH_TERRAINS_CFG, EASY_ROUGH_TERRAINS_CFG
from tema_lab.assets.robots.unitree import *


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    h_max = 0.08
    f_b = 1.25

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }
        
        # self.scene.robot.actuators = {
        #     "legs": delayed_actuators.DelayedUnitreeActuatorCfg(
        #         joint_names_expr=[".*"],
        #         stiffness=25, 
        #         damping=0.5, 
        #         friction=0.01,
        #         X1=13.5,
        #         X2=30,
        #         Y1=20.2,
        #         Y2=23.4,
        #         min_delay=0,  
        #         max_delay=4,  # in physics timesteps
        #     )
        # }      
        
        self.scene.terrain.terrain_generator = HARD_ROUGH_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = EASY_ROUGH_TERRAINS_CFG
        # self.curriculum = None
        if self.curriculum:
            self.scene.terrain.terrain_generator.curriculum = True
        
        # self.scene.terrain.terrain_generator.num_cols = 78
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.1"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.1_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.2"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.2_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.3"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_0.3_inv"].proportion = 0.0

        # ------------------------------Observations------------------------------
                
        if hasattr(self.observations, 'teacher'):
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
            self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
            self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
            
        else:
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names

        # ------------------------------Actions------------------------------

        self.actions.joint_pos_ftg.alpha = 0.01
        self.actions.joint_pos_ftg.f_alpha = 0.1
        self.actions.joint_pos_ftg.h_max = self.h_max
        self.actions.joint_pos_ftg.f_b = self.f_b
        self.actions.joint_pos_ftg.joint_names = self.scene.robot.joint_sdk_names

        # ------------------------------Events------------------------------

        self.events.randomize_rigid_base_mass.params["mass_distribution_params"] = (-2.0, 5.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2)
            },
            "velocity_range": {}
        }

        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_base_mass = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_base_com_position = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # ------------------------------Rewards------------------------------
        
        self.rewards.base_linear_velocity.weight = 10.0
        self.rewards.base_linear_velocity.params["std"] = 1.0
        self.rewards.base_angular_velocity.weight = 5.0
        self.rewards.base_angular_velocity.params["std"] = 2.0
        self.rewards.foot_clearance.weight = 3.0 
        self.rewards.foot_clearance.params["std"] = 0.01
        self.rewards.foot_clearance.params["target_height"] = self.h_max + 0.023
        self.rewards.gait.weight = 1.0
        self.rewards.gait.params["std"] = 0.05
        self.rewards.air_time.weight = 1.0
        self.rewards.air_time.params["mode_time"] = (1 / self.f_b) / 2
        self.rewards.action_rate.weight = -0.01
        self.rewards.action_smoothness.weight = -0.01
        self.rewards.air_time_variance.weight = -1.0
        self.rewards.base_orientation.weight = 1.0
        self.rewards.base_orientation.params["std"] = 0.05
        self.rewards.foot_slip.weight = -0.5
        self.rewards.joint_acc.weight = -1.0e-5
        self.rewards.joint_vel.weight = -0.01
        self.rewards.joint_vel.params["asset_cfg"].joint_names = ".*_hip_joint"
        self.rewards.joint_power.weight = -1.0e-4    
        
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.base_height.weight = 1.0
        self.rewards.base_height.params["std"] = 0.05
        self.rewards.base_height.params["target_height"] = self.actions.joint_pos_ftg.base_h
        self.rewards.contact_forces.weight = -1.0e-3
        self.rewards.stand_still.weight = -5.0
        self.rewards.stand_still_vel.weight = -1.0
        self.rewards.feet_contact_without_cmd.weight = 0.25
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_torque_limits.weight = -0.1
        self.rewards.lin_vel_z_l2.weight = -5.0
        self.rewards.ang_vel_xy_l2.weight = -2.0
        self.rewards.joint_mirror.weight = -0.1
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
            ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
        ]
        self.rewards.feet_distance_y_exp.weight = 3.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.2
        self.rewards.feet_distance_y_exp.params["std"] = 0.025

        # ------------------------------Terminations------------------------------

        self.terminations.illegal_contact = None
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base", ".*_hip", ".*_thigh", ".*_calf"]

        # ------------------------------Commands------------------------------

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 3.0) 
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        
        if self.__class__.__name__ == "Go2RoughEnvCfg":
            self.disable_zero_weight_rewards()