from isaaclab.utils import configclass
from .env_cfg import LocomotionVelocityRoughEnvCfg, ObservationsCfg
from tema_lab.lab.terrains.config import *
from tema_lab.assets.robots.unitree import *


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Sence------------------------------

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }
        
        self.scene.robot.actuators = {
            "legs": delayed_actuators.DelayedUnitreeActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=25, 
                damping=0.5, 
                friction=0.01,
                X1=13.5,
                X2=30,
                Y1=20.2,
                Y2=23.4,
                min_delay=0,  
                max_delay=4,  # in physics timesteps
            )
        }      
        
        self.scene.terrain.terrain_generator = HARD_ROUGH_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = EASY_ROUGH_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = TEST_ROUGH_TERRAINS_CFG
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

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = self.scene.robot.joint_sdk_names
        self.actions.joint_pos.clip = {
            ".*_hip_joint": (-1.0472, 1.0472),
            "F[L,R]_thigh_joint": (-1.5708, 3.4907),
            "R[L,R]_thigh_joint": (-0.5236, 4.5379),
            ".*_calf_joint": (-2.7227, -0.8378)
        }

        # ------------------------------Events------------------------------

        self.events.randomize_rigid_base_mass.params["mass_distribution_params"] = (-2.0, 5.0)

        # self.events.randomize_rigid_body_material = None
        # self.events.randomize_joint_parameters = None
        # self.events.randomize_rigid_base_mass = None
        # self.events.randomize_rigid_body_mass = None
        # self.events.randomize_base_com_position = None
        # self.events.randomize_com_positions = None
        # self.events.randomize_apply_external_force_torque = None
        # self.events.randomize_actuator_gains = None
        # self.events.randomize_push_robot = None

        # ------------------------------Rewards------------------------------

        self.rewards.base_linear_velocity.weight = 10.0
        self.rewards.base_linear_velocity.params["std"] = 0.25
        self.rewards.base_angular_velocity.weight = 5.0
        self.rewards.base_angular_velocity.params["std"] = 0.25
        self.rewards.foot_clearance.weight = 1.0 
        self.rewards.foot_clearance.params["std"] = 0.05
        self.rewards.foot_clearance.params["tanh_mult"] = 5.0
        self.rewards.foot_clearance.params["target_height"] = 0.06 + 0.023
        # self.rewards.gait.weight = 1.0
        # self.rewards.gait.params["std"] = 0.1
        self.rewards.air_time.weight = 3.0
        self.rewards.air_time.params["mode_time"] = 0.4
        self.rewards.action_rate.weight = -0.1
        self.rewards.action_smoothness.weight = -0.05
        self.rewards.air_time_variance.weight = -1.0
        self.rewards.base_orientation.weight = 0.5
        self.rewards.base_orientation.params["std"] = 0.05
        self.rewards.foot_slip.weight = -2.5
        self.rewards.joint_acc.weight = -2.5e-7
        self.rewards.joint_torques.weight = -2.0e-4
        self.rewards.joint_vel.weight = -1.0e-3
        self.rewards.joint_vel.params["asset_cfg"].joint_names = [".*_hip_joint", ".*_thigh_joint"]
        self.rewards.joint_power.weight = -2.0e-5    
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.base_height.weight = 0.5
        self.rewards.base_height.params["std"] = 0.05
        self.rewards.base_height.params["target_height"] = 0.35
        self.rewards.stand_still.weight = -5.0
        self.rewards.stand_still_vel.weight = -1.0
        self.rewards.feet_contact_without_cmd.weight = 0.25
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_torque_limits.weight = -0.1
        self.rewards.lin_vel_z_l2.weight = -5.0
        # self.rewards.ang_vel_xy_l2.weight = -0.5
        self.rewards.ang_vel_x_l2.weight = -0.1
        self.rewards.ang_vel_y_l2.weight = -2.0
        # self.rewards.feet_distance_y_exp.weight = 3.0
        # self.rewards.feet_distance_y_exp.params["stance_width"] = 0.2
        # self.rewards.feet_distance_y_exp.params["std"] = 0.025

        # ------------------------------Terminations------------------------------

        self.terminations.illegal_contact = None
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base", ".*_hip", ".*_thigh", ".*_calf"]

        # ------------------------------Commands------------------------------

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        
        if self.__class__.__name__ == "Go2RoughEnvCfg":
            self.disable_zero_weight_rewards()


class Go2RoughEnvCfgTeacher(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = ObservationsCfg.TeacherCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names


class Go2RoughEnvCfgDistillation(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.teacher = ObservationsCfg.TeacherCfg()
        self.observations.policy = ObservationsCfg.PolicyCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names