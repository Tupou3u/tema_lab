from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity_rma_v3_ftg.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import *  # isort: skip


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
    h_max = 0.08

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

        # ------------------------------Observations------------------------------
        
        if hasattr(self.observations, 'teacher'):
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            
        else:
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------

        self.actions.joint_pos_ftg.h_max = self.h_max
        self.actions.joint_pos_ftg.joint_names = self.joint_names

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
        
        self.rewards.base_linear_velocity.weight = 5.0
        self.rewards.base_angular_velocity.weight = 3.0
        # self.rewards.foot_clearance.weight = 3.0 
        # self.rewards.gait.weight = 3.0
        # self.rewards.air_time.weight = 5.0
        self.rewards.action_smoothness.weight = -0.1
        self.rewards.action_smoothness2.weight = -0.1
        self.rewards.air_time_variance.weight = -1.0
        self.rewards.base_orientation.weight = 1.0
        self.rewards.foot_slip.weight = -0.5
        self.rewards.joint_acc.weight = -1.0e-5
        # self.rewards.joint_pos.weight = -0.7
        # self.rewards.joint_torques.weight = -5.0e-4
        # self.rewards.joint_vel.weight = -0.1
        self.rewards.joint_vel.params["asset_cfg"].joint_names = ".*_hip_joint"
        self.rewards.joint_power.weight = -1.0e-4    
        # self.rewards.hip_pos.weight = -1.0
        
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.base_height_exp.weight = 1.0
        # self.rewards.contact_forces.weight = -1.5e-4
        # self.rewards.stand_still.weight = -5.0
        # self.rewards.stand_still_vel.weight = -1.0
        # self.rewards.feet_contact_without_cmd.weight = 0.25
        # self.rewards.feet_force_variance_without_cmd.weight = 1.0
        # self.rewards.joint_pos_limits.weight = -5.0
        # self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_torque_limits.weight = -0.1
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.5
        # self.rewards.joint_mirror.weight = -0.1  # without hip
        # self.rewards.feet_distance_y_exp.weight = 2.0
        # self.rewards.feet_distance_xy_exp.weight = 1.0
        
        self.rewards.base_linear_velocity.params["std"] = 1.0
        self.rewards.base_angular_velocity.params["std"] = 2.0
        self.rewards.foot_clearance.params["std"] = 0.01
        self.rewards.foot_clearance.params["tanh_mult"] = 5.0
        self.rewards.foot_clearance.params["target_height"] = self.h_max + 0.023
        self.rewards.base_height_exp.params["std"] = 0.05
        self.rewards.base_height_exp.params["target_height"] = 0.35
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.25
        
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        
        # ------------------------------Terminations------------------------------

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base", ".*_hip", ".*_thigh", ".*_calf"]
        self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------

        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        if self.__class__.__name__ == "Go2RoughEnvCfg":
            self.disable_zero_weight_rewards()
        