import math
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity_rma_v3.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrainobs['policy']['joint_pos'].flatten()
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_FR_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_FL_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_RR_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_RL_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)),
        gravity_bias=(0.0, 0.0, 0.0)
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,  # 0.02
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    
    base_pose = mdp.UniformPositionCommandCfg(
        asset_name="robot",
        sensor_name="height_scanner_base",
        rel_default_pos_envs=0.1,
        default_height=0.35,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPositionCommandCfg.Ranges(
            height=(0.1, 0.38), roll=(-math.pi / 6, math.pi / 6), pitch=(-math.pi / 6, math.pi / 6)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TeacherCfg(ObsGroup):
        """Observations for policy group."""

        # Observed Information
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            scale=1.0,
            clip=(-100, 100)
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, 
            scale=1.0,
            clip=(-100, 100)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            scale=1.0,
            clip=(-100, 100)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0,
            clip=(-100, 100)
        )
        pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_pose"},
            scale=1.0,
            clip=(-100, 100)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1.0,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=1.0,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
            scale=1.0,
            clip=(-100, 100),
        )

        # Privileged Information  
        phase = ObsTerm(
            func=mdp.phase_obs,
            params={"cycle_time": 20},
            scale=1.0,
            clip=(-100, 100),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            clip=(-100, 100)
        )
        height_scan_base = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_base")},
            scale=1.0,
            clip=(-100, 100)
        )
        height_scan_FR = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_FR_foot")},
            scale=1.0,
            clip=(-100, 100)
        )
        height_scan_FL = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_FL_foot")},
            scale=1.0,
            clip=(-100, 100)
        )
        height_scan_RR = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_RR_foot")},
            scale=1.0,
            clip=(-100, 100)
        )
        height_scan_RL = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_RL_foot")},
            scale=1.0,
            clip=(-100, 100)
        )
        contact_forces = ObsTerm(
            func=mdp.contact_forces_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
            scale=1.0,
            clip=(-100, 100),
        )
        contact_states = ObsTerm(
            func=mdp.contact_states_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"])},
            scale=1.0,
            clip=(-100, 100),
        )
        foot_ground_friction = ObsTerm(
            func=mdp.material_props_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            scale=1.0,
            clip=(-100, 100),
        )
        masses = ObsTerm(
            func=mdp.masses_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base", ".*_hip", ".*_thigh", ".*_calf", ".*_foot"])},
            scale=1.0,
            clip=(-100, 100),
        )
        coms = ObsTerm(
            func=mdp.coms_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            scale=1.0,
            clip=(-100, 100),
        )
        body_incoming_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip", ".*_thigh", ".*_calf", ".*_foot"])},
            scale=1.0,
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 33
            # self.flatten_history_dim = False
            
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),  
            scale=1.0,
            clip=(-100, 100)
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            noise=Unoise(n_min=-0.1, n_max=0.1), 
            scale=1.0,
            clip=(-100, 100)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), 
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0,
        )
        pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_pose"},
            scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 65
            self.flatten_history_dim = False

    # observation groups
    policy: TeacherCfg = TeacherCfg()

    # teacher: TeacherCfg = TeacherCfg()
    # policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base", ".*_hip", ".*_thigh", ".*_calf", ".*_foot"]),
            "mass_distribution_params": (0.8, 1.2), 
            "operation": "scale",
        },
    )

    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.8, 1.2),  
            "operation": "scale",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_distribution_params": ((-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05)),
            "operation": "add",
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-3.0, 3.0),   
            "torque_range": (-3.0, 3.0), 
        },
    )

    randomize_reset_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
        }
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2), 
            "damping_distribution_params": (0.8, 1.2),  
            "operation": "scale"
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-1.5, 1.5), 
                "y": (-1.5, 1.5),
                "z": (-1.5, 1.5)
            }
        },
    )


@configclass
class RewardsCfg:
    # -- task
    air_time = RewTerm(
        func=mdp.air_time_reward,
        weight=0.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    base_linear_velocity = RewTerm(
        func=mdp.base_linear_velocity_reward,
        weight=0.0,
        params={
            "std": 1.0, 
            "ramp_rate": 0.0, 
            "ramp_at_vel": 0.0, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    base_angular_velocity = RewTerm(
        func=mdp.base_angular_velocity_reward,
        weight=0.0,
        params={
            "std": 2.0, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    base_height = RewTerm(
        func=mdp.base_height_exp_from_command,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "std": 0.05
        },
    )
    
    base_roll = RewTerm(
        func=mdp.base_roll_exp,
        weight=0.0,
        params={
            "std": 0.05, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    base_pitch = RewTerm(
        func=mdp.base_pitch_exp,
        weight=0.0,
        params={
            "std": 0.05, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "std": 0.05,
            "tanh_mult": 5.0,
            "target_height": 0.07,
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True),
            "sensors": ["height_scanner_FR_foot", "height_scanner_FL_foot", "height_scanner_RR_foot", "height_scanner_RL_foot"],
        },
    )

    gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.3,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty, 
        weight=0.0 
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=0.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=0.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    joint_acc = RewTerm(
        func=mdp.joint_acceleration_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "linear_velocity_threshold": 0.1,
            "angular_velocity_threshold": 0.1,  
        },
    )

    joint_torques = RewTerm(
        func=mdp.joint_torques_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_vel = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=0.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -----------------------------------------------------------------------------------------------------------

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,  
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[
                    '.*_hip', 
                    '.*_thigh', 
                    '.*_calf'
                ]
            ), 
            "threshold": 1.0
        },
    )

    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 100.0
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "linear_velocity_threshold": 0.1,
            "angular_velocity_threshold": 0.1,
        },
    )
    
    feet_force_variance_without_cmd = RewTerm(
        func=mdp.feet_force_variance_without_cmd_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "linear_velocity_threshold": 0.1,
            "angular_velocity_threshold": 0.1,
            "std": 5.0
        },
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=0.0
    )
    
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, 
        weight=0.0,
        params={"soft_ratio": 1.0}
    )
    
    joint_torque_limits = RewTerm(
        func=mdp.applied_torque_limits, 
        weight=0.0
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2, 
        weight=0.0
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=0.0
    )
    
    lin_vel_w_z_l2 = RewTerm(
        func=mdp.lin_vel_w_z_l2, 
        weight=0.0
    )

    ang_vel_w_z_l2 = RewTerm(
        func=mdp.ang_vel_w_l2, 
        weight=0.0
    )
    
    joint_mirror = RewTerm(
        func=mdp.joint_mirror, 
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ]
        }
    )
    
    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "mirror_joints": [[], []],
        },
    )
    
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        try:
            if self.scene.height_scanner_FL_foot is not None:
                self.scene.height_scanner_FL_foot.update_period = self.decimation * self.sim.dt
            if self.scene.height_scanner_FR_foot is not None:
                self.scene.height_scanner_FR_foot.update_period = self.decimation * self.sim.dt
            if self.scene.height_scanner_RL_foot is not None:
                self.scene.height_scanner_RL_foot.update_period = self.decimation * self.sim.dt
            if self.scene.height_scanner_RR_foot is not None:
                self.scene.height_scanner_RR_foot.update_period = self.decimation * self.sim.dt
        except:
            pass

        try:
            self.observations.teacher.imu_lin_acc.scale = 1.0 / self.decimation
            self.observations.policy.imu_lin_acc.scale = 1.0 / self.decimation
        except:
            self.observations.policy.imu_lin_acc.scale = 1.0 / self.decimation

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)