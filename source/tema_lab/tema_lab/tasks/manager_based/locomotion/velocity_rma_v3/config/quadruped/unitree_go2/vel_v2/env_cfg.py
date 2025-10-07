from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
import tema_lab.tasks.manager_based.locomotion.velocity_rma_v3.mdp as mdp

from tema_lab.tasks.manager_based.locomotion.velocity_rma_v3.velocity_env_cfg import LocomotionVelocityRoughEnvCfg as BaseCfg

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TeacherCfg(ObsGroup):
        """Observations for policy group."""

        # Observed Information
        imu_ang_vel = ObsTerm(
            func=base_mdp.imu_ang_vel,
            scale=1.0,
            clip=(-100, 100),
        )
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, 
            scale=1.0,
            clip=(-100, 100)
        )
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0,
            clip=(-100, 100)
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            scale=1.0,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            scale=1.0,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=base_mdp.last_action,
            scale=1.0,
            clip=(-100, 100),
        )

        # Privileged Information  
        height_scan_base = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_base"), "offset": 0.0},
            scale=3.0,
            clip=(-1.0, 1.0)
        )
        height_scan_FR = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_FR_foot"), "offset": 0.0237},
            scale=10.0,
            clip=(-1.0, 1.0)
        )
        height_scan_FL = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_FL_foot"), "offset": 0.0237},
            scale=10.0,
            clip=(-1.0, 1.0)
        )
        height_scan_RR = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_RR_foot"), "offset": 0.0237},
            scale=10.0,
            clip=(-1.0, 1.0)
        )
        height_scan_RL = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_RL_foot"), "offset": 0.0237},
            scale=10.0,
            clip=(-1.0, 1.0)
        )

        base_lin_vel = ObsTerm(
            func=base_mdp.base_lin_vel,
            scale=0.1,
            clip=(-100, 100)
        )
        contact_forces = ObsTerm(
            func=mdp.contact_forces_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
            scale=0.025,
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
        body_masses = ObsTerm(
            func=mdp.body_masses_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"])},
            scale=1.0,
            clip=(-100, 100),
        )
        coms = ObsTerm(
            func=mdp.coms_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            scale=10.0,
            clip=(-100, 100),
        )
        base_external_force = ObsTerm(
            func=mdp.body_external_force_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            scale=0.2,
            clip=(-100, 100),
        )
        base_external_torque = ObsTerm(
            func=mdp.body_external_torque_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            scale=0.2,
            clip=(-100, 100),
        )
        body_incoming_wrench = ObsTerm(
            func=base_mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip", ".*_thigh", ".*_calf"])},
            scale=0.025,
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  
            self.history_length = 5
            self.flatten_history_dim = False          
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        imu_ang_vel = ObsTerm(
            func=base_mdp.imu_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),  
            scale=1.0,
            clip=(-100, 100)
        )
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), 
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=base_mdp.last_action,
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


@configclass
class LocomotionVelocityRoughEnvCfg(BaseCfg):    
    observations: ObservationsCfg = ObservationsCfg()

    # def __post_init__(self):
    #     super().__post_init__()

    #     self.scene.height_scanner_base.pattern_cfg.resolution = 0.1
    #     self.scene.height_scanner_base.pattern_cfg.size = (0.0, 0.0)
    #     self.scene.height_scanner_FR_foot.pattern_cfg.resolution = 0.1
    #     self.scene.height_scanner_FR_foot.pattern_cfg.size = (0.4, 0.4)
    #     self.scene.height_scanner_FL_foot.pattern_cfg.resolution = 0.1
    #     self.scene.height_scanner_FL_foot.pattern_cfg.size = (0.4, 0.4)
    #     self.scene.height_scanner_RR_foot.pattern_cfg.resolution = 0.1
    #     self.scene.height_scanner_RR_foot.pattern_cfg.size = (0.4, 0.4)
    #     self.scene.height_scanner_RL_foot.pattern_cfg.resolution = 0.1
    #     self.scene.height_scanner_RL_foot.pattern_cfg.size = (0.4, 0.4)