import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import tema_lab.tasks.manager_based.locomotion.velocity_rma_v3.mdp as mdp
from .....velocity_env_cfg import CommandsCfg as BaseCommandsCfg
from .....velocity_env_cfg import ObservationsCfg as BaseObservationsCfg
from .....velocity_env_cfg import RewardsCfg as BaseRewardsCfg
from .....velocity_env_cfg import LocomotionVelocityRoughEnvCfg as BaseLocomotionVelocityRoughEnvCfg


##
# MDP settings
##


@configclass
class CommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""
    
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
class ObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    # observation groups
    policy: BaseObservationsCfg.TeacherCfg = BaseObservationsCfg.TeacherCfg()

    # teacher: BaseObservationsCfg.TeacherCfg = BaseObservationsCfg.TeacherCfg()
    # policy: BaseObservationsCfg.PolicyCfg = BaseObservationsCfg.PolicyCfg()


@configclass
class RewardsCfg(BaseRewardsCfg):    
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


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(BaseLocomotionVelocityRoughEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
