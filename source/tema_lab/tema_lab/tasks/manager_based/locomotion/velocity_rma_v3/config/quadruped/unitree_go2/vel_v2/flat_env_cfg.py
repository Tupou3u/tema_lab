from isaaclab.utils import configclass
from .rough_env_cfg import Go2RoughEnvCfg, ObservationsCfg


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        if self.curriculum:
            self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        
        if self.__class__.__name__ == "Go2FlatEnvCfg":
            self.disable_zero_weight_rewards()

        
class Go2FlatEnvCfgTeacher(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = ObservationsCfg.TeacherCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names


class Go2FlatEnvCfgDistillation(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = ObservationsCfg.PolicyCfg()
        self.observations.teacher = ObservationsCfg.TeacherCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names
        self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.scene.robot.joint_sdk_names