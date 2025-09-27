from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .joint_actions import JointPositionFTGAction


@configclass
class JointPositionFTGActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = JointPositionFTGAction
    joint_names: list[str] = MISSING
    preserve_order: bool = False
    f_b: float = 5.0
    base_h: float = 0.35
    h_max: float = 0.06
    f_alpha: float = 0.01
    alpha: float = 0.01
    mode: str = "ftg"