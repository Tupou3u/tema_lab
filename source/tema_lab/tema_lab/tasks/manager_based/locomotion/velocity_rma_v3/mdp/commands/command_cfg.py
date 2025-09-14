import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .pos_command import UniformPositionCommand


@configclass
class UniformPositionCommandCfg(CommandTermCfg):
    """Configuration for the uniform position command generator (h, roll, pitch, yaw)."""

    class_type: type = UniformPositionCommand
    asset_name: str = MISSING
    sensor_name: str = MISSING
    rel_default_pos_envs: float = 0.0
    default_height: float = 0.35

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    
    goal_pos_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pos_goal"
    )

    current_pos_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pos_current"
    )

    goal_pos_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    current_pos_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    goal_pos_visualizer_cfg.markers["frame"].visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    current_pos_visualizer_cfg.markers["frame"].visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))