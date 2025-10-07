from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence

import torch
from isaaclab.sensors import RayCaster
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import TerrainVelocityCommandCfg


class TerrainVelocityCommand(UniformVelocityCommand):
    cfg: TerrainVelocityCommandCfg
    
    def __init__(self, cfg: TerrainVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
    def _update_command(self):
        super()._update_command()
        x = self.robot.data.root_pos_w[:, 0]
        y = self.robot.data.root_pos_w[:, 1]
        row = int((x + (self._env.scene["terrain"].terrain_generator.num_rows / 2) * self._env.scene["terrain"].terrain_generator.size[0]) // self._env.scene["terrain"].terrain_generator.size[0])
        col = int((y + (self._env.scene["terrain"].terrain_generator.num_cols / 2) * self._env.scene["terrain"].terrain_generator.size[1]) // self._env.scene["terrain"].terrain_generator.size[1])
        terrain_types = self._env.scene["terrain"]._terrain_types[row, col]
        alpha = torch.where(
            "stairs" in terrain_types,
            0.5, 
            1.0
        )