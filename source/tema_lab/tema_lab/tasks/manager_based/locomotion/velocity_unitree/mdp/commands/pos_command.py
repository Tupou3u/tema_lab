from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .command_cfg import UniformPositionCommandCfg


class UniformPositionCommand(CommandTerm):
    cfg: UniformPositionCommandCfg

    def __init__(self, cfg: UniformPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_scanner: RayCaster = env.scene[cfg.sensor_name]

        # crete buffers to store the command
        # -- command: height, roll, pitch
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_default_pos_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.h_mean = (self.cfg.ranges.height[0] + self.cfg.ranges.height[1]) / 2

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_default_pos_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.pos_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        adjusted_target_height = self.pos_command_b[:, 0] + torch.mean(self.height_scanner.data.ray_hits_w[..., 2], dim=1)
        curr_roll, curr_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        curr_roll, curr_pitch = math_utils.wrap_to_pi(curr_roll), math_utils.wrap_to_pi(curr_pitch)
        # logs data
        self.metrics["error_height"] += (
            torch.abs(adjusted_target_height - self.robot.data.root_pos_w[:, 2]) / max_command_step
        )
        self.metrics["error_roll"] += (
            torch.abs(self.pos_command_b[:, 1] - curr_roll) / max_command_step
        )
        self.metrics["error_pitch"] += (
            torch.abs(self.pos_command_b[:, 2] - curr_pitch) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        self.pos_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.roll)
        self.pos_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pitch)
        self.is_default_pos_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_pos_envs

    def _update_command(self):
        # if self.cfg.ranges.height[1] > self.cfg.ranges.height[0]:
        #     scale = torch.where(
        #         self.pos_command_b[:, 0] < self.h_mean,
        #         (self.pos_command_b[:, 0] - self.cfg.ranges.height[0]) / (self.h_mean - self.cfg.ranges.height[0]),
        #         1.0 - (self.pos_command_b[:, 0] - self.h_mean) / (self.cfg.ranges.height[1] - self.h_mean)
        #     )
        #     self.pos_command_b[:, 1] = self.pos_command_b[:, 1].clamp(
        #         self.cfg.ranges.roll[0] * scale, self.cfg.ranges.roll[1] * scale
        #     )
        #     self.pos_command_b[:, 2] = self.pos_command_b[:, 2].clamp(
        #         self.cfg.ranges.pitch[0] * scale, self.cfg.ranges.pitch[1] * scale
        #     )        

        default_pos_env_ids = self.is_default_pos_env.nonzero(as_tuple=False).flatten()
        self.pos_command_b[default_pos_env_ids, 0] = self.cfg.default_height + torch.mean(self.height_scanner.data.ray_hits_w[default_pos_env_ids, :, 2], dim=-1)
        self.pos_command_b[default_pos_env_ids, 1:] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_pos_visualizer"):
                # -- goal
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)
                # -- current
                self.current_pos_visualizer = VisualizationMarkers(self.cfg.current_pos_visualizer_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.current_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.current_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        goal_pos_w = self.robot.data.root_pos_w.clone()
        curr_roll, curr_pitch, curr_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        curr_roll, curr_pitch, curr_yaw = math_utils.wrap_to_pi(curr_roll), math_utils.wrap_to_pi(curr_pitch), math_utils.wrap_to_pi(curr_yaw)
        goal_pos_w[:, 2] = self.pos_command_b[:, 0] + torch.mean(self.height_scanner.data.ray_hits_w[..., 2], dim=1) + 0.2
        goal_quat_w = math_utils.quat_from_euler_xyz(
            self.pos_command_b[:, 1],
            self.pos_command_b[:, 2],
            curr_yaw
        )
        self.goal_pos_visualizer.visualize(goal_pos_w, goal_quat_w)
        # -- current body pose
        current_pos_w = self.robot.data.root_pos_w.clone()
        current_pos_w[:, 2] += 0.2
        self.current_pos_visualizer.visualize(current_pos_w, self.robot.data.root_quat_w)