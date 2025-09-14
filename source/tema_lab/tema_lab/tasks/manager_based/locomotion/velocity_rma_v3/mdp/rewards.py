# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# From Spot

def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_lin_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_ang_vel = torch.abs(asset.data.root_lin_vel_b[:, 2]).unsqueeze(dim=1).expand(-1, 4)
    cond = torch.logical_or(
        cmd > 0.0,
        torch.logical_or(body_ang_vel > angular_velocity_threshold, body_lin_vel > linear_velocity_threshold)
    )
    reward = torch.where(
        cond,
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


# only for grid pattern
def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    target_height: float, 
    std: float, 
    tanh_mult: float,
    sensors: list[str],
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    sensors: list[RayCaster] = [env.scene[name] for name in sensors]
    reward = torch.zeros(env.num_envs, device=env.device)

    # print(asset.data.body_pos_w[:, asset_cfg.body_ids, 2])

    for foot_idx, sensor in zip(asset_cfg.body_ids, sensors):
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = target_height
        else:
            # adjusted_target_height = target_height + ray_hits.max()
            adjusted_target_height = target_height + ray_hits[:, ray_hits.size(-1) // 2]

        foot_z_target_error = torch.square(asset.data.body_pos_w[:, foot_idx, 2] - adjusted_target_height)
        foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, foot_idx, :2], dim=1))
        reward += foot_z_target_error * foot_velocity_tanh
        
    return torch.exp(-reward / std)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.linear_velocity_threshold: float = cfg.params["linear_velocity_threshold"]
        self.angular_velocity_threshold: float = cfg.params["angular_velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_lin_vel = torch.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        body_ang_vel = torch.abs(self.asset.data.root_lin_vel_b[:, 2])
        cond = torch.logical_or(
            cmd > 0.0,
            torch.logical_or(
                body_lin_vel > self.linear_velocity_threshold,
                body_ang_vel > self.angular_velocity_threshold
            )
        )
        return torch.where(
            cond, sync_reward * async_reward - 1.0, 0.0
        ) 

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)
    

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    stand_still_scale: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cond = torch.logical_or(
        torch.logical_or(
            body_vel > linear_velocity_threshold, 
            body_ang_vel > angular_velocity_threshold
        ),
        cmd > 0.0
    )
    return torch.where(cond, reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


# Others

def foot_clearance_body_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    target_height: float, 
    tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1) > 0.1
    return reward


def feet_contact_without_cmd_reward(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
    ) -> torch.Tensor:
    """Reward feet contact"""

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    reward = torch.sum(is_contact, dim=1).float()
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    cond = torch.logical_or(
        torch.logical_or(
            body_lin_vel > linear_velocity_threshold, 
            body_ang_vel > angular_velocity_threshold
        ),
        cmd > 0.0
    )
    # print(reward)
    return torch.where(cond, 0.0, reward)


def feet_force_variance_without_cmd_reward(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
        std: float
    ) -> torch.Tensor:
    """Reward feet contact"""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    reward = torch.var(contact_forces, dim=-1)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    cond = torch.logical_or(
        torch.logical_or(
            body_lin_vel > linear_velocity_threshold, 
            body_ang_vel > angular_velocity_threshold
        ),
        cmd > 0.0
    )
    # print(contact_forces, reward)
    return torch.where(cond, 0.0, torch.exp(-reward / std))


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.05
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = target_height
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
        
    h_err = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-h_err / std)


def base_height_exp_from_command(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.05
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command("base_pose")[:, 0]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = target_height
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
        
    h_err = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-h_err / std)


def base_roll_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    std: float
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    target_roll = env.command_manager.get_command("base_pose")[:, 1]
    curr_roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    curr_roll = math_utils.wrap_to_pi(curr_roll)
    err = torch.abs(target_roll - curr_roll)
    return torch.exp(-err / std)


def base_pitch_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    std: float
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    target_pitch = env.command_manager.get_command("base_pose")[:, 2]
    _, curr_pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    curr_pitch = math_utils.wrap_to_pi(curr_pitch)
    err = torch.abs(target_pitch - curr_pitch)
    return torch.exp(-err / std)


def base_orientation_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    std: float
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll, pitch = math_utils.wrap_to_pi(roll).abs(), math_utils.wrap_to_pi(pitch).abs()
    return torch.exp(-(roll + pitch) / std)
    