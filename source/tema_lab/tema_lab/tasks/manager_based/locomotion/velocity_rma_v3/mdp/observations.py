# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def phase_obs(env: ManagerBasedRLEnv, f_b: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    phase = 2 * torch.pi * f_b * env.episode_length_buf.unsqueeze(1) * env.step_dt 
    phase_tensor = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    return phase_tensor


def foot_phase_obs(env: ManagerBasedRLEnv, f_b: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    gait_pattern = torch.tensor([0, torch.pi, torch.pi, 0]).to(env.device)
    phase = gait_pattern.repeat(env.num_envs, 1) + 2 * torch.pi * f_b * env.episode_length_buf.unsqueeze(1) * env.step_dt 
    phase_tensor = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    return phase_tensor


def coms_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_com_pose_b[:, asset_cfg.body_ids, :3].reshape(env.num_envs, -1)


def contact_forces_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)


def contact_states_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    return is_contact.float()


def body_masses_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_masses"):
        env.body_masses = asset.root_physx_view.get_masses().to(env.device)

    return (env.body_masses[:, asset_cfg.body_ids] / asset.data.default_mass[:, asset_cfg.body_ids].to(env.device))
    # return env.body_masses[:, asset_cfg.body_ids]


def material_props_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "material_props"):
        env.material_props = asset.root_physx_view.get_material_properties().to(env.device)

    return env.material_props[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def body_inertias_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_inertias"):
        env.body_inertias = asset.root_physx_view.get_inertias().to(env.device)

    body_inertias = env.body_inertias[:, :, [0, 4, 8]] / asset.data.default_inertia[:, :, [0, 4, 8]].to(env.device)
    return body_inertias[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def body_external_force_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:  
    asset: Articulation = env.scene[asset_cfg.name]
    return asset._external_force_b[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def body_external_torque_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:  
    asset: Articulation = env.scene[asset_cfg.name]
    return asset._external_torque_b[:, asset_cfg.body_ids].reshape(env.num_envs, -1)