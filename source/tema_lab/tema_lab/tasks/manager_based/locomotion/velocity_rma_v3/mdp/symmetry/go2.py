from __future__ import annotations

import torch
from torch import arctan2, arcsin, arccos, pi, sin, cos, tensor, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


L1 = 0.067
L2 = 0.213
L3 = 0.210
L4 = 0.094
BASE_LEN = 0.257
BASE_WIDTH = 0.0935

_pHip2B = tensor([
    [ L1 + BASE_LEN / 2,  BASE_WIDTH / 2, 0.0],
    [ L1 + BASE_LEN / 2, -BASE_WIDTH / 2, 0.0],
    [-L1 - BASE_LEN / 2,  BASE_WIDTH / 2, 0.0],
    [-L1 - BASE_LEN / 2, -BASE_WIDTH / 2, 0.0]
])
_sideSign = tensor([1, -1, 1, -1])


def forward_kinematics(q, frame='HIP'):
    if frame not in {'HIP', 'BODY'}:
        raise ValueError("Frame must be HIP or BODY")
            
    l1 = L4 * _sideSign
    l2 = -L2
    l3 = -L3
    
    s1 = sin(q[:, :, 0])
    s2 = sin(q[:, :, 1])
    s3 = sin(q[:, :, 2])
    c1 = cos(q[:, :, 0])
    c2 = cos(q[:, :, 1])
    c3 = cos(q[:, :, 2])
    
    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3
    
    pEe2H = torch.zeros_like(q)
    pEe2H[:, :, 0] = l3 * s23 + l2 * s2
    pEe2H[:, :, 1] = l3 * s1 * c23 + l1 * c1 + l2 * c2 * s1
    pEe2H[:, :, 2] = l3 * c1 * c23 - l1 * s1 + l2 * c1 * c2
    
    if frame == 'BODY':
        return pEe2H + _pHip2B
    else:
        return pEe2H
    
def inverse_kinematics(pEe, frame='HIP'):
    if frame not in {'HIP', 'BODY'}:
        raise ValueError("Frame must be HIP or BODY")
    
    if frame == 'BODY':
        pEe2H = pEe - _pHip2B
    else:
        pEe2H = pEe
        
    px, py, pz = pEe2H[:, :, 0], pEe2H[:, :, 1], pEe2H[:, :, 2]
    
    b2y = L4 * _sideSign
    b3z = -L2
    b4z = -L3
    
    a = L4
    c = sqrt(px**2 + py**2 + pz**2)
    b = sqrt(c**2 - a**2)
    
    q1 = q1_ik(py, pz, b2y)
    q3 = q3_ik(b3z, b4z, b)
    q2 = q2_ik(q1, q3, px, py, pz, b3z, b4z)
    
    return torch.stack([-q1, q2, q3], dim=-1)

def q1_ik(py, pz, l1):
    L = sqrt(py**2 + pz**2 - l1**2)
    return arctan2(pz * l1 + py * L, py * l1 - pz * L)

def q3_ik(b3z, b4z, b):
    temp = (b3z**2 + b4z**2 - b ** 2) / (2 * abs(b3z * b4z))
    temp = torch.clamp(temp, -1, 1)
    return -(pi - arccos(temp))

def q2_ik(q1, q3, px, py, pz, b3z, b4z):
    a1 = py * sin(q1) - pz * cos(q1)
    a2 = px
    m1 = b4z * sin(q3)
    m2 = b3z + b4z * cos(q3)
    return arctan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        num_envs = obs.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = torch.zeros(num_envs * 4, obs.shape[1], device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- left-right
        obs_aug[num_envs : 2 * num_envs] = _transform_obs_left_right(env.unwrapped, obs, obs_type)
        # -- front-back
        obs_aug[2 * num_envs : 3 * num_envs] = _transform_obs_front_back(env.unwrapped, obs, obs_type)
        # -- diagonal
        obs_aug[3 * num_envs :] = _transform_obs_front_back(env.unwrapped, obs_aug[num_envs : 2 * num_envs])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(num_envs * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = _transform_actions_left_right(env.unwrapped, actions)
        # -- front-back
        actions_aug[2 * num_envs : 3 * num_envs] = _transform_actions_front_back(env.unwrapped, actions)
        # -- diagonal
        actions_aug[3 * num_envs :] = _transform_actions_front_back(env.unwrapped, actions_aug[num_envs : 2 * num_envs])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # imu lin acc
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # imu ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_go2_joints_left_right(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_go2_joints_left_right(obs[:, 24:36])
    # last actions
    obs[:, 36:48] = _switch_go2_joints_left_right(obs[:, 36:48])

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)
        
    # contact forces
    obs[:, 235:247] = _switch_go2_joints_left_right(obs[:, 235:247], sign=False) * torch.tensor([1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1], device=device)
    
    # contact_states (without base) 
    obs[:, 248:260] = _switch_go2_joints_left_right(obs[:, 248:260], sign=False) * torch.tensor([1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1], device=device)
    
    # masses (without base)
    obs[:, 261:273] = _switch_go2_joints_left_right(obs[:, 261:273], sign=False)
    
    # foot_ground_friction
    obs[:, 276:288] = _switch_go2_joints_left_right(obs[:, 276:288], sign=False)

    return obs


def _transform_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # imu lin acc
    obs[:, :3] = obs[:, :3] * torch.tensor([-1, 1, 1], device=device)
    # imu ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([-1, 1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([-1, 1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_go2_joints_front_back2(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_go2_joints_front_back(obs[:, 24:36])
    # last actions
    last_q = env.scene["robot"].data.default_joint_pos + 0.25 * obs[:, 36:48]
    last_q_sym = _switch_go2_joints_front_back2(last_q)
    obs[:, 36:48] = (last_q_sym - env.scene["robot"].data.default_joint_pos) / 0.25

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)
        
    # contact forces
    obs[:, 235:247] = _switch_go2_joints_front_back(obs[:, 235:247], sign=False) * torch.tensor([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], device=device)
    
    # contact_states (without base) 
    obs[:, 248:260] = _switch_go2_joints_front_back(obs[:, 248:260], sign=False) * torch.tensor([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], device=device)
    
    # masses (without base)
    obs[:, 261:273] = _switch_go2_joints_front_back(obs[:, 261:273], sign=False)
    
    # foot_ground_friction
    obs[:, 276:288] = _switch_go2_joints_front_back(obs[:, 276:288], sign=False)

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right(actions[:])
    return actions


def _transform_actions_front_back(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    q = env.scene["robot"].data.default_joint_pos + 0.25 * actions
    q_sym = _switch_go2_joints_front_back2(q)
    actions = (q_sym - env.scene["robot"].data.default_joint_pos) / 0.25
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

Correspondingly, the joint ordering for the Go2 robot is:

* FR = left front --> [0, 1, 2]
* FL = left hind --> [3, 4, 5]
* RR = right front --> [6, 7, 8]
* RL = right hind --> [9, 10, 11]

"""


def _switch_go2_joints_left_right(joint_data: torch.Tensor, sign: bool = True) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 1, 2, 6, 7, 8]] = joint_data[..., [3, 4, 5, 9, 10, 11]]
    # right <-- left
    joint_data_switched[..., [3, 4, 5, 9, 10, 11]] = joint_data[..., [0, 1, 2, 6, 7, 8]]

    # Flip the sign of the HIP joints
    if sign:
        joint_data_switched[..., [0, 3, 6, 9]] *= -1.0

    return joint_data_switched


def _switch_go2_joints_front_back(joint_data: torch.Tensor, sign: bool = True) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # front <-- hind
    joint_data_switched[..., [0, 1, 2, 3, 4, 5]] = joint_data[..., [6, 7, 8, 9, 10, 11]]
    # hind <-- front
    joint_data_switched[..., [6, 7, 8, 9, 10, 11]] = joint_data[..., [0, 1, 2, 3, 4, 5]]
    
    # Flip the sign of the THIGH and CALF joints
    if sign:
        joint_data_switched[..., [1, 2, 4, 5, 7, 8, 10, 11]] *= -1

    return joint_data_switched


def _switch_go2_joints_front_back2(q: torch.Tensor) -> torch.Tensor:
    q = q.view(q.shape[0], 4, 3)
    pos_h = forward_kinematics(q)
    pos_h_sym = torch.zeros_like(pos_h)
    pos_h_sym[:, [0, 1]] = pos_h[:, [2, 3]]
    pos_h_sym[:, [2, 3]] = pos_h[:, [0, 1]]
    pos_h_sym *= tensor([
        [-1, 1, 1],
        [-1, 1, 1],
        [-1, 1, 1],
        [-1, 1, 1],
    ])
    q_sym = inverse_kinematics(pos_h_sym)
    return q_sym.view(q.shape[0], -1)