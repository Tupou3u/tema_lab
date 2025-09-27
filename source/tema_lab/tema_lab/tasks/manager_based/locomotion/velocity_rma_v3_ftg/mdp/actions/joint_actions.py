from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from . import actions_cfg
    
    
from torch import arctan2, arccos, pi, sin, cos, tensor, sqrt
from dataclasses import dataclass


@dataclass
class Go2:
    L1 = 0.067
    L2 = 0.213
    L3 = 0.210
    L4 = 0.094
    BASE_LEN = 0.257
    BASE_WIDTH = 0.0935
        

class Go2LegsUtils:
    def __init__(self, env: ManagerBasedRLEnv):
        self.device = env.device
        self._pHip2B = tensor([
            [Go2.L1 + Go2.BASE_LEN / 2,  Go2.BASE_WIDTH / 2, 0.0],
            [Go2.L1 + Go2.BASE_LEN / 2, -Go2.BASE_WIDTH / 2, 0.0],
            [-Go2.L1 - Go2.BASE_LEN / 2,  Go2.BASE_WIDTH / 2, 0.0],
            [-Go2.L1 - Go2.BASE_LEN / 2, -Go2.BASE_WIDTH / 2, 0.0]
        ]).to(self.device)
        self._sideSign = tensor([1, -1, 1, -1]).to(self.device)

    def h2b(self, pos_h):
        return pos_h + self._pHip2B
    
    def b2h(self, pos_b):
        return pos_b - self._pHip2B

    def forward_kinematics(self, q, frame='HIP'):
        if frame not in {'HIP', 'BODY'}:
            raise ValueError("Frame must be HIP or BODY")
                
        l1 = Go2.L4 * self._sideSign
        l2 = -Go2.L2
        l3 = -Go2.L3
        
        s1 = sin(q[:, :, 0])
        s2 = sin(q[:, :, 1])
        s3 = sin(q[:, :, 2])
        c1 = cos(q[:, :, 0])
        c2 = cos(q[:, :, 1])
        c3 = cos(q[:, :, 2])
        
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3
        
        pEe2H = torch.zeros_like(q).to(self.device)
        pEe2H[:, :, 0] = l3 * s23 + l2 * s2
        pEe2H[:, :, 1] = l3 * s1 * c23 + l1 * c1 + l2 * c2 * s1
        pEe2H[:, :, 2] = l3 * c1 * c23 - l1 * s1 + l2 * c1 * c2
        
        if frame == 'BODY':
            return self.h2b(pEe2H)
        else:
            return pEe2H
        
    def inverse_kinematics(self, pEe, frame='HIP'):
        if frame not in {'HIP', 'BODY'}:
            raise ValueError("Frame must be HIP or BODY")
        
        if frame == 'BODY':
            pEe2H = pEe - self._pHip2B
        else:
            pEe2H = pEe
            
        px, py, pz = pEe2H[:, :, 0], pEe2H[:, :, 1], pEe2H[:, :, 2]
        
        b2y = Go2.L4 * self._sideSign
        b3z = -Go2.L2
        b4z = -Go2.L3
        
        a = Go2.L4
        c = sqrt(px**2 + py**2 + pz**2)
        b = sqrt(c**2 - a**2)
        
        q1 = self.q1_ik(py, pz, b2y)
        q3 = self.q3_ik(b3z, b4z, b)
        q2 = self.q2_ik(q1, q3, px, py, pz, b3z, b4z)
        
        return torch.stack([-q1, q2, q3], dim=-1)

    def q1_ik(self, py, pz, l1):
        L = sqrt(py**2 + pz**2 - l1**2)
        return arctan2(pz * l1 + py * L, py * l1 - pz * L)

    def q3_ik(self, b3z, b4z, b):
        temp = (b3z**2 + b4z**2 - b ** 2) / (2 * abs(b3z * b4z))
        temp = torch.clamp(temp, -1, 1)
        return -(pi - arccos(temp))

    def q2_ik(self, q1, q3, px, py, pz, b3z, b4z):
        a1 = py * sin(q1) - pz * cos(q1)
        a2 = px
        m1 = b4z * sin(q3)
        m2 = b3z + b4z * cos(q3)
        return arctan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1)

    
class FTG:
    GAIT_PATTERN = [0, pi, pi, 0]
    
    def __init__(
        self, 
        env: ManagerBasedRLEnv,
        f_b: float,
        f_alpha: float,
        base_h: float,
        h_max: float
    ):
        self._env = env
        self.f_b = f_b
        self.f_alpha = f_alpha
        self.base_h = base_h
        self.h_max = h_max
        self.gait_pattern = tensor(self.GAIT_PATTERN).repeat(env.num_envs, 1).to(self._env.device)
        self.count = torch.zeros(self._env.num_envs, dtype=torch.long).to(self._env.device)
        self.phase = 2 * pi * self.f_b * self.count.unsqueeze(1) * env.step_dt 
        self.foot_phase = self.gait_pattern + self.phase
        self.stop_count = torch.zeros(self._env.num_envs, dtype=torch.long).to(self._env.device)
        self.on_ground_mask = torch.zeros((self._env.num_envs, 4), dtype=torch.bool).to(self._env.device)
    
    def get_phase(self, f_pi):
        if not hasattr(self._env, "reset_buf") or self._env.reset_buf is None:
            self._env.reset_buf = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.bool)
                
        self.phase = self.f_b * self.count.unsqueeze(1) * self._env.step_dt
        foot_phase = (self.gait_pattern + 2 * pi * (self.f_alpha * f_pi + self.phase)) % (2 * pi)
        stop_timeout = self.stop_count * self._env.step_dt > 2 / self.f_b
        
        self.foot_phase = torch.where(
            self.on_ground_mask * stop_timeout.unsqueeze(1),
            self.foot_phase,
            foot_phase
        )
                 
        # cmd = self._env.command_manager.get_command("base_velocity")
        cmd = self._env.observation_manager.compute()['policy'][:, 16:19]
        zero_cmd_envs = torch.norm(cmd, dim=1) == 0.0
        self.stop_count = torch.where(
            zero_cmd_envs, 
            self.stop_count + 1, 
            0
        )
        self.count += 1
        self.on_ground_mask = (self.foot_phase >= torch.pi) & (self.foot_phase <= 2 * torch.pi)
        all_feet_on_ground = self.on_ground_mask.all(dim=-1)
        reset_env = self._env.reset_buf | all_feet_on_ground
        reset_env_ids = reset_env.nonzero(as_tuple=False).squeeze(-1)
        self.count[reset_env_ids] = 0  
               
        return self.foot_phase
        
    def get_ftg_h(self, phi):
        return torch.where(
            torch.logical_and(phi >= 0, phi <= pi / 2), 
            self.h_max * (-2 * (2 * phi / pi) ** 3 + 3 * (2 * phi / pi) ** 2),
            torch.where(
                torch.logical_and(phi > pi / 2, phi <= pi),
                self.h_max * (2 * (2 * phi / pi - 1) ** 3 - 3 * (2 * phi / pi - 1) ** 2 + 1),
                0
            )
        )
        
    def get_ftg_pos(self, phi):
        h = self.get_ftg_h(phi)
        p_hip = torch.zeros(self._env.num_envs, 4, 3).to(self._env.device)
        p_hip[:, :, 1] = tensor([Go2.L4, -Go2.L4, Go2.L4, -Go2.L4])
        p_hip[:, :, 2] = -(self.base_h - h)
        return p_hip
        # roll, pitch, _ = math_utils.euler_xyz_from_quat(self.env.scene['imu'].data.quat_w)
        # rx = self.rot_x(roll)
        # ry = self.rot_y(pitch)
        # return rx @ ry @ p_hip
        
    def rot_x(self, q):
        rot = torch.zeros(self._env.num_envs, 3, 3).to(self._env.device)
        rot[:, 0, 0] = 1
        rot[:, 1, 1], rot[:, 1, 2] = cos(q), -sin(q)
        rot[:, 2, 1], rot[:, 2, 2] = sin(q), cos(q)
        return rot
    
    def rot_y(self, q):
        rot = torch.zeros(self._env.num_envs, 3, 3).to(self._env.device)
        rot[:, 0, 0], rot[:, 0, 2] = cos(q), sin(q)
        rot[:, 1, 1] = 1
        rot[:, 2, 0], rot[:, 2, 2] = -sin(q), cos(q)
        return rot


class JointPositionFTGAction(ActionTerm):

    cfg: actions_cfg.JointPositionFTGActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointPositionFTGActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self._env = env
        self._legs = Go2LegsUtils(env)
        self._env.ftg = FTG(env, f_b=cfg.f_b, base_h=cfg.base_h, h_max=cfg.h_max, f_alpha=cfg.f_alpha)
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_actions = 16
        
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions
        f_pi = self._processed_actions[:, :4]
        joint_actions = self._processed_actions[:, 4:]
        phi = self._env.ftg.get_phase(f_pi)
        a_ftg = self._env.ftg.get_ftg_pos(phi)
        
        if self.cfg.mode == "ftg":
            processed_actions = self.cfg.alpha * joint_actions.reshape(self._env.num_envs, 4, 3) + a_ftg
            processed_actions = self._legs.inverse_kinematics(processed_actions).reshape(self._env.num_envs, -1)
            
        elif self.cfg.mode == "joint":
            processed_actions = self._legs.inverse_kinematics(a_ftg).reshape(self._env.num_envs, -1) + self.cfg.alpha * joint_actions

        self._processed_actions = torch.where(torch.isnan(processed_actions), 0.0, processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        
    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)