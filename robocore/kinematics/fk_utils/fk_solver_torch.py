"""PyTorch forward kinematics solver.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

try:  # 延迟 import，避免未安装 torch 时影响其它模块
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("fk_solver_torch 需要 PyTorch，请先安装: pip install torch") from e

try:
    from robocore.utils.torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device(d=None):  # fallback (cpu/cuda only)
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

from robocore.utils.backend import set_backend, get_backend
from robocore.transform import (
    rpy_to_matrix,
    axis_angle_to_matrix,
    make_transform,
)

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

Tensor = torch.Tensor


class FKSolverTorch:
    """PyTorch-accelerated forward kinematics solver.
    
    Features:
    - Fast matrix operations using PyTorch
    - GPU acceleration support
    - Automatic differentiation compatible
    - Support for revolute, prismatic, and fixed joints
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize FK solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.num_dof()
        self.joint_chain = model._chain_joints
        self.actuated_joints = model._actuated
        self.base_link = model.base_link
        self.end_link = model.end_link
    
    def solve(
        self,
        q: Sequence[float] | Tensor,
        return_end_only: bool = False,
        device=None,
        dtype=torch.float64
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Compute forward kinematics (supports both single and batch).
        
        :param q: joint configuration(s).
            - Single: shape (n,) → returns dict {link_name: 4x4 tensor}
            - Batch: shape (B, n) → returns tensor [B, 4, 4] (end-effector only)
        :param return_end_only: if True, only return end-effector pose.
    :param device: torch device ('cpu', 'cuda', or None for auto; MPS removed).
        :param dtype: torch dtype (default: torch.float64).
        :return: 
            - Single mode: dict of link names to 4x4 pose matrices
            - Batch mode: tensor [B, 4, 4] of end-effector poses
        """
        device = select_device(device)
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        # Detect batch mode
        is_batch = q.ndim == 2
        
        if is_batch:
            # Batch mode: q shape [B, n]
            return self._solve_batch(q, device, dtype)
        else:
            # Single mode: q shape [n]
            return self._solve_single(q, return_end_only, device, dtype)
    
    def _solve_single(
        self,
        q: Tensor,
        return_end_only: bool,
        device,
        dtype
    ) -> Dict[str, torch.Tensor]:
        """Solve FK for single configuration."""
        if q.shape[0] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[0]}")
    def _solve_single(
        self,
        q: Tensor,
        return_end_only: bool,
        device,
        dtype
    ) -> Dict[str, torch.Tensor]:
        """Solve FK for single configuration."""
        if q.shape[0] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[0]}")
        
        q_map = {j.name: q[j.index] for j in self.actuated_joints}
        poses: Dict[str, torch.Tensor] = {
            self.base_link: torch.eye(4, dtype=dtype, device=q.device)
        }
        
        # Set backend to torch temporarily
        original_backend = get_backend()
        set_backend('torch', device=str(q.device), dtype=dtype)
        
        try:
            for joint in self.joint_chain:
                parent_pose = poses[joint.parent]
                
                # origin 变换
                R_o = rpy_to_matrix(
                    torch.tensor(joint.origin_rpy[0], dtype=dtype, device=q.device),
                    torch.tensor(joint.origin_rpy[1], dtype=dtype, device=q.device),
                    torch.tensor(joint.origin_rpy[2], dtype=dtype, device=q.device),
                )
                t_o = torch.tensor(joint.origin_xyz, dtype=dtype, device=q.device)
                T_origin = make_transform(R_o, t_o)
                
                if joint.joint_type == "revolute":
                    R_m = axis_angle_to_matrix(
                        torch.tensor(joint.axis, dtype=dtype, device=q.device),
                        q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
                    )
                    t_m = torch.zeros(3, dtype=dtype, device=q.device)
                elif joint.joint_type == "prismatic":
                    R_m = torch.eye(3, dtype=dtype, device=q.device)
                    axis_vec = torch.tensor(joint.axis, dtype=dtype, device=q.device)
                    t_m = axis_vec * q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
                else:
                    R_m = torch.eye(3, dtype=dtype, device=q.device)
                    t_m = torch.zeros(3, dtype=dtype, device=q.device)
                
                T_motion = make_transform(R_m, t_m)
                child_pose = parent_pose @ T_origin @ T_motion
                poses[joint.child] = child_pose
        finally:
            # Restore backend
            set_backend(original_backend)
        
        poses["end"] = poses.get(self.end_link, list(poses.values())[-1])
        
        if return_end_only:
            return {"end": poses["end"]}
        
        return poses
    
    def _solve_batch(
        self,
        q_batch: Tensor,
        device,
        dtype
    ) -> torch.Tensor:
        """Solve FK for batch of configurations.
        
        :param q_batch: joint configurations [B, n]
        :return: end-effector poses [B, 4, 4]
        """
        batch_size = q_batch.shape[0]
        n_joints = q_batch.shape[1]
        
        if n_joints != self.n:
            raise ValueError(f"Expected {self.n} joints, got {n_joints}")
        
        # Initialize batch of identity matrices [B, 4, 4]
        T_batch = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Get robot kinematic chain
        joint_specs = self.actuated_joints
        
        # Set backend to torch temporarily
        original_backend = get_backend()
        set_backend('torch', device=str(device), dtype=dtype)
        
        try:
            # Process each joint in the chain
            for js in joint_specs:
                # Get joint angle for this joint across all samples [B]
                theta = q_batch[:, js.index]
                
                # 1. Translation from origin
                origin_xyz = torch.tensor(js.origin_xyz, device=device, dtype=dtype)  # [3]
                
                # 2. Rotation from origin (roll-pitch-yaw) using new transform API
                origin_rpy = torch.tensor(js.origin_rpy, device=device, dtype=dtype)  # [3]
                origin_rpy_batch = origin_rpy.unsqueeze(0).repeat(batch_size, 1)  # [B, 3]
                R_origin = rpy_to_matrix(
                    origin_rpy_batch[:, 0],
                    origin_rpy_batch[:, 1],
                    origin_rpy_batch[:, 2]
                )  # [B, 3, 3]
                
                # 3. Joint rotation (revolute around axis)
                axis = torch.tensor(js.axis, device=device, dtype=dtype)  # [3]
                R_joint = self._axis_angle_to_rotation_matrix_batch(axis, theta, device, dtype)  # [B, 3, 3]
                
                # Combine rotations: R_total = R_origin @ R_joint
                R_total = torch.matmul(R_origin, R_joint)  # [B, 3, 3]
                
                # Build 4x4 transformation matrix for each sample
                T_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                T_joint[:, :3, :3] = R_total
                T_joint[:, :3, 3] = origin_xyz  # Same translation for all samples
                
                # Accumulate transformation: T_batch = T_batch @ T_joint (batch matmul)
                T_batch = torch.matmul(T_batch, T_joint)
        finally:
            # Restore original backend
            set_backend(original_backend)
        
        return T_batch
    
    @staticmethod
    def _axis_angle_to_rotation_matrix_batch(
        axis: torch.Tensor,
        angles: torch.Tensor,
        device,
        dtype
    ) -> torch.Tensor:
        """Convert batch of axis-angle rotations to rotation matrices.
        
        :param axis: rotation axis [3]
        :param angles: rotation angles [B]
        :return: rotation matrices [B, 3, 3]
        """
        batch_size = angles.shape[0]
        
        # Normalize axis
        axis = axis / (torch.norm(axis) + 1e-10)
        
        # Rodrigues' formula: R = I + sin(θ) * K + (1-cos(θ)) * K²
        # where K is the skew-symmetric matrix of axis
        kx, ky, kz = axis[0], axis[1], axis[2]
        
        # Skew-symmetric matrix K (same for all samples)
        K = torch.tensor([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ], device=device, dtype=dtype)
        
        # K²
        K2 = torch.matmul(K, K)
        
        # Compute for batch
        cos_theta = torch.cos(angles)  # [B]
        sin_theta = torch.sin(angles)  # [B]
        
        # Broadcast: I + sin(θ)*K + (1-cos(θ))*K²
        I = torch.eye(3, device=device, dtype=dtype)
        R_batch = I.unsqueeze(0) + sin_theta.view(batch_size, 1, 1) * K.unsqueeze(0) + \
                  (1 - cos_theta).view(batch_size, 1, 1) * K2.unsqueeze(0)
        
        return R_batch


__all__ = ["FKSolverTorch"]
