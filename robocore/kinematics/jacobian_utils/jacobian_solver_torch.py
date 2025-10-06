"""PyTorch Jacobian solver.

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

from typing import TYPE_CHECKING, Literal, Optional
import math

try:
    import torch
except ImportError as e:
    raise ImportError("jacobian_solver_torch 需要 PyTorch, 请先: pip install torch") from e

try:
    from ...utils.torch_utils import select_device
except Exception:
    def select_device(device=None):
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

from robocore.utils.backend import set_backend, get_backend
from robocore.transform import rotation_error

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

Tensor = torch.Tensor

__all__ = ["JacobianSolverTorch"]


class JacobianSolverTorch:
    """PyTorch-accelerated Jacobian solver.
    
    Features:
    - Analytic (geometric) Jacobian computation
    - Numeric (finite-difference) Jacobian computation  
    - Autograd-based Jacobian computation
    - GPU acceleration support
    - Automatic differentiation compatible
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize Jacobian solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.num_dof()
    
    def solve(
        self,
        q: Tensor | list,
        method: Literal["analytic", "numeric", "autograd"] = "analytic",
        epsilon: float = 5e-5,
        use_central_diff: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        target_link: str | None = None,
    ) -> Tensor:
        """Compute Jacobian matrix (supports both single and batch modes).
        
        :param q: joint configuration(s).
            - Single mode: (n,) → returns [6, n]
            - Batch mode: (B, n) → returns [B, 6, n]
        :param method: 'analytic', 'numeric', or 'autograd'.
        :param epsilon: finite difference step size (numeric only).
        :param use_central_diff: use central difference if True (numeric only).
        :param device: torch device.
        :param dtype: torch dtype.
        :return: Jacobian matrix as torch tensor.
        """
        device = select_device(device)
        
        # Default dtype: 统一使用 float64（可被用户覆盖）
        if dtype is None:
            dtype = torch.float64
        
        # Convert to tensor
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        # Detect batch mode
        is_batch = q.ndim == 2
        
        if is_batch:
            # Batch mode: q is [B, n]
            if method == "analytic":
                if target_link is not None:
                    # For batch + partial target, fall back to per-sample analytic computation
                    J_list = []
                    for qb in q:
                        J_list.append(self._solve_analytic(qb, device, dtype, target_link=target_link))
                    return torch.stack(J_list, dim=0)
                return self._solve_analytic_batch(q, device, dtype)
            else:
                raise NotImplementedError(f"Batch mode only supports 'analytic' method, got '{method}'")
        else:
            # Single mode: q is (n,)
            if method == "analytic":
                return self._solve_analytic(q, device, dtype, target_link=target_link)
            elif method == "numeric":
                return self._solve_numeric(q, epsilon, use_central_diff, device, dtype)
            elif method == "autograd":
                return self._solve_autograd(q, device, dtype)
            else:
                raise ValueError(f"Unknown method '{method}', expected 'analytic', 'numeric', or 'autograd'")
    
    def _solve_analytic(self, q, device, dtype, target_link: str | None = None) -> Tensor:
        """Compute analytic (geometric) Jacobian.

        If target_link is provided, the forward traversal is terminated early
        once the joint whose child link equals target_link is reached. This
        yields a local Jacobian for that intermediate link (with respect to the
        world frame, angular rows expressed in the local link frame consistent
        with the end-effector convention). J columns for joints appearing after
        the truncated link in the kinematic chain (if any) will be zero because
        they do not influence that link pose.
        """
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        if q.shape[0] != self.n:
            raise ValueError(f"Configuration length {q.shape[0]} != dof {self.n}")
        
        # Build forward transforms
        T_parent = torch.eye(4, dtype=dtype, device=device)
        q_map = {js.name: q[js.index] for js in self.model._actuated}
        
        p_list = [None] * self.n
        z_list = [None] * self.n
        
        end_T = T_parent
        
        reached_target = False
        for urdf_joint in self.model._chain_joints:
            R_origin = self._rpy_matrix_torch(
                torch.tensor(urdf_joint.origin_rpy[0], dtype=dtype, device=device),
                torch.tensor(urdf_joint.origin_rpy[1], dtype=dtype, device=device),
                torch.tensor(urdf_joint.origin_rpy[2], dtype=dtype, device=device),
            )
            t_origin = torch.tensor(urdf_joint.origin_xyz, dtype=dtype, device=device)
            
            T_origin = torch.eye(4, dtype=dtype, device=device)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = t_origin
            T_joint_origin = T_parent @ T_origin
            
            if urdf_joint.joint_type in ("revolute", "prismatic"):
                js = next(js for js in self.model._actuated if js.name == urdf_joint.name)
                axis_local = torch.tensor(urdf_joint.axis, dtype=dtype, device=device)
                axis_norm = torch.linalg.norm(axis_local)
                if axis_norm > 1e-10:
                    axis_local = axis_local / axis_norm
                z_i = T_joint_origin[:3, :3] @ axis_local
                p_i = T_joint_origin[:3, 3].clone()
                p_list[js.index] = p_i
                z_list[js.index] = z_i
            
            R_motion = torch.eye(3, dtype=dtype, device=device)
            t_motion = torch.zeros(3, dtype=dtype, device=device)
            
            if urdf_joint.joint_type == "revolute":
                theta = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=device))
                R_motion = self._axis_rotation_torch(
                    torch.tensor(urdf_joint.axis, dtype=dtype, device=device),
                    theta
                )
            elif urdf_joint.joint_type == "prismatic":
                d = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=device))
                t_motion = self._axis_translation_torch(
                    torch.tensor(urdf_joint.axis, dtype=dtype, device=device),
                    d
                )
            
            T_motion = torch.eye(4, dtype=dtype, device=device)
            T_motion[:3, :3] = R_motion
            T_motion[:3, 3] = t_motion
            
            T_child = T_joint_origin @ T_motion
            T_parent = T_child
            end_T = T_child
            if target_link is not None and urdf_joint.child == target_link:
                reached_target = True
                break
        
        p_end = end_T[:3, 3]
        
        # Assemble Jacobian
        J_geo = torch.zeros((6, self.n), dtype=dtype, device=device)
        for i in range(self.n):
            z_i = z_list[i]
            p_i = p_list[i]
            if z_i is None or p_i is None:
                # If target_link caused early stop, remaining joints have no influence → leave zero column
                if target_link is not None and reached_target:
                    continue
                raise RuntimeError("Internal error: missing joint axis or origin position")

            js = self.model._actuated[i]
            if js.joint_type == "revolute":
                J_geo[:3, i] = torch.linalg.cross(z_i, (p_end - p_i))
                J_geo[3:6, i] = z_i
            elif js.joint_type == "prismatic":
                J_geo[:3, i] = z_i
        
        # Transform angular part to end-effector frame
        R_end = end_T[:3, :3]
        J = J_geo.clone()
        J[3:6, :] = R_end.T @ J_geo[3:6, :]
        return J
    
    def _solve_numeric(self, q, epsilon, use_central_diff, device, dtype) -> Tensor:
        """Compute numeric Jacobian using finite differences."""
        # Import FK solver here to avoid circular dependency
        from ..fk_utils.fk_solver_torch import FKSolverTorch
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        J = torch.zeros((6, self.n), dtype=dtype, device=device)
        fk_solver = FKSolverTorch(self.model)
        
        if use_central_diff:
            T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
            R_ref = T_ref[:3, :3].clone()
            
            for i in range(self.n):
                qp = q.clone()
                qp[i] += epsilon
                T_pos = fk_solver.solve(qp, return_end_only=True, device=device, dtype=dtype)["end"]
                p_pos = T_pos[:3, 3]
                R_pos = T_pos[:3, :3]
                
                qn = q.clone()
                qn[i] -= epsilon
                T_neg = fk_solver.solve(qn, return_end_only=True, device=device, dtype=dtype)["end"]
                p_neg = T_neg[:3, 3]
                R_neg = T_neg[:3, :3]
                
                J[:3, i] = (p_pos - p_neg) / (2 * epsilon)
                
                err_pos = rotation_error(R_ref, R_pos)
                err_neg = rotation_error(R_ref, R_neg)
                J[3:6, i] = (err_pos - err_neg) / (2 * epsilon)
        else:
            T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
            R_ref = T_ref[:3, :3]
            p_ref = T_ref[:3, 3]
            
            for i in range(self.n):
                qp = q.clone()
                qp[i] += epsilon
                T_pos = fk_solver.solve(qp, return_end_only=True, device=device, dtype=dtype)["end"]
                p_pos = T_pos[:3, 3]
                R_pos = T_pos[:3, :3]
                
                J[:3, i] = (p_pos - p_ref) / epsilon
                err = rotation_error(R_ref, R_pos)
                J[3:6, i] = err / epsilon
        
        return J
    
    def _solve_autograd(self, q, device, dtype) -> Tensor:
        """Compute Jacobian using PyTorch autograd."""
        from ..fk_utils.fk_solver_torch import FKSolverTorch
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device, requires_grad=True)
        else:
            q = q.to(dtype=dtype, device=device)
            q.requires_grad_(True)
        
        fk_solver = FKSolverTorch(self.model)
        
        def pose_vec(q_):
            T = fk_solver.solve(q_, return_end_only=True, device=device, dtype=dtype)["end"]
            p = T[:3, 3]
            R = T[:3, :3].reshape(-1)
            return torch.cat([p, R])  # (12,)
        
        J_big = torch.autograd.functional.jacobian(
            pose_vec, q, create_graph=False, vectorize=False
        )  # (12,n)
        pJ = J_big[:3, :]  # (3,n)
        R_flat_J = J_big[3:, :]  # (9,n)
        
        # Current pose
        T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
        R = T_ref[:3, :3]
        
        J = torch.zeros((6, self.n), dtype=dtype, device=device)
        J[:3, :] = pJ
        R_T = R.transpose(0, 1)
        
        for j in range(self.n):
            dR_flat = R_flat_J[:, j]
            dR = dR_flat.view(3, 3)
            skew = dR @ R_T
            wx = (skew[2, 1] - skew[1, 2]) * 0.5
            wy = (skew[0, 2] - skew[2, 0]) * 0.5
            wz = (skew[1, 0] - skew[0, 1]) * 0.5
            w_world = torch.stack([wx, wy, wz])
            w_end = R_T @ w_world
            J[3:6, j] = w_end
        
        return J.detach()
    
    def _solve_analytic_batch(self, q_batch: Tensor, device, dtype) -> Tensor:
        """Compute batch geometric Jacobian for multiple configurations in parallel.
        
        :param q_batch: batch of joint configurations [B, n]
        :param device: torch device
        :param dtype: torch dtype
        :return: batch of Jacobian matrices [B, 6, n]
        """
        from ..fk_utils.fk_solver_torch import FKSolverTorch
        
        batch_size = q_batch.shape[0]
        n_joints = q_batch.shape[1]
        
        if n_joints != self.n:
            raise ValueError(f"Expected {self.n} joints, got {n_joints}")
        
        # Initialize Jacobian [B, 6, N]
        J_batch = torch.zeros(batch_size, 6, n_joints, device=device, dtype=dtype)
        
        # Get end-effector position for all samples [B, 4, 4]
        fk_solver = FKSolverTorch(self.model)
        T_ee_batch = fk_solver.solve(q_batch, device=device, dtype=dtype)  # auto-detects batch mode
        p_ee_batch = T_ee_batch[:, :3, 3]  # [B, 3]
        
        # Process each joint
        for js in self.model._actuated:
            joint_idx = js.index
            
            # Compute FK up to this joint for all samples
            T_joint_batch = self._forward_kinematics_to_joint_batch(
                q_batch, joint_idx, device, dtype
            )
            
            # Extract position and z-axis
            p_joint_batch = T_joint_batch[:, :3, 3]  # [B, 3]
            z_axis_batch = T_joint_batch[:, :3, 2]  # [B, 3] - third column of rotation matrix
            
            # For revolute joint:
            # J_v[i] = z[i] × (p_ee - p[i])  (linear velocity contribution)
            # J_ω[i] = z[i]                  (angular velocity contribution)
            
            if js.joint_type == "revolute":
                # Linear part: cross product z × (p_ee - p_joint)
                r = p_ee_batch - p_joint_batch  # [B, 3]
                J_linear = torch.cross(z_axis_batch, r, dim=1)  # [B, 3]
                
                # Angular part: just the z-axis
                J_angular = z_axis_batch  # [B, 3]
                
                # Assemble into Jacobian
                J_batch[:, :3, joint_idx] = J_linear
                J_batch[:, 3:6, joint_idx] = J_angular
            elif js.joint_type == "prismatic":
                # Prismatic: only linear motion along axis
                J_batch[:, :3, joint_idx] = z_axis_batch
        
        return J_batch
    
    def _forward_kinematics_to_joint_batch(
        self, 
        q_batch: Tensor, 
        joint_idx: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """Compute FK up to specific joint for batch of configurations.
        
        :param q_batch: batch of joint configs [B, n]
        :param joint_idx: index of target joint (0-indexed)
        :param device: torch device
        :param dtype: torch dtype
        :return: transformation matrices up to joint [B, 4, 4]
        """
        batch_size = q_batch.shape[0]
        
        # Initialize batch of identity matrices [B, 4, 4]
        T_batch = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Get joints up to and including target
        joint_specs = self.model._actuated[:joint_idx + 1]
        
        # Process each joint in the chain
        for js in joint_specs:
            theta = q_batch[:, js.index]  # [B]
            
            # Find corresponding URDF joint
            urdf_joint = next(j for j in self.model._chain_joints if j.name == js.name)
            
            # Build transformation matrix for this joint
            origin_xyz = torch.tensor(urdf_joint.origin_xyz, device=device, dtype=dtype)
            origin_rpy = torch.tensor(urdf_joint.origin_rpy, device=device, dtype=dtype)
            axis = torch.tensor(urdf_joint.axis, device=device, dtype=dtype)
            
            # Rotation matrices
            R_origin = self._rpy_to_rotation_matrix_batch(
                origin_rpy.unsqueeze(0).repeat(batch_size, 1),
                device, dtype
            )  # [B, 3, 3]
            R_joint = self._axis_angle_to_rotation_matrix_batch(axis, theta, device, dtype)  # [B, 3, 3]
            R_total = torch.matmul(R_origin, R_joint)  # [B, 3, 3]
            
            # Build 4x4 transformation [B, 4, 4]
            T_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            T_joint[:, :3, :3] = R_total
            T_joint[:, :3, 3] = origin_xyz  # broadcast to all batches
            
            # Accumulate transformation
            T_batch = torch.matmul(T_batch, T_joint)
        
        return T_batch
    
    @staticmethod
    def _rpy_to_rotation_matrix_batch(
        rpy_batch: Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """Convert batch of RPY to rotation matrices.
        
        :param rpy_batch: [B, 3] roll-pitch-yaw angles
        :return: [B, 3, 3] rotation matrices
        """
        batch_size = rpy_batch.shape[0]
        r = rpy_batch[:, 0]  # [B]
        p = rpy_batch[:, 1]  # [B]
        y = rpy_batch[:, 2]  # [B]
        
        sr, cr = torch.sin(r), torch.cos(r)
        sp, cp = torch.sin(p), torch.cos(p)
        sy, cy = torch.sin(y), torch.cos(y)
        
        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        
        return R
    
    @staticmethod
    def _axis_angle_to_rotation_matrix_batch(
        axis: Tensor,
        theta_batch: Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """Convert batch of axis-angle to rotation matrices (Rodrigues formula).
        
        :param axis: rotation axis [3] (shared across batch)
        :param theta_batch: rotation angles [B]
        :return: [B, 3, 3] rotation matrices
        """
        batch_size = theta_batch.shape[0]
        
        # Normalize axis
        norm = torch.linalg.norm(axis)
        if norm < 1e-12:
            return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        
        k = axis / norm  # [3]
        kx, ky, kz = k[0], k[1], k[2]
        
        # Precompute trigonometric values
        cos_theta = torch.cos(theta_batch)  # [B]
        sin_theta = torch.sin(theta_batch)  # [B]
        one_minus_cos = 1.0 - cos_theta  # [B]
        
        # Build rotation matrices using Rodrigues formula
        # R = I + sin(θ)K + (1-cos(θ))K²
        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        
        # Diagonal: cos(θ) + (1-cos(θ))k_i²
        R[:, 0, 0] = cos_theta + one_minus_cos * kx * kx
        R[:, 1, 1] = cos_theta + one_minus_cos * ky * ky
        R[:, 2, 2] = cos_theta + one_minus_cos * kz * kz
        
        # Off-diagonal terms
        R[:, 0, 1] = one_minus_cos * kx * ky - sin_theta * kz
        R[:, 0, 2] = one_minus_cos * kx * kz + sin_theta * ky
        R[:, 1, 0] = one_minus_cos * ky * kx + sin_theta * kz
        R[:, 1, 2] = one_minus_cos * ky * kz - sin_theta * kx
        R[:, 2, 0] = one_minus_cos * kz * kx - sin_theta * ky
        R[:, 2, 1] = one_minus_cos * kz * ky + sin_theta * kx
        
        return R

    
    # ============================================================================
    # Helper functions
    # ============================================================================
    
    @staticmethod
    def _rpy_matrix_torch(r, p, y):
        """Compute rotation matrix from roll-pitch-yaw."""
        sr, cr = torch.sin(r), torch.cos(r)
        sp, cp = torch.sin(p), torch.cos(p)
        sy, cy = torch.sin(y), torch.cos(y)
        return torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ])
    
    @staticmethod
    def _axis_rotation_torch(axis, theta):
        """Compute rotation matrix for rotation about axis by angle."""
        norm = torch.linalg.norm(axis)
        if norm < 1e-12:
            return torch.eye(3, dtype=axis.dtype, device=axis.device)
        a = axis / norm
        ax, ay, az = a
        ct, st = torch.cos(theta), torch.sin(theta)
        vt = 1 - ct
        return torch.stack([
            torch.stack([ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st]),
            torch.stack([ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st]),
            torch.stack([az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt]),
        ])
    
    @staticmethod
    def _axis_translation_torch(axis, d):
        """Compute translation along axis."""
        norm = torch.linalg.norm(axis)
        if norm < 1e-12:
            return torch.zeros(3, dtype=axis.dtype, device=axis.device)
        return axis / norm * d
