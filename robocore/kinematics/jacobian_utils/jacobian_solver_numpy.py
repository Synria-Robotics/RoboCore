"""NumPy Jacobian solver.

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

from typing import TYPE_CHECKING, Literal
import numpy as np
import math
from robocore.transform import rotation_error
from robocore.utils.backend import set_backend, get_backend
from robocore.kinematics.fk import forward_kinematics

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

__all__ = ["JacobianSolverNumPy"]


class JacobianSolverNumPy:
    """NumPy-accelerated Jacobian solver.
    
    Features:
    - Analytic (geometric) Jacobian computation
    - Numeric (finite-difference) Jacobian computation
    - Central or forward difference options
    - Consistent with IK error representation
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize Jacobian solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.num_dof()
    
    def solve(
        self,
        q: np.ndarray,
        method: Literal["analytic", "numeric"] = "analytic",
        epsilon: float = 5e-5,
        use_central_diff: bool = True,
        target_link: str | None = None,
    ) -> np.ndarray:
        """Compute 6×n Jacobian matrix.
        
        :param q: joint configuration (n,).
        :param method: 'analytic' for geometric or 'numeric' for finite-difference.
        :param epsilon: finite difference step size (numeric only).
        :param use_central_diff: use central difference if True (numeric only).
        :return: 6×n Jacobian matrix (top 3 rows: linear, bottom 3 rows: angular).
        """
        # Ensure NumPy backend
        prev_backend = get_backend()
        set_backend('numpy')

        try:
            if method == "analytic":
                return self._solve_analytic(q, target_link=target_link)
            elif method == "numeric":
                if target_link is not None:
                    # For now numeric local Jacobian uses full end Jacobian then slice rows belonging to target link
                    # Simpler: recompute by truncating chain to target_link
                    return self._solve_numeric(q, epsilon, use_central_diff, target_link=target_link)
                return self._solve_numeric(q, epsilon, use_central_diff, target_link=None)
            else:
                raise ValueError(f"Unknown method '{method}', expected 'analytic' or 'numeric'")
        finally:
            set_backend(prev_backend)
    
    def _solve_analytic(self, q: np.ndarray, target_link: str | None = None) -> np.ndarray:
        """Compute analytic (geometric) Jacobian.
        
        For each actuated joint i (in world frame):
          Revolute:
            Jv_i = z_i × (p_end - p_i)
            Jw_i = z_i
          Prismatic:
            Jv_i = z_i
            Jw_i = 0
        
        The angular part is then transformed to end-effector frame.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] != self.n:
            raise ValueError(f"Configuration length {q.shape[0]} != dof {self.n}")
        
        # Build forward transforms to get joint origins and axes in world frame
        T_parent = np.eye(4, dtype=np.float64)
        q_map = {js.name: q[js.index] for js in self.model._actuated}
        
        # Storage for each actuated joint
        p_list = [None] * self.n
        z_list = [None] * self.n
        
        end_T = T_parent
        
        # Iterate over chain joints
        for urdf_joint in self.model._chain_joints:
            # Parent transform
            R_origin = self._rpy_matrix(*urdf_joint.origin_rpy)
            t_origin = np.array(urdf_joint.origin_xyz, dtype=np.float64)
            
            T_origin = np.eye(4, dtype=np.float64)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = t_origin
            T_joint_origin = T_parent @ T_origin
            
            # If actuated, record axis & origin position BEFORE motion transform
            if urdf_joint.joint_type in ("revolute", "prismatic"):
                js = next(js for js in self.model._actuated if js.name == urdf_joint.name)
                axis_local = np.asarray(urdf_joint.axis, dtype=np.float64)
                axis_norm = np.linalg.norm(axis_local)
                if axis_norm > 1e-10:
                    axis_local = axis_local / axis_norm
                z_i = T_joint_origin[:3, :3] @ axis_local
                p_i = T_joint_origin[:3, 3].copy()
                p_list[js.index] = p_i
                z_list[js.index] = z_i
            
            # Apply motion of this joint
            R_motion = np.eye(3, dtype=np.float64)
            t_motion = np.zeros(3, dtype=np.float64)
            
            if urdf_joint.joint_type == "revolute":
                theta = q_map.get(urdf_joint.name, 0.0)
                R_motion = self._axis_rotation(urdf_joint.axis, theta)
            elif urdf_joint.joint_type == "prismatic":
                d = q_map.get(urdf_joint.name, 0.0)
                t_motion = self._axis_translation(urdf_joint.axis, d)
            
            T_motion = np.eye(4, dtype=np.float64)
            T_motion[:3, :3] = R_motion
            T_motion[:3, 3] = t_motion
            
            T_child = T_joint_origin @ T_motion
            T_parent = T_child
            end_T = T_child
            if target_link is not None and urdf_joint.child == target_link:
                # Stop traversal early at target_link (treat as pseudo end-effector)
                break

        p_end = end_T[:3, 3]

        # Assemble Jacobian (geometric world-frame)
        J_geo = np.zeros((6, self.n), dtype=np.float64)
        for i in range(self.n):
            z_i = z_list[i]
            p_i = p_list[i]
            if z_i is None or p_i is None:
                # If we early-stopped at target_link, joints beyond it produce zero columns
                if target_link is not None:
                    continue
                raise RuntimeError("Internal error: missing joint axis or origin position")
            js = self.model._actuated[i]
            if js.joint_type == "revolute":
                J_geo[:3, i] = np.cross(z_i, (p_end - p_i))
                J_geo[3:6, i] = z_i
            elif js.joint_type == "prismatic":
                J_geo[:3, i] = z_i

        # Transform angular part to end-effector frame
        R_end = end_T[:3, :3]
        J = J_geo.copy()
        J[3:6, :] = R_end.T @ J_geo[3:6, :]
        return J
    
    def _solve_numeric(
        self, 
        q: np.ndarray, 
        epsilon: float,
        use_central_diff: bool,
        target_link: str | None = None,
    ) -> np.ndarray:
        """Compute numeric Jacobian using finite differences."""
        q = np.asarray(q, dtype=np.float64)
        J = np.zeros((6, self.n), dtype=np.float64)
        
        if use_central_diff:
            # Central difference - use standalone FK to avoid circular dependency
            if target_link is None:
                fk_ref = forward_kinematics(self.model, q.tolist(), backend='numpy', return_end=True)
            else:
                fk_ref = self._fk_until(q, target_link)
            R_ref = np.array(
                fk_ref[:3, :3] if isinstance(fk_ref, np.ndarray) 
                else [row[:3] for row in fk_ref[:3]], 
                dtype=np.float64
            )
            
            for i in range(self.n):
                # Positive perturbation
                q_pos = q.copy()
                q_pos[i] += epsilon
                if target_link is None:
                    fk_pos = forward_kinematics(self.model, q_pos.tolist(), backend='numpy', return_end=True)
                else:
                    fk_pos = self._fk_until(q_pos, target_link)
                R_pos = np.array(
                    fk_pos[:3, :3] if isinstance(fk_pos, np.ndarray) 
                    else [row[:3] for row in fk_pos[:3]], 
                    dtype=np.float64
                )
                p_pos = np.array(
                    fk_pos[:3, 3] if isinstance(fk_pos, np.ndarray) 
                    else [fk_pos[0][3], fk_pos[1][3], fk_pos[2][3]], 
                    dtype=np.float64
                )
                
                # Negative perturbation
                q_neg = q.copy()
                q_neg[i] -= epsilon
                if target_link is None:
                    fk_neg = forward_kinematics(self.model, q_neg.tolist(), backend='numpy', return_end=True)
                else:
                    fk_neg = self._fk_until(q_neg, target_link)
                R_neg = np.array(
                    fk_neg[:3, :3] if isinstance(fk_neg, np.ndarray) 
                    else [row[:3] for row in fk_neg[:3]], 
                    dtype=np.float64
                )
                p_neg = np.array(
                    fk_neg[:3, 3] if isinstance(fk_neg, np.ndarray) 
                    else [fk_neg[0][3], fk_neg[1][3], fk_neg[2][3]], 
                    dtype=np.float64
                )
                
                # Position derivative
                J[:3, i] = (p_pos - p_neg) / (2 * epsilon)
                
                # Orientation derivative
                err_pos = rotation_error(R_ref, R_pos)
                err_neg = rotation_error(R_ref, R_neg)
                J[3:6, i] = (err_pos - err_neg) / (2 * epsilon)
        else:
            # Forward difference - use standalone FK to avoid circular dependency
            if target_link is None:
                fk_ref = forward_kinematics(self.model, q.tolist(), backend='numpy', return_end=True)
            else:
                fk_ref = self._fk_until(q, target_link)
            R_ref = np.array(
                fk_ref[:3, :3] if isinstance(fk_ref, np.ndarray) 
                else [row[:3] for row in fk_ref[:3]], 
                dtype=np.float64
            )
            p_ref = np.array(
                fk_ref[:3, 3] if isinstance(fk_ref, np.ndarray) 
                else [fk_ref[0][3], fk_ref[1][3], fk_ref[2][3]], 
                dtype=np.float64
            )
            
            for i in range(self.n):
                q_pert = q.copy()
                q_pert[i] += epsilon
                if target_link is None:
                    fk_pert = forward_kinematics(self.model, q_pert.tolist(), backend='numpy', return_end=True)
                else:
                    fk_pert = self._fk_until(q_pert, target_link)
                R_pert = np.array(
                    fk_pert[:3, :3] if isinstance(fk_pert, np.ndarray) 
                    else [row[:3] for row in fk_pert[:3]], 
                    dtype=np.float64
                )
                p_pert = np.array(
                    fk_pert[:3, 3] if isinstance(fk_pert, np.ndarray) 
                    else [fk_pert[0][3], fk_pert[1][3], fk_pert[2][3]], 
                    dtype=np.float64
                )
                
                J[:3, i] = (p_pert - p_ref) / epsilon
                err = rotation_error(R_ref, R_pert)
                J[3:6, i] = err / epsilon
        
        return J

    # ------------------------------------------------------------------
    # Local helper: partial FK to specified target_link (end-only)
    # ------------------------------------------------------------------
    def _fk_until(self, q: np.ndarray, target_link: str) -> np.ndarray:
        """Compute FK up to (and including) target_link, returning its 4x4 pose.

        This avoids computing full chain when only an intermediate link Jacobian is needed.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] != self.n:
            raise ValueError(f"Configuration length {q.shape[0]} != dof {self.n}")
        q_map = {js.name: q[js.index] for js in self.model._actuated}
        T_parent = np.eye(4, dtype=np.float64)
        for urdf_joint in self.model._chain_joints:
            R_origin = self._rpy_matrix(*urdf_joint.origin_rpy)
            t_origin = np.array(urdf_joint.origin_xyz, dtype=np.float64)
            T_origin = np.eye(4, dtype=np.float64)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = t_origin
            T_joint_origin = T_parent @ T_origin
            R_motion = np.eye(3, dtype=np.float64)
            t_motion = np.zeros(3, dtype=np.float64)
            if urdf_joint.joint_type == "revolute":
                theta = q_map.get(urdf_joint.name, 0.0)
                R_motion = self._axis_rotation(urdf_joint.axis, theta)
            elif urdf_joint.joint_type == "prismatic":
                d = q_map.get(urdf_joint.name, 0.0)
                t_motion = self._axis_translation(urdf_joint.axis, d)
            T_motion = np.eye(4, dtype=np.float64)
            T_motion[:3, :3] = R_motion
            T_motion[:3, 3] = t_motion
            T_child = T_joint_origin @ T_motion
            T_parent = T_child
            if urdf_joint.child == target_link:
                return T_child
        raise ValueError(f"target_link '{target_link}' not found in kinematic chain")
    
    # ============================================================================
    # Helper functions
    # ============================================================================
    
    @staticmethod
    def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute rotation matrix from roll-pitch-yaw."""
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)
    
    @staticmethod
    def _axis_rotation(axis, theta: float) -> np.ndarray:
        """Compute rotation matrix for rotation about axis by angle."""
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / norm, ay / norm, az / norm
        ct = math.cos(theta)
        st = math.sin(theta)
        vt = 1.0 - ct
        return np.array([
            [ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st],
            [ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st],
            [az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt],
        ], dtype=np.float64)
    
    @staticmethod
    def _axis_translation(axis, d: float) -> np.ndarray:
        """Compute translation along axis."""
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        return np.array([ax * d / norm, ay * d / norm, az * d / norm], dtype=np.float64)
