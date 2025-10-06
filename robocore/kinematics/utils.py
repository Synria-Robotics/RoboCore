"""Kinematics utility functions.

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
import numpy as np
from robocore.transform.conversions import matrix_to_axis_angle


def relative_pose_error(T_left: np.ndarray, T_right: np.ndarray, T_rel_desired: np.ndarray) -> np.ndarray:
    """Compute 6D error for relative pose constraint.
    
    Constraint: T_rel = T_left^{-1} @ T_right should equal T_rel_desired
    Error is computed in the left arm's coordinate frame.
    
    :param T_left: Left arm end-effector pose (4x4)
    :param T_right: Right arm end-effector pose (4x4)
    :param T_rel_desired: Desired relative transformation (4x4)
    :return: 6D error [e_pos (3), e_ori (3)] in left arm frame
    """
    # Current relative transform
    T_left_inv = np.linalg.inv(T_left)
    T_rel_current = T_left_inv @ T_right
    
    # Position error (in left arm frame)
    p_rel_current = T_rel_current[0:3, 3]
    p_rel_desired = T_rel_desired[0:3, 3]
    e_pos = p_rel_desired - p_rel_current
    
    # Orientation error (axis-angle in left arm frame)
    R_rel_current = T_rel_current[0:3, 0:3]
    R_rel_desired = T_rel_desired[0:3, 0:3]
    R_error = R_rel_desired @ R_rel_current.T
    
    # Convert rotation matrix to axis-angle
    axis, angle = matrix_to_axis_angle(R_error)
    e_ori = axis * angle
    
    return np.concatenate([e_pos, e_ori])


def relative_jacobian(left_model, right_model, q_left, q_right, backend: str = 'numpy') -> np.ndarray:
    """Construct Jacobian for relative pose task (numerical differentiation).
    
    Maps joint velocities to relative pose velocity:
        ė_rel = J_rel @ q̇   where q̇ = [q̇_left; q̇_right]
    
    For stability and correctness, this uses numerical differentiation
    rather than analytical adjoint formulation.
    
    :param left_model: Left arm robot model
    :param right_model: Right arm robot model
    :param q_left: Left arm joint configuration
    :param q_right: Right arm joint configuration
    :param backend: Backend for computation ('numpy' or 'auto')
    :return: 6 x (nL + nR) relative Jacobian matrix
    """
    eps = 1e-7
    
    def rel_pose_6d(qL, qR):
        """Compute relative pose as 6D vector (position + axis-angle)."""
        TL = left_model.fk(qL, backend=backend, return_end=True)
        TR = right_model.fk(qR, backend=backend, return_end=True)
        
        # Ensure numpy
        if hasattr(TL, 'detach'):
            TL = TL.detach().cpu().numpy()
            TR = TR.detach().cpu().numpy()
        else:
            TL = np.array(TL)
            TR = np.array(TR)
        
        T_rel = np.linalg.inv(TL) @ TR
        
        p_rel = T_rel[0:3, 3]
        R_rel = T_rel[0:3, 0:3]
        
        axis, angle = matrix_to_axis_angle(R_rel)
        ori_rel = axis * angle
        
        return np.concatenate([p_rel, ori_rel])
    
    # Base configuration
    q_left_np = np.array(q_left, dtype=float)
    q_right_np = np.array(q_right, dtype=float)
    q_combined = np.concatenate([q_left_np, q_right_np])
    
    nL = len(q_left_np)
    nR = len(q_right_np)
    
    # Numerical Jacobian
    J_rel = np.zeros((6, nL + nR))
    p0 = rel_pose_6d(q_left_np, q_right_np)
    
    for i in range(nL + nR):
        q_pert = q_combined.copy()
        q_pert[i] += eps
        
        qL_p = q_pert[:nL]
        qR_p = q_pert[nL:]
        
        p_plus = rel_pose_6d(qL_p, qR_p)
        J_rel[:, i] = (p_plus - p0) / eps
    
    return J_rel


__all__ = ["relative_pose_error", "relative_jacobian"]
