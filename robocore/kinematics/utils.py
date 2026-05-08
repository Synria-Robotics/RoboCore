"""Kinematics utility functions.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations
import numpy as np
from robocore.transform.conversions import matrix_to_axis_angle


def _get_torch_module():
    try:
        from importlib import import_module

        return import_module('torch')
    except ImportError:
        return None


def _is_torch_tensor(value) -> bool:
    torch_module = _get_torch_module()
    return torch_module is not None and isinstance(value, torch_module.Tensor)


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


def relative_jacobian(left_model, right_model, q_left, q_right) -> np.ndarray:
    """Construct Jacobian for relative pose task (numerical differentiation).
    
    Maps joint velocities to relative pose velocity:
        ė_rel = J_rel @ q̇   where q̇ = [q̇_left; q̇_right]
    
    For stability and correctness, this uses numerical differentiation
    rather than analytical adjoint formulation.
    
    :param left_model: Left arm robot model
    :param right_model: Right arm robot model
    :param q_left: Left arm joint configuration
    :param q_right: Right arm joint configuration
    :return: 6 x (nL + nR) relative Jacobian matrix
    """
    eps = 1e-7
    
    def rel_pose_6d(qL, qR):
        """Compute relative pose as 6D vector (position + axis-angle)."""
        TL = left_model.fk(qL, return_end=True)
        TR = right_model.fk(qR, return_end=True)
        
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


def ensure_batch(q):
    """Convert 1D to 2D batch=1, return (q_batch, was_single).
    
    :param q: joint configuration(s), numpy array or torch tensor
    :return: tuple of (q_batch, was_single) where was_single indicates if input was 1D
    """
    was_single = q.ndim == 1
    if was_single:
        if isinstance(q, np.ndarray):
            q = q[np.newaxis, :]
        elif _is_torch_tensor(q):
            q = q.unsqueeze(0)
        else:
            # Fallback for other array-like types
            q = np.asarray(q)[np.newaxis, :]
    return q, was_single


def restore_single(result, was_single):
    """Restore single format if input was single.
    
    :param result: result from batch computation
    :param was_single: whether original input was single (1D)
    :return: result in original format (single if was_single, batch otherwise)
    """
    if was_single:
        if isinstance(result, (list, tuple)):
            return result[0] if len(result) > 0 else result
        elif hasattr(result, 'ndim') and result.ndim > 2:
            return result[0]
        elif isinstance(result, dict) and 'q' in result:
            # IK result dict - keep as dict but extract single from batch
            result_single = {}
            for k, v in result.items():
                if isinstance(v, list) and len(v) > 0:
                    # NumPy IK batch result: dict with lists
                    result_single[k] = v[0]
                elif hasattr(v, 'ndim') and v.ndim > 0:
                    if isinstance(v, np.ndarray) and v.ndim > 1:
                        result_single[k] = v[0]
                    elif _is_torch_tensor(v) and v.ndim > 1:
                        result_single[k] = v[0]
                    else:
                        result_single[k] = v
                else:
                    result_single[k] = v
            return result_single
    return result


__all__ = ["relative_pose_error", "relative_jacobian", "ensure_batch", "restore_single"]
