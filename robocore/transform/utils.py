"""Advanced transformation utilities.

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

from robocore.utils.backend import get_backend_manager
from .conversions import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
)
from .so3 import rotation_multiply, rotation_inverse


def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    :param q1: Start quaternion [x, y, z, w], shape (4,) or (N, 4)
    :param q2: End quaternion [x, y, z, w], shape (4,) or (N, 4)
    :param t: Interpolation parameter [0, 1], shape () or (N,)
    :return: Interpolated quaternion, shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    q1 = bm.ensure_array(q1)
    q2 = bm.ensure_array(q2)
    t = bm.ensure_array(t)
    
    # Normalize quaternions
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    
    batch = q1.ndim > 1 or q2.ndim > 1 or t.ndim > 0
    
    if not batch:
        q1 = q1.reshape(1, 4)
        q2 = q2.reshape(1, 4)
        t = t.reshape(1)
    
    # Compute dot product
    dot = (q1 * q2).sum(axis=-1)
    
    # If dot < 0, negate q2 to take shorter path
    if bm.is_torch:
        q2 = xp.where((dot < 0)[..., None], -q2, q2)
    else:
        q2 = xp.where((dot < 0)[..., None], -q2, q2)
    dot = xp.abs(dot)
    
    # If quaternions are very close, use linear interpolation
    threshold = 0.9995
    
    if bm.is_torch:
        # SLERP
        theta = xp.arccos(xp.clip(dot, -1.0, 1.0))
        sin_theta = xp.sin(theta)
        
        # Avoid division by zero
        safe_sin = xp.maximum(sin_theta, 1e-10)
        
        w1 = xp.sin((1 - t) * theta) / safe_sin
        w2 = xp.sin(t * theta) / safe_sin
        
        # Linear interpolation fallback
        w1_linear = 1 - t
        w2_linear = t
        
        # Choose based on threshold
        use_slerp = dot < threshold
        w1 = xp.where(use_slerp, w1, w1_linear)
        w2 = xp.where(use_slerp, w2, w2_linear)
        
        q = w1[..., None] * q1 + w2[..., None] * q2
    else:
        theta = xp.arccos(xp.clip(dot, -1.0, 1.0))
        sin_theta = xp.sin(theta)
        
        safe_sin = xp.maximum(sin_theta, 1e-10)
        
        w1 = xp.sin((1 - t) * theta) / safe_sin
        w2 = xp.sin(t * theta) / safe_sin
        
        w1_linear = 1 - t
        w2_linear = t
        
        use_slerp = dot < threshold
        w1 = xp.where(use_slerp, w1, w1_linear)
        w2 = xp.where(use_slerp, w2, w2_linear)
        
        q = w1[..., None] * q1 + w2[..., None] * q2
    
    q = quaternion_normalize(q)
    
    return q if batch else q[0]


def rotation_interpolate(R1, R2, t, method='slerp'):
    """
    Interpolate between two rotation matrices.
    
    :param R1: Start rotation matrix, shape (3, 3)
    :param R2: End rotation matrix, shape (3, 3)
    :param t: Interpolation parameter [0, 1]
    :param method: Interpolation method ('slerp' or 'linear')
    :return: Interpolated rotation matrix, shape (3, 3)
    """
    if method == 'slerp':
        # Convert to quaternions and use SLERP
        q1 = matrix_to_quaternion(R1)
        q2 = matrix_to_quaternion(R2)
        q_interp = slerp(q1, q2, t)
        return quaternion_to_matrix(q_interp)
    elif method == 'linear':
        # Simple linear interpolation (not geodesic)
        bm = get_backend_manager()
        R1 = bm.ensure_array(R1)
        R2 = bm.ensure_array(R2)
        t = bm.ensure_array(t)
        return (1 - t) * R1 + t * R2
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def rotation_distance(R1, R2):
    """
    Compute angular distance between two rotations (in radians).
    
    :param R1: First rotation matrix, shape (3, 3) or (N, 3, 3)
    :param R2: Second rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: Angular distance in radians, shape () or (N,)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R1 = bm.ensure_array(R1)
    R2 = bm.ensure_array(R2)
    
    # R_diff = R1^T @ R2
    R_diff = rotation_multiply(rotation_inverse(R1), R2)
    
    # angle = arccos((trace(R_diff) - 1) / 2)
    if R_diff.ndim == 2:
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
    else:
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    angle = xp.arccos(xp.clip((trace - 1) / 2, -1.0, 1.0))
    return angle


def rotation_error(R_current, R_target):
    """
    Compute rotation error as axis-angle vector.
    
    :param R_current: Current rotation matrix, shape (3, 3)
    :param R_target: Target rotation matrix, shape (3, 3)
    :return: Error vector (axis * angle), shape (3,)
    """
    bm = get_backend_manager()
    
    R_current = bm.ensure_array(R_current)
    R_target = bm.ensure_array(R_target)
    
    # R_error = R_current^T @ R_target
    R_error = rotation_multiply(rotation_inverse(R_current), R_target)
    
    # Convert to axis-angle
    axis, angle = matrix_to_axis_angle(R_error)
    
    # Return compact form
    from .conversions import axis_angle_to_compact
    return axis_angle_to_compact(axis, angle)


def orientation_error(R_current, R_target):
    """
    Compute orientation error (alias for rotation_error for backward compatibility).
    
    :param R_current: Current rotation matrix, shape (3, 3)
    :param R_target: Target rotation matrix, shape (3, 3)
    :return: Error vector (axis * angle), shape (3,)
    """
    return rotation_error(R_current, R_target)


def is_rotation_matrix(R, tol=1e-6):
    """
    Check if matrix is a valid rotation matrix.
    
    Validates:
    - R @ R^T = I (orthogonality)
    - det(R) = 1 (proper rotation)
    
    :param R: Matrix to check, shape (3, 3) or (N, 3, 3)
    :param tol: Tolerance for checks
    :return: Boolean or boolean array
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R = bm.ensure_array(R)
    batch = R.ndim == 3
    
    if not batch:
        R = R.reshape(1, 3, 3)
    
    # Check orthogonality: R @ R^T = I
    if bm.is_torch:
        RRT = xp.bmm(R, R.transpose(-2, -1))
    else:
        RRT = R @ R.transpose(0, 2, 1)
    
    I = bm.eye(3)
    if batch:
        if bm.is_torch:
            I = I.unsqueeze(0).expand(len(R), -1, -1)
        else:
            I = xp.tile(I, (len(R), 1, 1))
    
    ortho_error = xp.abs(RRT - I).max(axis=(-2, -1))
    is_ortho = ortho_error < tol
    
    # Check determinant = 1
    det = xp.linalg.det(R)
    is_proper = xp.abs(det - 1.0) < tol
    
    valid = is_ortho & is_proper
    
    return valid if batch else valid[0]


def is_transform_matrix(T, tol=1e-6):
    """
    Check if matrix is a valid SE(3) transformation matrix.
    
    Validates:
    - Bottom row is [0, 0, 0, 1]
    - Rotation part is valid
    
    :param T: Matrix to check, shape (4, 4) or (N, 4, 4)
    :param tol: Tolerance for checks
    :return: Boolean or boolean array
    """
    bm = get_backend_manager()
    xp = bm.module
    
    T = bm.ensure_array(T)
    batch = T.ndim == 3
    
    if not batch:
        T = T.reshape(1, 4, 4)
    
    # Check bottom row
    expected_bottom = bm.array([0, 0, 0, 1])
    if batch:
        if bm.is_torch:
            expected_bottom = expected_bottom.unsqueeze(0).expand(len(T), -1)
        else:
            expected_bottom = xp.tile(expected_bottom, (len(T), 1))
    
    bottom_error = xp.abs(T[..., 3, :] - expected_bottom).max(axis=-1)
    valid_bottom = bottom_error < tol
    
    # Check rotation part
    R = T[..., :3, :3]
    valid_rotation = is_rotation_matrix(R, tol)
    
    valid = valid_bottom & valid_rotation
    
    return valid if batch else valid[0]


def look_at(eye, target, up):
    """
    Create look-at transformation matrix (camera/view matrix).
    
    :param eye: Camera position, shape (3,)
    :param target: Target position to look at, shape (3,)
    :param up: Up direction vector, shape (3,)
    :return: Transformation matrix, shape (4, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    eye = bm.ensure_array(eye)
    target = bm.ensure_array(target)
    up = bm.ensure_array(up)
    
    # Forward direction (from eye to target)
    forward = target - eye
    forward = forward / xp.linalg.norm(forward)
    
    # Right direction (perpendicular to forward and up)
    right = xp.cross(forward, up)
    right = right / xp.linalg.norm(right)
    
    # Recompute up (perpendicular to right and forward)
    up_ortho = xp.cross(right, forward)
    
    # Build rotation matrix (columns are right, up, -forward)
    if bm.is_torch:
        R = xp.stack([right, up_ortho, -forward], dim=1)
    else:
        R = xp.stack([right, up_ortho, -forward], axis=1)
    
    # Build transformation matrix
    from .se3 import make_transform
    T = make_transform(R, eye)
    
    return T
