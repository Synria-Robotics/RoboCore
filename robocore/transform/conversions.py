"""Rotation representation conversions.

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
from .so3 import (
    rpy_to_matrix, 
    quaternion_to_matrix, 
    axis_angle_to_matrix,
    skew_symmetric
)


# ============================================================================
# Rotation Matrix → Other representations
# ============================================================================

def matrix_to_rpy(R):
    """
    Convert rotation matrix to Roll-Pitch-Yaw (Robotics convention).
    
    RPY corresponds to extrinsic XYZ rotations: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Expanded matrix (cr=cos(roll), sr=sin(roll), etc.):
    R = [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr ]
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr ]
        [ -sp,    cp*sr,              cp*cr            ]
    
    Extraction formulas:
    - pitch = arcsin(-R[2,0])
    - roll = atan2(R[2,1], R[2,2])
    - yaw = atan2(R[1,0], R[0,0])
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: [roll, pitch, yaw] in radians, shape (3,) or (N, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R = bm.ensure_array(R)
    batch = R.ndim == 3
    
    if not batch:
        R = R.reshape(1, 3, 3)
    
    # Extract pitch from R[2,0] = -sin(pitch)
    pitch = xp.arcsin(xp.clip(-R[..., 2, 0], -1.0, 1.0))
    
    # Handle gimbal lock (cos(pitch) ≈ 0)
    cos_pitch = xp.cos(pitch)
    threshold = 1e-6
    
    if bm.is_torch:
        # Non-singular: roll = atan2(R[2,1], R[2,2]),  yaw = atan2(R[1,0], R[0,0])
        roll = xp.arctan2(R[..., 2, 1], R[..., 2, 2])
        yaw = xp.arctan2(R[..., 1, 0], R[..., 0, 0])
        
        # Gimbal lock fallback
        gimbal = xp.abs(cos_pitch) <= threshold
        roll = xp.where(gimbal, xp.zeros_like(roll), roll)
        yaw = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 1, 1]), yaw)
    else:
        roll = xp.arctan2(R[..., 2, 1], R[..., 2, 2])
        yaw = xp.arctan2(R[..., 1, 0], R[..., 0, 0])
        
        gimbal = xp.abs(cos_pitch) <= threshold
        roll = xp.where(gimbal, 0.0, roll)
        yaw = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 1, 1]), yaw)
    
    if bm.is_torch:
        rpy = xp.stack([roll, pitch, yaw], dim=-1)
    else:
        rpy = xp.stack([roll, pitch, yaw], axis=-1)
    
    return rpy if batch else rpy[0]


def matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion [x, y, z, w].
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R = bm.ensure_array(R)
    batch = R.ndim == 3
    
    if not batch:
        R = R.reshape(1, 3, 3)
    
    # Shepperd's method for numerical stability
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    if bm.is_torch:
        q = xp.zeros((len(R), 4), dtype=bm.get_dtype(), device=bm.get_device())
        
        # Case 1: w is largest
        mask_w = trace > 0
        s = xp.sqrt(trace[mask_w] + 1.0) * 2
        q[mask_w, 3] = 0.25 * s
        q[mask_w, 0] = (R[mask_w, 2, 1] - R[mask_w, 1, 2]) / s
        q[mask_w, 1] = (R[mask_w, 0, 2] - R[mask_w, 2, 0]) / s
        q[mask_w, 2] = (R[mask_w, 1, 0] - R[mask_w, 0, 1]) / s
        
        # Case 2: x is largest
        mask_x = (~mask_w) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
        s = xp.sqrt(1.0 + R[mask_x, 0, 0] - R[mask_x, 1, 1] - R[mask_x, 2, 2]) * 2
        q[mask_x, 3] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / s
        q[mask_x, 0] = 0.25 * s
        q[mask_x, 1] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / s
        q[mask_x, 2] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / s
        
        # Case 3: y is largest
        mask_y = (~mask_w) & (~mask_x) & (R[..., 1, 1] > R[..., 2, 2])
        s = xp.sqrt(1.0 + R[mask_y, 1, 1] - R[mask_y, 0, 0] - R[mask_y, 2, 2]) * 2
        q[mask_y, 3] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / s
        q[mask_y, 0] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / s
        q[mask_y, 1] = 0.25 * s
        q[mask_y, 2] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / s
        
        # Case 4: z is largest
        mask_z = (~mask_w) & (~mask_x) & (~mask_y)
        s = xp.sqrt(1.0 + R[mask_z, 2, 2] - R[mask_z, 0, 0] - R[mask_z, 1, 1]) * 2
        q[mask_z, 3] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / s
        q[mask_z, 0] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / s
        q[mask_z, 1] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / s
        q[mask_z, 2] = 0.25 * s
    else:
        q = xp.zeros((len(R), 4), dtype=bm.get_dtype())
        
        for i in range(len(R)):
            if trace[i] > 0:
                s = xp.sqrt(trace[i] + 1.0) * 2
                q[i, 3] = 0.25 * s
                q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
                q[i, 1] = (R[i, 0, 2] - R[i, 2, 0]) / s
                q[i, 2] = (R[i, 1, 0] - R[i, 0, 1]) / s
            elif R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
                s = xp.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                q[i, 3] = (R[i, 2, 1] - R[i, 1, 2]) / s
                q[i, 0] = 0.25 * s
                q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
                q[i, 2] = (R[i, 0, 2] + R[i, 2, 0]) / s
            elif R[i, 1, 1] > R[i, 2, 2]:
                s = xp.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                q[i, 3] = (R[i, 0, 2] - R[i, 2, 0]) / s
                q[i, 0] = (R[i, 0, 1] + R[i, 1, 0]) / s
                q[i, 1] = 0.25 * s
                q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s
            else:
                s = xp.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / s
                q[i, 0] = (R[i, 0, 2] + R[i, 2, 0]) / s
                q[i, 1] = (R[i, 1, 2] + R[i, 2, 1]) / s
                q[i, 2] = 0.25 * s
    
    return q if batch else q[0]


def matrix_to_axis_angle(R):
    """
    Convert rotation matrix to axis-angle representation.
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: (axis, angle) where axis is shape (3,) or (N, 3), angle is () or (N,)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R = bm.ensure_array(R)
    batch = R.ndim == 3
    
    if not batch:
        R = R.reshape(1, 3, 3)
    
    # Angle from trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = xp.arccos(xp.clip((trace - 1) / 2, -1.0, 1.0))
    
    # Axis from skew-symmetric part
    if bm.is_torch:
        axis = xp.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim=-1)
    else:
        axis = xp.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], axis=-1)
    
    # Normalize axis
    axis_norm = xp.linalg.norm(axis, axis=-1, keepdims=True)
    eps = bm.ensure_array(1e-10) if bm.is_torch else 1e-10
    axis = axis / xp.maximum(axis_norm, eps)
    
    if not batch:
        return axis[0], angle[0]
    return axis, angle

def matrix_to_euler(R, seq='xyz'):
    """
    Convert rotation matrix to Euler angles.
    
    Supports both intrinsic (lowercase) and extrinsic (uppercase) conventions:
    - Intrinsic (e.g., 'xyz'): rotations about rotating axes
      'xyz' → first rotate about X, then new Y, then new Z
      Matrix: R = Rz(γ) @ Ry(β) @ Rx(α)
    
    - Extrinsic (e.g., 'XYZ'): rotations about fixed axes  
      'XYZ' → first rotate about fixed X, then fixed Y, then fixed Z
      Matrix: R = Rx(α) @ Ry(β) @ Rz(γ)
    
    This matches SciPy's Rotation.from_euler() convention.
    
    Supported sequences: xyz, XYZ, zyx, ZYX, xzy, XZY, yxz, YXZ, yzx, YZX, zxy, ZXY
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :param seq: Rotation sequence (lowercase=intrinsic, uppercase=extrinsic)
    :return: Euler angles [α, β, γ], shape (3,) or (N, 3)
    """
    bm = get_backend_manager()
    xp = bm.module

    R = bm.ensure_array(R)
    batch = R.ndim == 3

    if not batch:
        R = R.reshape(1, 3, 3)

    # Determine intrinsic vs extrinsic
    is_intrinsic = seq.islower()
    seq_lower = seq.lower()
    
    # For extrinsic, convert to equivalent intrinsic problem
    # Extrinsic ABC = Intrinsic CBA (reversed sequence, same angles)
    if not is_intrinsic:
        seq_lower = seq_lower[::-1]  # reverse sequence
    
    # Now solve as intrinsic
    if seq_lower == 'zyx':
        # ZYX Euler (intrinsic): First rotate Z by α, then new Y by β, then new X by γ
        # Equivalent extrinsic: R = Rx(γ) @ Ry(β) @ Rz(α)
        # Matrix elements:
        # R = [ cβ*cα,              -cβ*sα,              sβ    ]
        #     [ sγ*sβ*cα + cγ*sα,   -sγ*sβ*sα + cγ*cα,   -sγ*cβ ]
        #     [-cγ*sβ*cα + sγ*sα,    cγ*sβ*sα + sγ*cα,    cγ*cβ ]

        # Extract β from R[0,2] = sin(β)
        beta = xp.arcsin(xp.clip(R[..., 0, 2], -1.0, 1.0))

        # Handle gimbal lock
        cos_beta = xp.cos(beta)
        threshold = 1e-6

        if bm.is_torch:
            # Non-singular: α = atan2(-R[0,1], R[0,0]),  γ = atan2(-R[1,2], R[2,2])
            alpha = xp.arctan2(-R[..., 0, 1], R[..., 0, 0])
            gamma = xp.arctan2(-R[..., 1, 2], R[..., 2, 2])

            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 1, 1]), gamma)
        else:
            alpha = xp.arctan2(-R[..., 0, 1], R[..., 0, 0])
            gamma = xp.arctan2(-R[..., 1, 2], R[..., 2, 2])

            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 1, 1]), gamma)

        if bm.is_torch:
            euler = xp.stack([alpha, beta, gamma], dim=-1)
        else:
            euler = xp.stack([alpha, beta, gamma], axis=-1)

    # Now solve as intrinsic
    if seq_lower == 'zyx':
        # ZYX intrinsic: R = Rx(γ) @ Ry(β) @ Rz(α)
        beta = xp.arcsin(xp.clip(R[..., 0, 2], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(-R[..., 0, 1], R[..., 0, 0])
            gamma = xp.arctan2(-R[..., 1, 2], R[..., 2, 2])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 1, 1]), gamma)
        else:
            alpha = xp.arctan2(-R[..., 0, 1], R[..., 0, 0])
            gamma = xp.arctan2(-R[..., 1, 2], R[..., 2, 2])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 1, 1]), gamma)
        
        angles = [alpha, beta, gamma]
        
    elif seq_lower == 'xyz':
        # XYZ intrinsic: R = Rz(γ) @ Ry(β) @ Rx(α)
        beta = xp.arcsin(xp.clip(-R[..., 2, 0], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(R[..., 2, 1], R[..., 2, 2])
            gamma = xp.arctan2(R[..., 1, 0], R[..., 0, 0])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 1, 1]), gamma)
        else:
            alpha = xp.arctan2(R[..., 2, 1], R[..., 2, 2])
            gamma = xp.arctan2(R[..., 1, 0], R[..., 0, 0])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 1, 1]), gamma)
        
        angles = [alpha, beta, gamma]
        
    elif seq_lower == 'xzy':
        # XZY intrinsic: R = Ry(γ) @ Rz(β) @ Rx(α)
        # R[1,0] = sin(β)
        beta = xp.arcsin(xp.clip(R[..., 1, 0], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(-R[..., 1, 2], R[..., 1, 1])
            gamma = xp.arctan2(-R[..., 2, 0], R[..., 0, 0])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 2, 1], R[..., 2, 2]), gamma)
        else:
            alpha = xp.arctan2(-R[..., 1, 2], R[..., 1, 1])
            gamma = xp.arctan2(-R[..., 2, 0], R[..., 0, 0])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 2, 1], R[..., 2, 2]), gamma)
        
        angles = [alpha, beta, gamma]
        
    elif seq_lower == 'yxz':
        # YXZ intrinsic: R = Rz(γ) @ Rx(β) @ Ry(α)
        # R[2,1] = sin(β)
        beta = xp.arcsin(xp.clip(R[..., 2, 1], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(-R[..., 2, 0], R[..., 2, 2])
            gamma = xp.arctan2(-R[..., 0, 1], R[..., 1, 1])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 0, 0]), gamma)
        else:
            alpha = xp.arctan2(-R[..., 2, 0], R[..., 2, 2])
            gamma = xp.arctan2(-R[..., 0, 1], R[..., 1, 1])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 1, 0], R[..., 0, 0]), gamma)
        
        angles = [alpha, beta, gamma]
        
    elif seq_lower == 'yzx':
        # YZX intrinsic: R = Rx(γ) @ Rz(β) @ Ry(α)
        # R[0,1] = -sin(β)
        beta = xp.arcsin(xp.clip(-R[..., 0, 1], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(R[..., 0, 2], R[..., 0, 0])
            gamma = xp.arctan2(R[..., 2, 1], R[..., 1, 1])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 2, 0], R[..., 2, 2]), gamma)
        else:
            alpha = xp.arctan2(R[..., 0, 2], R[..., 0, 0])
            gamma = xp.arctan2(R[..., 2, 1], R[..., 1, 1])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 2, 0], R[..., 2, 2]), gamma)
            gamma = xp.where(gimbal, xp.arctan2(R[..., 0, 2], R[..., 2, 2]), gamma)
        
        angles = [alpha, beta, gamma]
        
    elif seq_lower == 'zxy':
        # ZXY intrinsic: R = Ry(γ) @ Rx(β) @ Rz(α)
        # R[1,2] = -sin(β)
        beta = xp.arcsin(xp.clip(-R[..., 1, 2], -1.0, 1.0))
        cos_beta = xp.cos(beta)
        threshold = 1e-6
        
        if bm.is_torch:
            alpha = xp.arctan2(R[..., 1, 0], R[..., 1, 1])
            gamma = xp.arctan2(R[..., 0, 2], R[..., 2, 2])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, xp.zeros_like(alpha), alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 0, 0]), gamma)
        else:
            alpha = xp.arctan2(R[..., 1, 0], R[..., 1, 1])
            gamma = xp.arctan2(R[..., 0, 2], R[..., 2, 2])
            gimbal = xp.abs(cos_beta) <= threshold
            alpha = xp.where(gimbal, 0.0, alpha)
            gamma = xp.where(gimbal, xp.arctan2(-R[..., 0, 1], R[..., 0, 0]), gamma)
        
        angles = [alpha, beta, gamma]
        
    else:
        raise NotImplementedError(
            f"Euler sequence '{seq}' not implemented. "
            f"Supported: xyz/XYZ, zyx/ZYX, xzy/XZY, yxz/YXZ, yzx/YZX, zxy/ZXY"
        )
    
    # If extrinsic, reverse the angle order to match extrinsic convention
    if not is_intrinsic:
        angles = angles[::-1]
    
    if bm.is_torch:
        euler = xp.stack(angles, dim=-1)
    else:
        euler = xp.stack(angles, axis=-1)
    
    return euler if batch else euler[0]


# ============================================================================
# Quaternion operations and conversions
# ============================================================================

def quaternion_normalize(q):
    """
    Normalize quaternion to unit length.
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: Normalized quaternion, shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    q = bm.ensure_array(q)
    norm = xp.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm


def quaternion_conjugate(q):
    """
    Compute quaternion conjugate: [x,y,z,w] → [-x,-y,-z,w].
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: Conjugate quaternion, shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    q = bm.ensure_array(q)
    
    q_conj = q.copy() if bm.is_numpy else q.clone()
    q_conj[..., :3] = -q_conj[..., :3]
    return q_conj


def quaternion_inverse(q):
    """
    Compute quaternion inverse (conjugate for unit quaternions).
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: Inverse quaternion, shape (4,) or (N, 4)
    """
    return quaternion_normalize(quaternion_conjugate(q))


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q = q1 * q2.
    
    :param q1: First quaternion [x, y, z, w], shape (4,) or (N, 4)
    :param q2: Second quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: Product quaternion, shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    q1 = bm.ensure_array(q1)
    q2 = bm.ensure_array(q2)
    
    batch = q1.ndim > 1 or q2.ndim > 1
    
    if not batch:
        q1 = q1.reshape(1, 4)
        q2 = q2.reshape(1, 4)
    
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    if bm.is_torch:
        q = xp.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dim=-1)
    else:
        q = xp.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], axis=-1)
    
    return q if batch else q[0]


def quaternion_to_rpy(q):
    """
    Convert quaternion to Roll-Pitch-Yaw.
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: [roll, pitch, yaw], shape (3,) or (N, 3)
    """
    R = quaternion_to_matrix(q)
    return matrix_to_rpy(R)


def quaternion_to_axis_angle(q):
    """
    Convert quaternion to axis-angle.
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: (axis, angle) where axis is shape (3,) or (N, 3), angle is () or (N,)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    q = bm.ensure_array(q)
    q = quaternion_normalize(q)
    
    batch = q.ndim > 1
    if not batch:
        q = q.reshape(1, 4)
    
    # angle = 2 * arccos(w)
    angle = 2 * xp.arccos(xp.clip(q[..., 3], -1.0, 1.0))
    
    # axis = [x, y, z] / sin(angle/2)
    sin_half = xp.sin(angle / 2)
    axis = q[..., :3] / xp.maximum(sin_half[..., None], 1e-10)
    
    if not batch:
        return axis[0], angle[0]
    return axis, angle


# ============================================================================
# RPY conversions
# ============================================================================

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert Roll-Pitch-Yaw to quaternion.
    
    :param roll: Roll angle in radians, shape () or (N,)
    :param pitch: Pitch angle in radians, shape () or (N,)
    :param yaw: Yaw angle in radians, shape () or (N,)
    :return: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    """
    R = rpy_to_matrix(roll, pitch, yaw)
    return matrix_to_quaternion(R)


def rpy_to_axis_angle(roll, pitch, yaw):
    """
    Convert Roll-Pitch-Yaw to axis-angle.
    
    :param roll: Roll angle in radians, shape () or (N,)
    :param pitch: Pitch angle in radians, shape () or (N,)
    :param yaw: Yaw angle in radians, shape () or (N,)
    :return: (axis, angle) where axis is shape (3,) or (N, 3), angle is () or (N,)
    """
    R = rpy_to_matrix(roll, pitch, yaw)
    return matrix_to_axis_angle(R)


# ============================================================================
# Axis-angle conversions
# ============================================================================

def axis_angle_to_quaternion(axis, angle):
    """
    Convert axis-angle to quaternion.
    
    :param axis: Rotation axis (unit vector), shape (3,) or (N, 3)
    :param angle: Rotation angle in radians, shape () or (N,)
    :return: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    axis = bm.ensure_array(axis)
    angle = bm.ensure_array(angle)
    
    batch = axis.ndim > 1 or angle.ndim > 0
    
    if not batch:
        axis = axis.reshape(1, 3)
        angle = angle.reshape(1)
    
    # Normalize axis
    axis = axis / xp.linalg.norm(axis, axis=-1, keepdims=True)
    
    half_angle = angle / 2
    sin_half = xp.sin(half_angle)
    cos_half = xp.cos(half_angle)
    
    if bm.is_torch:
        q = xp.zeros((len(axis), 4), dtype=bm.get_dtype(), device=bm.get_device())
    else:
        q = xp.zeros((len(axis), 4), dtype=bm.get_dtype())
    
    q[..., :3] = axis * sin_half[..., None]
    q[..., 3] = cos_half
    
    return q if batch else q[0]


def axis_angle_to_rpy(axis, angle):
    """
    Convert axis-angle to Roll-Pitch-Yaw.
    
    :param axis: Rotation axis (unit vector), shape (3,) or (N, 3)
    :param angle: Rotation angle in radians, shape () or (N,)
    :return: [roll, pitch, yaw], shape (3,) or (N, 3)
    """
    R = axis_angle_to_matrix(axis, angle)
    return matrix_to_rpy(R)


def axis_angle_to_compact(axis, angle):
    """
    Convert axis-angle to compact representation (axis * angle).
    
    :param axis: Rotation axis (unit vector), shape (3,) or (N, 3)
    :param angle: Rotation angle in radians, shape () or (N,)
    :return: Compact vector, shape (3,) or (N, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    axis = bm.ensure_array(axis)
    angle = bm.ensure_array(angle)
    
    batch = axis.ndim > 1 or angle.ndim > 0
    
    if not batch:
        return axis * angle
    
    return axis * angle[..., None]


def compact_to_axis_angle(compact):
    """
    Convert compact representation to axis-angle.
    
    :param compact: Compact vector (axis * angle), shape (3,) or (N, 3)
    :return: (axis, angle) where axis is shape (3,) or (N, 3), angle is () or (N,)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    compact = bm.ensure_array(compact)
    batch = compact.ndim > 1
    
    angle = xp.linalg.norm(compact, axis=-1)
    axis = compact / xp.maximum(angle[..., None], 1e-10)
    
    if not batch:
        return axis, angle
    return axis, angle
