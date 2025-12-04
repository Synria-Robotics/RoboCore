"""SO(3) rotation matrix operations (3x3 matrices).

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


def rotation_x(theta):
    """
    Rotation matrix around X-axis.
    
    :param theta: Rotation angle in radians, shape () or (N,)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    theta = bm.ensure_array(theta)
    
    batch = theta.ndim > 0
    if not batch:
        theta = theta.reshape(1)
    
    c = xp.cos(theta)
    s = xp.sin(theta)
    
    if bm.is_torch:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
        R[:, 0, 0] = 1
        R[:, 1, 1] = c
        R[:, 1, 2] = -s
        R[:, 2, 1] = s
        R[:, 2, 2] = c
    else:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype())
        R[:, 0, 0] = 1
        R[:, 1, 1] = c
        R[:, 1, 2] = -s
        R[:, 2, 1] = s
        R[:, 2, 2] = c
    
    return R if batch else R[0]


def rotation_y(theta):
    """
    Rotation matrix around Y-axis.
    
    :param theta: Rotation angle in radians, shape () or (N,)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    theta = bm.ensure_array(theta)
    
    batch = theta.ndim > 0
    if not batch:
        theta = theta.reshape(1)
    
    c = xp.cos(theta)
    s = xp.sin(theta)
    
    if bm.is_torch:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 1, 1] = 1
        R[:, 2, 0] = -s
        R[:, 2, 2] = c
    else:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype())
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 1, 1] = 1
        R[:, 2, 0] = -s
        R[:, 2, 2] = c
    
    return R if batch else R[0]


def rotation_z(theta):
    """
    Rotation matrix around Z-axis.
    
    :param theta: Rotation angle in radians, shape () or (N,)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    theta = bm.ensure_array(theta)
    
    batch = theta.ndim > 0
    if not batch:
        theta = theta.reshape(1)
    
    c = xp.cos(theta)
    s = xp.sin(theta)
    
    if bm.is_torch:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
    else:
        R = xp.zeros((len(theta), 3, 3), dtype=bm.get_dtype())
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
    
    return R if batch else R[0]


def rpy_to_matrix(roll, pitch, yaw):
    """
    Convert Roll-Pitch-Yaw to rotation matrix (ZYX Euler angles).
    
    :param roll: Roll angle in radians, shape () or (N,)
    :param pitch: Pitch angle in radians, shape () or (N,)
    :param yaw: Yaw angle in radians, shape () or (N,)
    :return: Rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll), shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    roll = bm.ensure_array(roll)
    pitch = bm.ensure_array(pitch)
    yaw = bm.ensure_array(yaw)
    
    # Detect batch mode
    batch = roll.ndim > 0 or pitch.ndim > 0 or yaw.ndim > 0
    
    if not batch:
        roll = roll.reshape(1) if roll.ndim == 0 else roll
        pitch = pitch.reshape(1) if pitch.ndim == 0 else pitch
        yaw = yaw.reshape(1) if yaw.ndim == 0 else yaw
    
    cr = xp.cos(roll)
    sr = xp.sin(roll)
    cp = xp.cos(pitch)
    sp = xp.sin(pitch)
    cy = xp.cos(yaw)
    sy = xp.sin(yaw)
    
    if bm.is_torch:
        R = xp.zeros((len(roll), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
    else:
        R = xp.zeros((len(roll), 3, 3), dtype=bm.get_dtype())
    
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    
    return R if batch else R[0]


def axis_angle_to_matrix(axis, angle):
    """
    Convert axis-angle to rotation matrix using Rodrigues formula.
    
    :param axis: Rotation axis (unit vector), shape (3,) or (N, 3)
    :param angle: Rotation angle in radians, shape () or (N,)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    axis = bm.ensure_array(axis)
    angle = bm.ensure_array(angle)
    
    # Normalize axis
    if axis.ndim == 1:
        axis = axis / xp.linalg.norm(axis)
        batch = False
        axis = axis.reshape(1, 3)
        angle = angle.reshape(1) if angle.ndim == 0 else angle
    else:
        axis = axis / xp.linalg.norm(axis, axis=-1, keepdims=True)
        batch = True
        if angle.ndim == 0:
            angle = angle.reshape(1)
    
    K = skew_symmetric(axis)
    c = xp.cos(angle)
    s = xp.sin(angle)
    
    I = bm.eye(3)
    if batch:
        I = I.reshape(1, 3, 3)
    
    # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
    if bm.is_torch:
        R = I + s.reshape(-1, 1, 1) * K + (1 - c).reshape(-1, 1, 1) * xp.bmm(K, K)
    else:
        # For batch K: (N, 3, 3), compute K @ K for each element using matmul
        if K.ndim == 3:
            # Batch matrix multiplication
            K_squared = xp.matmul(K, K)
        else:
            K_squared = K @ K
        R = I + s.reshape(-1, 1, 1) * K + (1 - c).reshape(-1, 1, 1) * K_squared
    
    return R if batch else R[0]


def quaternion_to_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    :param q: Quaternion [x, y, z, w], shape (4,) or (N, 4)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    q = bm.ensure_array(q)
    batch = q.ndim > 1
    
    if not batch:
        q = q.reshape(1, 4)
    
    # Normalize quaternion
    q = q / xp.linalg.norm(q, axis=-1, keepdims=True)
    
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    if bm.is_torch:
        R = xp.zeros((len(q), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
    else:
        R = xp.zeros((len(q), 3, 3), dtype=bm.get_dtype())
    
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R if batch else R[0]


def euler_to_matrix(alpha, beta, gamma, seq='xyz'):
    """
    Convert Euler angles to rotation matrix.
    
    Supports both intrinsic (lowercase) and extrinsic (uppercase) conventions:
    
    **Intrinsic rotations** (lowercase, e.g., 'xyz'):
    - Rotations about rotating/body-fixed axes
    - 'xyz' intrinsic = rotate first around X, then new Y, then new Z
    - Matrix: R = Rz(γ) @ Ry(β) @ Rx(α)  [right-to-left application]
    
    **Extrinsic rotations** (uppercase, e.g., 'XYZ'):
    - Rotations about fixed/space axes
    - 'XYZ' extrinsic = rotate about fixed X, then fixed Y, then fixed Z
    - Matrix: R = Rx(α) @ Ry(β) @ Rz(γ)  [left-to-right application]
    
    This matches SciPy's Rotation.from_euler() convention.
    
    Supported: xyz/XYZ, zyx/ZYX, xzy/XZY, yxz/YXZ, yzx/YZX, zxy/ZXY
    
    :param alpha: First rotation angle in radians, shape () or (N,)
    :param beta: Second rotation angle in radians, shape () or (N,)
    :param gamma: Third rotation angle in radians, shape () or (N,)
    :param seq: Rotation sequence (lowercase=intrinsic, uppercase=extrinsic)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    is_intrinsic = seq.islower()
    seq_lower = seq.lower()
    
    # Get rotation functions
    rot_map = {'x': rotation_x, 'y': rotation_y, 'z': rotation_z}
    
    if is_intrinsic:
        # Intrinsic: R = R3(gamma) @ R2(beta) @ R1(alpha)
        R1 = rot_map[seq_lower[0]](alpha)
        R2 = rot_map[seq_lower[1]](beta)
        R3 = rot_map[seq_lower[2]](gamma)
        return rotation_multiply(R3, rotation_multiply(R2, R1))
    else:
        # Extrinsic: R = R1(alpha) @ R2(beta) @ R3(gamma)
        R1 = rot_map[seq_lower[0]](alpha)
        R2 = rot_map[seq_lower[1]](beta)
        R3 = rot_map[seq_lower[2]](gamma)
        return rotation_multiply(R1, rotation_multiply(R2, R3))


def rotation_multiply(R1, R2):
    """
    Multiply two rotation matrices: R = R1 @ R2.
    
    :param R1: First rotation matrix, shape (3, 3) or (N, 3, 3)
    :param R2: Second rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: Product rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R1 = bm.ensure_array(R1)
    R2 = bm.ensure_array(R2)
    
    batch = R1.ndim == 3 or R2.ndim == 3
    
    if bm.is_torch and batch:
        if R1.ndim == 2:
            R1 = R1.unsqueeze(0)
        if R2.ndim == 2:
            R2 = R2.unsqueeze(0)
        return xp.bmm(R1, R2).squeeze(0) if (R1.shape[0] == 1 and R2.shape[0] == 1) else xp.bmm(R1, R2)
    else:
        return R1 @ R2


def rotation_inverse(R):
    """
    Invert rotation matrix (transpose for orthogonal matrices).
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: Inverse rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    R = bm.ensure_array(R)
    
    if R.ndim == 2:
        return R.T
    else:
        if bm.is_torch:
            return R.transpose(-2, -1)
        else:
            return R.transpose(0, 2, 1)


def rotation_apply(R, vectors):
    """
    Apply rotation to vectors.
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :param vectors: Vectors to rotate, shape (3,), (N, 3), or (M, 3)
    :return: Rotated vectors, shape matches input
    """
    bm = get_backend_manager()
    xp = bm.module
    
    R = bm.ensure_array(R)
    vectors = bm.ensure_array(vectors)
    
    if R.ndim == 2 and vectors.ndim == 1:
        # Single rotation, single vector
        return R @ vectors
    elif R.ndim == 2 and vectors.ndim == 2:
        # Single rotation, multiple vectors
        return (R @ vectors.T).T
    elif R.ndim == 3 and vectors.ndim == 2:
        # Multiple rotations, multiple vectors (batch)
        if bm.is_torch:
            return xp.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)
        else:
            return xp.einsum('nij,nj->ni', R, vectors)
    else:
        # General case
        if bm.is_torch:
            return xp.matmul(R, vectors.unsqueeze(-1)).squeeze(-1)
        else:
            return xp.einsum('...ij,...j->...i', R, vectors)


def skew_symmetric(v):
    """
    Create skew-symmetric matrix from vector: [v]×
    
    :param v: Vector, shape (3,) or (N, 3)
    :return: Skew-symmetric matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    v = bm.ensure_array(v)
    batch = v.ndim > 1
    
    if not batch:
        v = v.reshape(1, 3)
    
    if bm.is_torch:
        K = xp.zeros((len(v), 3, 3), dtype=bm.get_dtype(), device=bm.get_device())
    else:
        K = xp.zeros((len(v), 3, 3), dtype=bm.get_dtype())
    
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    
    return K if batch else K[0]


def rotation_from_vectors(v1, v2):
    """
    Compute rotation matrix that rotates v1 to v2.
    
    :param v1: Source vector, shape (3,) or (N, 3)
    :param v2: Target vector, shape (3,) or (N, 3)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    v1 = bm.ensure_array(v1)
    v2 = bm.ensure_array(v2)
    
    batch = v1.ndim > 1 or v2.ndim > 1
    
    if not batch:
        v1 = v1.reshape(1, 3)
        v2 = v2.reshape(1, 3)
    
    # Normalize vectors
    v1 = v1 / xp.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / xp.linalg.norm(v2, axis=-1, keepdims=True)
    
    # Compute rotation axis and angle
    if bm.is_torch:
        cross = xp.cross(v1, v2, dim=-1)
        dot = (v1 * v2).sum(dim=-1)
    else:
        cross = xp.cross(v1, v2, axis=-1)
        dot = (v1 * v2).sum(axis=-1)
    
    angle = xp.arccos(xp.clip(dot, -1.0, 1.0))
    
    # Handle parallel vectors
    axis_norm = xp.linalg.norm(cross, axis=-1)
    
    # Use skew-symmetric approach
    K = skew_symmetric(cross)
    
    I = bm.eye(3)
    if batch:
        I = I.reshape(1, 3, 3)
    
    # Rodrigues formula: R = I + K + K²/(1+cos(θ))
    # For numerical stability when vectors are nearly parallel
    safe_norm = xp.maximum(axis_norm, 1e-10)
    K_normalized = K / safe_norm.reshape(-1, 1, 1)
    
    if bm.is_torch:
        K2 = xp.bmm(K_normalized, K_normalized)
    else:
        K2 = K_normalized @ K_normalized.transpose(0, 2, 1)
    
    R = I + xp.sin(angle).reshape(-1, 1, 1) * K_normalized + \
        (1 - xp.cos(angle)).reshape(-1, 1, 1) * K2
    
    return R if batch else R[0]
