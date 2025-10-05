"""SE(3) homogeneous transformation operations (4x4 matrices).

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


def make_transform(R, t):
    """
    Create 4x4 homogeneous transformation matrix from rotation and translation.
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :param t: Translation vector, shape (3,) or (N, 3)
    :return: Transformation matrix, shape (4, 4) or (N, 4, 4)
    """
    bm = get_backend_manager()
    
    R = bm.ensure_array(R)
    t = bm.ensure_array(t)
    
    batch = R.ndim == 3 or t.ndim == 2
    
    if not batch:
        # Single transform
        T = bm.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    else:
        # Batch transforms
        if R.ndim == 2:
            R = R.reshape(1, 3, 3)
        if t.ndim == 1:
            t = t.reshape(1, 3)
        
        n = max(R.shape[0], t.shape[0])
        
        if bm.is_torch:
            T = bm.eye(4).unsqueeze(0).repeat(n, 1, 1)
        else:
            T = bm.module.tile(bm.eye(4), (n, 1, 1))
        
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        
        return T


def translation_transform(t):
    """
    Create pure translation transformation.
    
    :param t: Translation vector, shape (3,) or (N, 3)
    :return: Transformation matrix, shape (4, 4) or (N, 4, 4)
    """
    bm = get_backend_manager()
    t = bm.ensure_array(t)
    
    batch = t.ndim > 1
    
    if not batch:
        R = bm.eye(3)
        return make_transform(R, t)
    else:
        n = t.shape[0]
        if bm.is_torch:
            R = bm.eye(3).unsqueeze(0).repeat(n, 1, 1)
        else:
            R = bm.module.tile(bm.eye(3), (n, 1, 1))
        return make_transform(R, t)


def rotation_transform(R):
    """
    Create pure rotation transformation (zero translation).
    
    :param R: Rotation matrix, shape (3, 3) or (N, 3, 3)
    :return: Transformation matrix, shape (4, 4) or (N, 4, 4)
    """
    bm = get_backend_manager()
    R = bm.ensure_array(R)
    
    batch = R.ndim == 3
    
    if not batch:
        t = bm.zeros(3)
        return make_transform(R, t)
    else:
        n = R.shape[0]
        t = bm.zeros((n, 3))
        return make_transform(R, t)


def get_rotation(T):
    """
    Extract 3x3 rotation part from transformation matrix.
    
    :param T: Transformation matrix, shape (4, 4) or (N, 4, 4)
    :return: Rotation matrix, shape (3, 3) or (N, 3, 3)
    """
    bm = get_backend_manager()
    T = bm.ensure_array(T)
    return T[..., :3, :3]


def get_translation(T):
    """
    Extract translation vector from transformation matrix.
    
    :param T: Transformation matrix, shape (4, 4) or (N, 4, 4)
    :return: Translation vector, shape (3,) or (N, 3)
    """
    bm = get_backend_manager()
    T = bm.ensure_array(T)
    return T[..., :3, 3]


def transform_multiply(T1, T2):
    """
    Multiply two transformation matrices: T = T1 @ T2.
    
    :param T1: First transformation, shape (4, 4) or (N, 4, 4)
    :param T2: Second transformation, shape (4, 4) or (N, 4, 4)
    :return: Product transformation, shape (4, 4) or (N, 4, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    T1 = bm.ensure_array(T1)
    T2 = bm.ensure_array(T2)
    
    batch = T1.ndim == 3 or T2.ndim == 3
    
    if bm.is_torch and batch:
        if T1.ndim == 2:
            T1 = T1.unsqueeze(0)
        if T2.ndim == 2:
            T2 = T2.unsqueeze(0)
        result = xp.bmm(T1, T2)
        return result.squeeze(0) if (T1.shape[0] == 1 and T2.shape[0] == 1) else result
    else:
        return T1 @ T2


def transform_inverse(T):
    """
    Compute inverse of transformation matrix.
    
    For SE(3): T^{-1} = [R^T, -R^T @ t; 0, 1]
    
    :param T: Transformation matrix, shape (4, 4) or (N, 4, 4)
    :return: Inverse transformation, shape (4, 4) or (N, 4, 4)
    """
    bm = get_backend_manager()
    xp = bm.module
    
    T = bm.ensure_array(T)
    batch = T.ndim == 3
    
    R = get_rotation(T)
    t = get_translation(T)
    
    # R^T
    if batch:
        if bm.is_torch:
            R_inv = R.transpose(-2, -1)
        else:
            R_inv = R.transpose(0, 2, 1)
    else:
        R_inv = R.T
    
    # -R^T @ t
    if batch:
        if bm.is_torch:
            t_inv = -xp.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
        else:
            t_inv = -xp.einsum('nij,nj->ni', R_inv, t)
    else:
        t_inv = -R_inv @ t
    
    return make_transform(R_inv, t_inv)


def transform_apply(T, points):
    """
    Apply transformation to points (homogeneous coordinates).
    
    :param T: Transformation matrix, shape (4, 4) or (N, 4, 4)
    :param points: Points to transform, shape (3,), (N, 3), or (M, 3)
    :return: Transformed points, shape matches input
    """
    bm = get_backend_manager()
    xp = bm.module
    
    T = bm.ensure_array(T)
    points = bm.ensure_array(points)
    
    R = get_rotation(T)
    t = get_translation(T)
    
    # Apply: p' = R @ p + t
    if T.ndim == 2 and points.ndim == 1:
        # Single transform, single point
        return R @ points + t
    elif T.ndim == 2 and points.ndim == 2:
        # Single transform, multiple points
        return (R @ points.T).T + t
    elif T.ndim == 3 and points.ndim == 2:
        # Batch transforms, batch points
        if bm.is_torch:
            return xp.bmm(R, points.unsqueeze(-1)).squeeze(-1) + t
        else:
            return xp.einsum('nij,nj->ni', R, points) + t
    else:
        # General case
        if bm.is_torch:
            return xp.matmul(R, points.unsqueeze(-1)).squeeze(-1) + t
        else:
            return xp.einsum('...ij,...j->...i', R, points) + t


def transform_interpolate(T1, T2, t):
    """
    Interpolate between two transformations using SLERP for rotation.
    
    :param T1: Start transformation, shape (4, 4)
    :param T2: End transformation, shape (4, 4)
    :param t: Interpolation parameter [0, 1]
    :return: Interpolated transformation, shape (4, 4)
    """
    from .conversions import matrix_to_quaternion, quaternion_to_matrix
    from .utils import slerp
    
    bm = get_backend_manager()
    xp = bm.module
    
    T1 = bm.ensure_array(T1)
    T2 = bm.ensure_array(T2)
    t = bm.ensure_array(t)
    
    R1 = get_rotation(T1)
    R2 = get_rotation(T2)
    t1 = get_translation(T1)
    t2 = get_translation(T2)
    
    # Convert rotations to quaternions
    q1 = matrix_to_quaternion(R1)
    q2 = matrix_to_quaternion(R2)
    
    # SLERP for rotation
    q_interp = slerp(q1, q2, t)
    R_interp = quaternion_to_matrix(q_interp)
    
    # Linear interpolation for translation
    t_interp = t1 + t * (t2 - t1)
    
    return make_transform(R_interp, t_interp)
