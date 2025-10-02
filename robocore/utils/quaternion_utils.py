"""Quaternion utilities for RoboCore.

This module provides quaternion conversion utilities with a standardized
quaternion representation: **xyzw order** (x, y, z, w).

This is consistent with:
- scipy.spatial.transform.Rotation.as_quat() default
- ROS (Robot Operating System)
- PyBullet
- Many robotics libraries

Convention:
-----------
All quaternions in RoboCore use **xyzw order**: [qx, qy, qz, qw]
where qw is the scalar (real) part.

Examples:
---------
>>> import numpy as np
>>> from robocore.utils.quaternion_utils import rotation_matrix_to_quaternion
>>> 
>>> # Identity rotation
>>> R = np.eye(3)
>>> quat = rotation_matrix_to_quaternion(R)
>>> print(quat)  # [0, 0, 0, 1] - xyzw order
>>> 
>>> # 90 degree rotation around Z-axis
>>> R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
>>> quat = rotation_matrix_to_quaternion(R)
>>> print(quat)  # [0, 0, 0.707, 0.707] - xyzw order
"""

from __future__ import annotations
from typing import Union, Sequence
import numpy as np

# Check scipy availability
_HAS_SCIPY = False
try:
    from scipy.spatial.transform import Rotation
    _HAS_SCIPY = True
except ImportError:
    pass


def rotation_matrix_to_quaternion(
    rotation_matrix: np.ndarray,
    ensure_scipy: bool = False
) -> np.ndarray:
    """Convert rotation matrix to quaternion (xyzw order).
    
    Parameters
    ----------
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    ensure_scipy : bool, optional
        If True, require scipy (more accurate). If False and scipy is not
        available, use numpy-only implementation (by default False)
    
    Returns
    -------
    np.ndarray
        Quaternion in xyzw order: [qx, qy, qz, qw]
    
    Raises
    ------
    ImportError
        If ensure_scipy=True but scipy is not available
    ValueError
        If rotation_matrix is not a valid 3x3 matrix
    
    Examples
    --------
    >>> R = np.eye(3)
    >>> q = rotation_matrix_to_quaternion(R)
    >>> print(q)
    [0. 0. 0. 1.]
    """
    R = np.asarray(rotation_matrix)
    
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")
    
    if ensure_scipy and not _HAS_SCIPY:
        raise ImportError("scipy is required for rotation_matrix_to_quaternion with ensure_scipy=True")
    
    if _HAS_SCIPY:
        # Use scipy (default, most accurate)
        return Rotation.from_matrix(R).as_quat()  # xyzw order by default
    else:
        # Numpy-only implementation (Shepperd's method)
        return _rotation_matrix_to_quaternion_numpy(R)


def quaternion_to_rotation_matrix(
    quaternion: Union[np.ndarray, Sequence[float]],
    ensure_scipy: bool = False
) -> np.ndarray:
    """Convert quaternion (xyzw order) to rotation matrix.
    
    Parameters
    ----------
    quaternion : np.ndarray or sequence
        Quaternion in xyzw order: [qx, qy, qz, qw]
    ensure_scipy : bool, optional
        If True, require scipy (more accurate). If False and scipy is not
        available, use numpy-only implementation (by default False)
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    
    Raises
    ------
    ImportError
        If ensure_scipy=True but scipy is not available
    ValueError
        If quaternion is not a valid 4-element vector
    
    Examples
    --------
    >>> q = np.array([0, 0, 0, 1])  # Identity quaternion (xyzw)
    >>> R = quaternion_to_rotation_matrix(q)
    >>> print(R)
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    """
    q = np.asarray(quaternion, dtype=np.float64)
    
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")
    
    if ensure_scipy and not _HAS_SCIPY:
        raise ImportError("scipy is required for quaternion_to_rotation_matrix with ensure_scipy=True")
    
    if _HAS_SCIPY:
        # Use scipy (default, most accurate)
        return Rotation.from_quat(q).as_matrix()  # scipy expects xyzw
    else:
        # Numpy-only implementation
        return _quaternion_to_rotation_matrix_numpy(q)


def euler_to_quaternion(
    euler_angles: Union[np.ndarray, Sequence[float]],
    sequence: str = 'xyz',
    degrees: bool = False
) -> np.ndarray:
    """Convert Euler angles to quaternion (xyzw order).
    
    Parameters
    ----------
    euler_angles : np.ndarray or sequence
        Euler angles (3 elements)
    sequence : str, optional
        Rotation sequence (e.g., 'xyz', 'zyx'), by default 'xyz'
    degrees : bool, optional
        If True, input is in degrees, by default False (radians)
    
    Returns
    -------
    np.ndarray
        Quaternion in xyzw order: [qx, qy, qz, qw]
    
    Raises
    ------
    ImportError
        If scipy is not available (required for this function)
    
    Examples
    --------
    >>> euler = np.array([0, 0, np.pi/2])  # 90 deg rotation around Z
    >>> q = euler_to_quaternion(euler, sequence='xyz')
    >>> print(q)
    [0.         0.         0.70710678 0.70710678]
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for euler_to_quaternion")
    
    euler = np.asarray(euler_angles, dtype=np.float64)
    return Rotation.from_euler(sequence, euler, degrees=degrees).as_quat()  # xyzw


def quaternion_to_euler(
    quaternion: Union[np.ndarray, Sequence[float]],
    sequence: str = 'xyz',
    degrees: bool = False
) -> np.ndarray:
    """Convert quaternion (xyzw order) to Euler angles.
    
    Parameters
    ----------
    quaternion : np.ndarray or sequence
        Quaternion in xyzw order: [qx, qy, qz, qw]
    sequence : str, optional
        Rotation sequence (e.g., 'xyz', 'zyx'), by default 'xyz'
    degrees : bool, optional
        If True, output is in degrees, by default False (radians)
    
    Returns
    -------
    np.ndarray
        Euler angles (3 elements)
    
    Raises
    ------
    ImportError
        If scipy is not available (required for this function)
    
    Examples
    --------
    >>> q = np.array([0, 0, 0.7071, 0.7071])  # 90 deg rotation around Z
    >>> euler = quaternion_to_euler(q, sequence='xyz')
    >>> print(euler)
    [0.         0.         1.57079633]
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for quaternion_to_euler")
    
    q = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(q).as_euler(sequence, degrees=degrees)  # scipy expects xyzw


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (xyzw order).
    
    Computes q1 * q2 (apply q1, then q2).
    
    Parameters
    ----------
    q1 : np.ndarray
        First quaternion in xyzw order: [qx, qy, qz, qw]
    q2 : np.ndarray
        Second quaternion in xyzw order: [qx, qy, qz, qw]
    
    Returns
    -------
    np.ndarray
        Result quaternion in xyzw order: [qx, qy, qz, qw]
    
    Examples
    --------
    >>> q1 = np.array([0, 0, 0, 1])  # Identity
    >>> q2 = np.array([0, 0, 0.7071, 0.7071])  # 90 deg around Z
    >>> q_result = quaternion_multiply(q1, q2)
    >>> print(q_result)
    [0.     0.     0.7071 0.7071]
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    
    # xyzw order: [qx, qy, qz, qw]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    # Hamilton product
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # qx
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # qy
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # qz
        w1*w2 - x1*x2 - y1*y2 - z1*z2   # qw
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate (xyzw order).
    
    For unit quaternions, this is the same as the inverse.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in xyzw order: [qx, qy, qz, qw]
    
    Returns
    -------
    np.ndarray
        Conjugate quaternion in xyzw order: [-qx, -qy, -qz, qw]
    
    Examples
    --------
    >>> q = np.array([0.1, 0.2, 0.3, 0.9])
    >>> q_conj = quaternion_conjugate(q)
    >>> print(q_conj)
    [-0.1 -0.2 -0.3  0.9]
    """
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (xyzw order).
    
    For unit quaternions, this is the same as conjugate.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in xyzw order: [qx, qy, qz, qw]
    
    Returns
    -------
    np.ndarray
        Inverse quaternion in xyzw order
    
    Examples
    --------
    >>> q = np.array([0, 0, 0.7071, 0.7071])
    >>> q_inv = quaternion_inverse(q)
    >>> # q * q_inv should give identity [0, 0, 0, 1]
    """
    q = np.asarray(q, dtype=np.float64)
    norm_sq = np.dot(q, q)
    return quaternion_conjugate(q) / norm_sq


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in xyzw order: [qx, qy, qz, qw]
    
    Returns
    -------
    np.ndarray
        Normalized quaternion in xyzw order
    
    Examples
    --------
    >>> q = np.array([1, 1, 1, 1])
    >>> q_norm = quaternion_normalize(q)
    >>> print(q_norm)
    [0.5 0.5 0.5 0.5]
    >>> print(np.linalg.norm(q_norm))
    1.0
    """
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# ============================================================================
# NumPy-only implementations (fallback when scipy is not available)
# ============================================================================

def _rotation_matrix_to_quaternion_numpy(R: np.ndarray) -> np.ndarray:
    """NumPy-only rotation matrix to quaternion conversion (Shepperd's method).
    
    Returns quaternion in xyzw order.
    """
    # Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # Return in xyzw order
    return np.array([qx, qy, qz, qw])


def _quaternion_to_rotation_matrix_numpy(q: np.ndarray) -> np.ndarray:
    """NumPy-only quaternion to rotation matrix conversion.
    
    Expects quaternion in xyzw order.
    """
    # Normalize first
    q = q / np.linalg.norm(q)
    
    qx, qy, qz, qw = q
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


# ============================================================================
# Convenience functions
# ============================================================================

def identity_quaternion() -> np.ndarray:
    """Return identity quaternion (xyzw order).
    
    Returns
    -------
    np.ndarray
        Identity quaternion [0, 0, 0, 1] in xyzw order
    
    Examples
    --------
    >>> q = identity_quaternion()
    >>> print(q)
    [0. 0. 0. 1.]
    """
    return np.array([0.0, 0.0, 0.0, 1.0])


def is_valid_quaternion(q: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if quaternion is valid (unit length, 4 elements).
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion to check
    tol : float, optional
        Tolerance for unit length check, by default 1e-6
    
    Returns
    -------
    bool
        True if valid quaternion
    
    Examples
    --------
    >>> q = np.array([0, 0, 0, 1])
    >>> print(is_valid_quaternion(q))
    True
    >>> q = np.array([1, 1, 1, 1])
    >>> print(is_valid_quaternion(q))
    False
    """
    q = np.asarray(q)
    if q.shape != (4,):
        return False
    norm = np.linalg.norm(q)
    return abs(norm - 1.0) < tol
