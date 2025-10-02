"""Unified coordinate and rotation transformation utilities.

This module provides a comprehensive set of transformation tools supporting both
NumPy and PyTorch backends for robotics applications.

Features:
- Roll-Pitch-Yaw (RPY) to rotation matrix conversion
- Axis-angle rotation (Rodrigues' formula)
- Homogeneous transformation matrix construction
- Orientation error computation (axis-angle representation)
- Translation along axis
- Automatic backend selection

Usage:
    # NumPy backend
    R = rpy_to_rotation_matrix(0.1, 0.2, 0.3, backend='numpy')
    T = make_transform(R, [1, 2, 3], backend='numpy')
    
    # PyTorch backend
    R = rpy_to_rotation_matrix(0.1, 0.2, 0.3, backend='torch', device='cuda')
    err = orientation_error(R1, R2, backend='torch')
"""

from __future__ import annotations
from typing import Union, Optional, Sequence
import numpy as np

# Optional torch import
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore


# ============================================================================
# Backend Selection
# ============================================================================

def _select_backend(backend: str) -> str:
    """Select computation backend.
    
    :param backend: 'auto', 'numpy', or 'torch'.
    :return: selected backend name.
    """
    if backend == 'auto':
        return 'torch' if _HAS_TORCH else 'numpy'
    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unsupported backend '{backend}', expected 'auto'|'numpy'|'torch'")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError("Torch backend requested but PyTorch is not available")
    return backend


# ============================================================================
# NumPy Implementations
# ============================================================================

def rpy_to_rotation_matrix_numpy(
    roll: float,
    pitch: float,
    yaw: float
) -> np.ndarray:
    """Compute rotation matrix from roll-pitch-yaw angles (NumPy).
    
    Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    :param roll: rotation around x-axis (radians).
    :param pitch: rotation around y-axis (radians).
    :param yaw: rotation around z-axis (radians).
    :return: 3x3 rotation matrix.
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=np.float64)


def axis_angle_to_rotation_matrix_numpy(
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """Compute rotation matrix from axis-angle representation (NumPy).
    
    Uses Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    where [k]× is the skew-symmetric matrix of the unit axis.
    
    :param axis: rotation axis (3,), will be normalized.
    :param angle: rotation angle in radians.
    :return: 3x3 rotation matrix.
    """
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3, dtype=np.float64)
    
    # Normalize axis
    axis = axis / norm
    ax, ay, az = axis
    
    # Rodrigues' formula
    ct = np.cos(angle)
    st = np.sin(angle)
    vt = 1 - ct
    
    return np.array([
        [ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st],
        [ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st],
        [az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt]
    ], dtype=np.float64)


def axis_translation_numpy(
    axis: np.ndarray,
    distance: float
) -> np.ndarray:
    """Compute translation vector along an axis (NumPy).
    
    :param axis: direction axis (3,), will be normalized.
    :param distance: translation distance.
    :return: translation vector (3,).
    """
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return (axis / norm) * distance


def make_transform_numpy(
    rotation: np.ndarray,
    translation: Union[np.ndarray, Sequence[float]]
) -> np.ndarray:
    """Construct 4x4 homogeneous transformation matrix (NumPy).
    
    :param rotation: 3x3 rotation matrix.
    :param translation: 3D translation vector.
    :return: 4x4 transformation matrix [R t; 0 1].
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def orientation_error_numpy(
    R_current: np.ndarray,
    R_target: np.ndarray
) -> np.ndarray:
    """Compute orientation error from current to target rotation (NumPy).
    
    Returns axis-angle error: ω * θ where ω is the rotation axis (unit vector)
    and θ is the rotation angle needed to align R_current with R_target.
    
    Algorithm:
    1. Compute relative rotation: R_err = R_current^T @ R_target
    2. Extract angle: θ = arccos((trace(R_err) - 1) / 2)
    3. Extract axis from skew-symmetric part
    4. Return ω * θ
    
    :param R_current: current rotation matrix (3x3).
    :param R_target: target rotation matrix (3x3).
    :return: axis-angle error vector (3,).
    """
    # Compute relative rotation
    R_err = R_current.T @ R_target
    
    # Extract angle from trace
    trace = np.trace(R_err)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    
    # Zero rotation case
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    
    # π rotation singularity handling
    if angle > np.pi - 1e-6:
        # Use diagonal elements to find axis
        diag = np.diag(R_err)
        axis = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0))
        # Determine signs from off-diagonal elements
        if R_err[0, 1] + R_err[1, 0] < 0:
            axis[1] = -axis[1]
        if R_err[0, 2] + R_err[2, 0] < 0:
            axis[2] = -axis[2]
        axis = axis / (np.linalg.norm(axis) + 1e-15)
        return axis * angle
    
    # Extract axis from skew-symmetric part
    denom = 2.0 * np.sin(angle)
    wx = (R_err[2, 1] - R_err[1, 2]) / denom
    wy = (R_err[0, 2] - R_err[2, 0]) / denom
    wz = (R_err[1, 0] - R_err[0, 1]) / denom
    
    # Return axis * angle
    return np.array([wx * angle, wy * angle, wz * angle], dtype=np.float64)


# ============================================================================
# PyTorch Implementations
# ============================================================================

def rpy_to_rotation_matrix_torch(
    roll,
    pitch,
    yaw,
    device=None,
    dtype=None
):
    """Compute rotation matrix from roll-pitch-yaw angles (PyTorch).
    
    Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    :param roll: rotation around x-axis (radians), scalar tensor or float.
    :param pitch: rotation around y-axis (radians), scalar tensor or float.
    :param yaw: rotation around z-axis (radians), scalar tensor or float.
    :param device: torch device.
    :param dtype: torch dtype.
    :return: 3x3 rotation matrix tensor.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    # Convert to tensors if needed
    if not isinstance(roll, torch.Tensor):
        roll = torch.tensor(roll, device=device, dtype=dtype)
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(pitch, device=device, dtype=dtype)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, device=device, dtype=dtype)
    
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    
    return torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
        torch.stack([-sp, cp * sr, cp * cr]),
    ])


def axis_angle_to_rotation_matrix_torch(axis, angle, device=None, dtype=None):
    """Compute rotation matrix from axis-angle representation (PyTorch).
    
    Uses Rodrigues' formula.
    
    :param axis: rotation axis tensor (3,), will be normalized.
    :param angle: rotation angle in radians, scalar tensor or float.
    :param device: torch device.
    :param dtype: torch dtype.
    :return: 3x3 rotation matrix tensor.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, device=device, dtype=dtype)
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, device=device, dtype=dtype)
    
    norm = torch.linalg.norm(axis)
    if norm < 1e-12:
        return torch.eye(3, dtype=axis.dtype, device=axis.device)
    
    # Normalize axis
    axis = axis / norm
    ax, ay, az = axis
    
    # Rodrigues' formula
    ct = torch.cos(angle)
    st = torch.sin(angle)
    vt = 1 - ct
    
    return torch.stack([
        torch.stack([ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st]),
        torch.stack([ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st]),
        torch.stack([az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt]),
    ])


def axis_translation_torch(axis, distance, device=None, dtype=None):
    """Compute translation vector along an axis (PyTorch).
    
    :param axis: direction axis tensor (3,), will be normalized.
    :param distance: translation distance, scalar tensor or float.
    :param device: torch device.
    :param dtype: torch dtype.
    :return: translation vector tensor (3,).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, device=device, dtype=dtype)
    if not isinstance(distance, torch.Tensor):
        distance = torch.tensor(distance, device=device, dtype=dtype)
    
    norm = torch.linalg.norm(axis)
    if norm < 1e-12:
        return torch.zeros(3, dtype=axis.dtype, device=axis.device)
    return axis / norm * distance


def make_transform_torch(rotation, translation, device=None, dtype=None):
    """Construct 4x4 homogeneous transformation matrix (PyTorch).
    
    :param rotation: 3x3 rotation matrix tensor.
    :param translation: 3D translation vector tensor.
    :param device: torch device.
    :param dtype: torch dtype.
    :return: 4x4 transformation matrix tensor [R t; 0 1].
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.tensor(rotation, device=device, dtype=dtype)
    if not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, device=device, dtype=dtype)
    
    T = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def orientation_error_torch(R_current, R_target):
    """Compute orientation error from current to target rotation (PyTorch).
    
    Returns axis-angle error: ω * θ with robust handling of singularities.
    
    :param R_current: current rotation matrix tensor (3x3).
    :param R_target: target rotation matrix tensor (3x3).
    :return: axis-angle error vector tensor (3,).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    import math
    
    # Compute relative rotation
    R_err = R_current.transpose(0, 1) @ R_target
    
    # Extract angle with clamping
    trace_val = torch.clamp((torch.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(trace_val)
    
    # Zero rotation case
    if angle < 1e-12:
        return torch.zeros(3, dtype=R_current.dtype, device=R_current.device)
    
    # π rotation singularity handling
    if angle > math.pi - 1e-6:
        # Use diagonal to find axis
        axis = torch.stack([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        if torch.linalg.norm(axis) < 1e-8:
            diag = torch.diag(R_err)
            axis = torch.sqrt(torch.clamp(diag + 1.0, min=0))
        axis = axis / (torch.linalg.norm(axis) + 1e-15)
        return axis * angle
    
    # Extract axis from skew-symmetric part
    denom = 2.0 * torch.sin(angle)
    axis = torch.stack([
        (R_err[2, 1] - R_err[1, 2]) / (denom + 1e-15),
        (R_err[0, 2] - R_err[2, 0]) / (denom + 1e-15),
        (R_err[1, 0] - R_err[0, 1]) / (denom + 1e-15),
    ])
    return axis * angle


# ============================================================================
# Unified High-Level Interface
# ============================================================================

def rpy_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    backend: str = 'auto',
    device=None,
    dtype=None
):
    """Compute rotation matrix from roll-pitch-yaw angles.
    
    Unified interface supporting both NumPy and PyTorch backends.
    
    :param roll: rotation around x-axis (radians).
    :param pitch: rotation around y-axis (radians).
    :param yaw: rotation around z-axis (radians).
    :param backend: 'auto', 'numpy', or 'torch'.
    :param device: torch device (torch backend only).
    :param dtype: torch dtype (torch backend only).
    :return: 3x3 rotation matrix (numpy array or torch tensor).
    """
    b = _select_backend(backend)
    if b == 'numpy':
        return rpy_to_rotation_matrix_numpy(roll, pitch, yaw)
    else:
        return rpy_to_rotation_matrix_torch(roll, pitch, yaw, device=device, dtype=dtype)


def axis_angle_to_rotation_matrix(
    axis,
    angle: float,
    *,
    backend: str = 'auto',
    device=None,
    dtype=None
):
    """Compute rotation matrix from axis-angle representation.
    
    :param axis: rotation axis (3,), will be normalized.
    :param angle: rotation angle in radians.
    :param backend: 'auto', 'numpy', or 'torch'.
    :param device: torch device (torch backend only).
    :param dtype: torch dtype (torch backend only).
    :return: 3x3 rotation matrix.
    """
    b = _select_backend(backend)
    if b == 'numpy':
        axis_np = np.asarray(axis, dtype=np.float64)
        return axis_angle_to_rotation_matrix_numpy(axis_np, angle)
    else:
        return axis_angle_to_rotation_matrix_torch(axis, angle, device=device, dtype=dtype)


def axis_translation(
    axis,
    distance: float,
    *,
    backend: str = 'auto',
    device=None,
    dtype=None
):
    """Compute translation vector along an axis.
    
    :param axis: direction axis (3,), will be normalized.
    :param distance: translation distance.
    :param backend: 'auto', 'numpy', or 'torch'.
    :param device: torch device (torch backend only).
    :param dtype: torch dtype (torch backend only).
    :return: translation vector (3,).
    """
    b = _select_backend(backend)
    if b == 'numpy':
        axis_np = np.asarray(axis, dtype=np.float64)
        return axis_translation_numpy(axis_np, distance)
    else:
        return axis_translation_torch(axis, distance, device=device, dtype=dtype)


def make_transform(
    rotation,
    translation,
    *,
    backend: str = 'auto',
    device=None,
    dtype=None
):
    """Construct 4x4 homogeneous transformation matrix.
    
    :param rotation: 3x3 rotation matrix.
    :param translation: 3D translation vector.
    :param backend: 'auto', 'numpy', or 'torch'.
    :param device: torch device (torch backend only).
    :param dtype: torch dtype (torch backend only).
    :return: 4x4 transformation matrix [R t; 0 1].
    """
    b = _select_backend(backend)
    if b == 'numpy':
        rotation_np = np.asarray(rotation, dtype=np.float64)
        translation_np = np.asarray(translation, dtype=np.float64)
        return make_transform_numpy(rotation_np, translation_np)
    else:
        return make_transform_torch(rotation, translation, device=device, dtype=dtype)


def orientation_error(
    R_current,
    R_target,
    *,
    backend: str = 'auto'
):
    """Compute orientation error from current to target rotation.
    
    Returns axis-angle error representation.
    
    :param R_current: current rotation matrix (3x3).
    :param R_target: target rotation matrix (3x3).
    :param backend: 'auto', 'numpy', or 'torch'.
    :return: axis-angle error vector (3,).
    """
    b = _select_backend(backend)
    if b == 'numpy':
        R_current_np = np.asarray(R_current, dtype=np.float64)
        R_target_np = np.asarray(R_target, dtype=np.float64)
        return orientation_error_numpy(R_current_np, R_target_np)
    else:
        return orientation_error_torch(R_current, R_target)


__all__ = [
    # Unified high-level interface
    "rpy_to_rotation_matrix",
    "axis_angle_to_rotation_matrix",
    "axis_translation",
    "make_transform",
    "orientation_error",
    # NumPy backend (for direct use)
    "rpy_to_rotation_matrix_numpy",
    "axis_angle_to_rotation_matrix_numpy",
    "axis_translation_numpy",
    "make_transform_numpy",
    "orientation_error_numpy",
    # PyTorch backend (for direct use)
    "rpy_to_rotation_matrix_torch",
    "axis_angle_to_rotation_matrix_torch",
    "axis_translation_torch",
    "make_transform_torch",
    "orientation_error_torch",
]
