"""Batch Forward Kinematics implementation for PyTorch.

This module provides true parallel/vectorized FK computation for batches of
joint configurations, leveraging PyTorch's automatic vectorization and GPU
acceleration.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


def batch_forward_kinematics_torch(
    model: "RobotModel",
    q_batch: torch.Tensor,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute forward kinematics for a batch of joint configurations in parallel.
    
    This is a true vectorized implementation that processes all samples
    simultaneously using PyTorch's batch matrix operations.
    
    Parameters
    ----------
    model : RobotModel
        Robot model
    q_batch : torch.Tensor
        Batch of joint configurations, shape [B, N] where:
        - B is batch size
        - N is number of joints (DOF)
    device : str
        PyTorch device ('cpu', 'cuda', 'mps')
    dtype : torch.dtype
        Data type for computation
    
    Returns
    -------
    torch.Tensor
        Batch of homogeneous transformation matrices, shape [B, 4, 4]
        Each [i] is the end-effector pose for configuration q_batch[i]
    
    Examples
    --------
    >>> import torch
    >>> from robocore.modeling.robot_model import RobotModel
    >>> 
    >>> model = RobotModel('path/to/robot.urdf')
    >>> q_batch = torch.randn(100, 6)  # 100 samples, 6 DOF
    >>> T_batch = batch_forward_kinematics_torch(model, q_batch, device='cuda')
    >>> print(T_batch.shape)  # [100, 4, 4]
    """
    # Ensure tensor is on correct device with correct dtype
    q_batch = q_batch.to(device=device, dtype=dtype)
    batch_size = q_batch.shape[0]
    n_joints = q_batch.shape[1]
    
    if n_joints != model.dof():
        raise ValueError(f"Expected {model.dof()} joints, got {n_joints}")
    
    # Initialize batch of identity matrices [B, 4, 4]
    T_batch = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Get robot kinematic chain
    joint_specs = model._actuated
    
    # Process each joint in the chain (cannot fully vectorize due to sequential nature)
    # But we can vectorize across the batch dimension
    for js in joint_specs:
        # Get joint angle for this joint across all samples [B]
        theta = q_batch[:, js.index]
        
        # Build transformation matrix for this joint for all samples [B, 4, 4]
        # T_joint = Trans(origin_xyz) @ Rot(origin_rpy) @ Rot(axis, theta)
        
        # 1. Translation from origin
        origin_xyz = torch.tensor(js.origin_xyz, device=device, dtype=dtype)  # [3]
        
        # 2. Rotation from origin (roll-pitch-yaw)
        origin_rpy = torch.tensor(js.origin_rpy, device=device, dtype=dtype)  # [3]
        R_origin = _rpy_to_rotation_matrix_batch(
            origin_rpy.unsqueeze(0).repeat(batch_size, 1)
        )  # [B, 3, 3]
        
        # 3. Joint rotation (revolute around axis)
        axis = torch.tensor(js.axis, device=device, dtype=dtype)  # [3]
        R_joint = _axis_angle_to_rotation_matrix_batch(axis, theta)  # [B, 3, 3]
        
        # Combine rotations: R_total = R_origin @ R_joint
        R_total = torch.matmul(R_origin, R_joint)  # [B, 3, 3]
        
        # Build 4x4 transformation matrix for each sample
        T_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        T_joint[:, :3, :3] = R_total
        T_joint[:, :3, 3] = origin_xyz  # Same translation for all samples
        
        # Accumulate transformation: T_batch = T_batch @ T_joint (batch matmul)
        T_batch = torch.matmul(T_batch, T_joint)
    
    return T_batch


def _rpy_to_rotation_matrix_batch(rpy_batch: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of roll-pitch-yaw to rotation matrices.
    
    Parameters
    ----------
    rpy_batch : torch.Tensor
        Shape [B, 3] - roll, pitch, yaw for each sample
    
    Returns
    -------
    torch.Tensor
        Shape [B, 3, 3] - rotation matrices
    """
    roll = rpy_batch[:, 0]
    pitch = rpy_batch[:, 1]
    yaw = rpy_batch[:, 2]
    
    # Precompute sines and cosines
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    
    # Build rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    batch_size = rpy_batch.shape[0]
    R = torch.zeros(batch_size, 3, 3, device=rpy_batch.device, dtype=rpy_batch.dtype)
    
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


def _axis_angle_to_rotation_matrix_batch(axis: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of axis-angle rotations to rotation matrices using Rodrigues' formula.
    
    Parameters
    ----------
    axis : torch.Tensor
        Rotation axis, shape [3] (same for all samples)
    angles : torch.Tensor
        Rotation angles, shape [B]
    
    Returns
    -------
    torch.Tensor
        Shape [B, 3, 3] - rotation matrices
    """
    batch_size = angles.shape[0]
    device = angles.device
    dtype = angles.dtype
    
    # Normalize axis
    axis = axis / torch.linalg.norm(axis)
    
    # Rodrigues' formula: R = I + sin(θ)*K + (1-cos(θ))*K²
    # where K is the skew-symmetric matrix of the axis
    
    cos_theta = torch.cos(angles)  # [B]
    sin_theta = torch.sin(angles)  # [B]
    
    # Skew-symmetric matrix K
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    
    # K² = K @ K
    K2 = torch.matmul(K, K)
    
    # Build rotation matrices for batch
    I = torch.eye(3, device=device, dtype=dtype)
    R = I.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
    
    # R = I + sin(θ)*K + (1-cos(θ))*K²
    R = R + sin_theta.view(-1, 1, 1) * K.unsqueeze(0)
    R = R + (1 - cos_theta).view(-1, 1, 1) * K2.unsqueeze(0)
    
    return R


def extract_position_batch(T_batch: torch.Tensor) -> torch.Tensor:
    """Extract position vectors from batch of transformation matrices.
    
    Parameters
    ----------
    T_batch : torch.Tensor
        Shape [B, 4, 4]
    
    Returns
    -------
    torch.Tensor
        Shape [B, 3] - position vectors
    """
    return T_batch[:, :3, 3]


def extract_rotation_batch(T_batch: torch.Tensor) -> torch.Tensor:
    """Extract rotation matrices from batch of transformation matrices.
    
    Parameters
    ----------
    T_batch : torch.Tensor
        Shape [B, 4, 4]
    
    Returns
    -------
    torch.Tensor
        Shape [B, 3, 3] - rotation matrices
    """
    return T_batch[:, :3, :3]


def rotation_matrix_to_quaternion_batch(R_batch: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation matrices to quaternions (xyzw order).
    
    Parameters
    ----------
    R_batch : torch.Tensor
        Shape [B, 3, 3]
    
    Returns
    -------
    torch.Tensor
        Shape [B, 4] - quaternions in xyzw order
    """
    batch_size = R_batch.shape[0]
    device = R_batch.device
    dtype = R_batch.dtype
    
    # Using Shepperd's method for numerical stability
    trace = R_batch[:, 0, 0] + R_batch[:, 1, 1] + R_batch[:, 2, 2]
    
    quat = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = 0.5 / torch.sqrt(trace[mask1] + 1.0)
        quat[mask1, 3] = 0.25 / s  # qw
        quat[mask1, 0] = (R_batch[mask1, 2, 1] - R_batch[mask1, 1, 2]) * s  # qx
        quat[mask1, 1] = (R_batch[mask1, 0, 2] - R_batch[mask1, 2, 0]) * s  # qy
        quat[mask1, 2] = (R_batch[mask1, 1, 0] - R_batch[mask1, 0, 1]) * s  # qz
    
    # Case 2: R[0,0] is max diagonal
    mask2 = (~mask1) & (R_batch[:, 0, 0] > R_batch[:, 1, 1]) & (R_batch[:, 0, 0] > R_batch[:, 2, 2])
    if mask2.any():
        s = 2.0 * torch.sqrt(1.0 + R_batch[mask2, 0, 0] - R_batch[mask2, 1, 1] - R_batch[mask2, 2, 2])
        quat[mask2, 3] = (R_batch[mask2, 2, 1] - R_batch[mask2, 1, 2]) / s  # qw
        quat[mask2, 0] = 0.25 * s  # qx
        quat[mask2, 1] = (R_batch[mask2, 0, 1] + R_batch[mask2, 1, 0]) / s  # qy
        quat[mask2, 2] = (R_batch[mask2, 0, 2] + R_batch[mask2, 2, 0]) / s  # qz
    
    # Case 3: R[1,1] is max diagonal
    mask3 = (~mask1) & (~mask2) & (R_batch[:, 1, 1] > R_batch[:, 2, 2])
    if mask3.any():
        s = 2.0 * torch.sqrt(1.0 + R_batch[mask3, 1, 1] - R_batch[mask3, 0, 0] - R_batch[mask3, 2, 2])
        quat[mask3, 3] = (R_batch[mask3, 0, 2] - R_batch[mask3, 2, 0]) / s  # qw
        quat[mask3, 0] = (R_batch[mask3, 0, 1] + R_batch[mask3, 1, 0]) / s  # qx
        quat[mask3, 1] = 0.25 * s  # qy
        quat[mask3, 2] = (R_batch[mask3, 1, 2] + R_batch[mask3, 2, 1]) / s  # qz
    
    # Case 4: R[2,2] is max diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = 2.0 * torch.sqrt(1.0 + R_batch[mask4, 2, 2] - R_batch[mask4, 0, 0] - R_batch[mask4, 1, 1])
        quat[mask4, 3] = (R_batch[mask4, 1, 0] - R_batch[mask4, 0, 1]) / s  # qw
        quat[mask4, 0] = (R_batch[mask4, 0, 2] + R_batch[mask4, 2, 0]) / s  # qx
        quat[mask4, 1] = (R_batch[mask4, 1, 2] + R_batch[mask4, 2, 1]) / s  # qy
        quat[mask4, 2] = 0.25 * s  # qz
    
    return quat
