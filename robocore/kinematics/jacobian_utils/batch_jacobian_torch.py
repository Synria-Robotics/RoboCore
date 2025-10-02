"""Batch Jacobian computation for PyTorch.

This module provides true parallel/vectorized Jacobian computation for batches of
joint configurations, leveraging PyTorch's automatic vectorization and GPU acceleration.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


def batch_geometric_jacobian_torch(
    model: "RobotModel",
    q_batch: torch.Tensor,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute geometric Jacobian for a batch of joint configurations in parallel.
    
    The geometric Jacobian relates joint velocities to end-effector twist:
        [v; ω] = J(q) * dq/dt
    where v is linear velocity, ω is angular velocity.
    
    Parameters
    ----------
    model : RobotModel
        Robot model
    q_batch : torch.Tensor
        Batch of joint configurations, shape [B, N]
    device : str
        PyTorch device ('cpu', 'cuda', 'mps')
    dtype : torch.dtype
        Data type for computation
    
    Returns
    -------
    torch.Tensor
        Batch of Jacobian matrices, shape [B, 6, N]
        First 3 rows: linear velocity
        Last 3 rows: angular velocity
    
    Examples
    --------
    >>> import torch
    >>> from robocore.modeling.robot_model import RobotModel
    >>> 
    >>> model = RobotModel('path/to/robot.urdf')
    >>> q_batch = torch.randn(100, 6)  # 100 samples, 6 DOF
    >>> J_batch = batch_geometric_jacobian_torch(model, q_batch, device='cuda')
    >>> print(J_batch.shape)  # [100, 6, 6]
    """
    from robocore.kinematics.fk_utils.batch_fk_torch import (
        batch_forward_kinematics_torch,
        extract_position_batch
    )
    
    q_batch = q_batch.to(device=device, dtype=dtype)
    batch_size = q_batch.shape[0]
    n_joints = q_batch.shape[1]
    
    if n_joints != model.dof():
        raise ValueError(f"Expected {model.dof()} joints, got {n_joints}")
    
    # Initialize Jacobian [B, 6, N]
    J_batch = torch.zeros(batch_size, 6, n_joints, device=device, dtype=dtype)
    
    # Get end-effector position for all samples [B, 3]
    T_ee_batch = batch_forward_kinematics_torch(model, q_batch, device=device, dtype=dtype)
    p_ee_batch = extract_position_batch(T_ee_batch)
    
    # Process each joint
    joint_specs = model._actuated
    
    for js in joint_specs:
        joint_idx = js.index
        
        # Compute FK up to this joint for all samples
        # We need position and z-axis of this joint
        T_joint_batch = _batch_forward_kinematics_to_joint_torch(
            model, q_batch, joint_idx, device=device, dtype=dtype
        )
        
        # Extract position and z-axis
        p_joint_batch = extract_position_batch(T_joint_batch)  # [B, 3]
        z_axis_batch = T_joint_batch[:, :3, 2]  # [B, 3] - third column of rotation matrix
        
        # For revolute joint:
        # J_v[i] = z[i] × (p_ee - p[i])  (linear velocity contribution)
        # J_ω[i] = z[i]                  (angular velocity contribution)
        
        # Linear part: cross product z × (p_ee - p_joint)
        r = p_ee_batch - p_joint_batch  # [B, 3]
        J_linear = torch.cross(z_axis_batch, r, dim=1)  # [B, 3]
        
        # Angular part: just the z-axis
        J_angular = z_axis_batch  # [B, 3]
        
        # Assign to Jacobian
        J_batch[:, 0:3, joint_idx] = J_linear
        J_batch[:, 3:6, joint_idx] = J_angular
    
    return J_batch


def _batch_forward_kinematics_to_joint_torch(
    model: "RobotModel",
    q_batch: torch.Tensor,
    joint_idx: int,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute FK up to (and including) a specific joint for a batch.
    
    Parameters
    ----------
    model : RobotModel
        Robot model
    q_batch : torch.Tensor
        Batch of joint configurations, shape [B, N]
    joint_idx : int
        Index of target joint (0-indexed)
    device : str
        PyTorch device
    dtype : torch.dtype
        Data type
    
    Returns
    -------
    torch.Tensor
        Transformation matrices up to joint, shape [B, 4, 4]
    """
    batch_size = q_batch.shape[0]
    
    # Initialize batch of identity matrices
    T_batch = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Get joints up to and including target
    joint_specs = model._actuated[:joint_idx + 1]
    
    # Process each joint (same as batch_forward_kinematics_torch but truncated)
    for js in joint_specs:
        theta = q_batch[:, js.index]
        
        # Build transformation matrix for this joint
        origin_xyz = torch.tensor(js.origin_xyz, device=device, dtype=dtype)
        origin_rpy = torch.tensor(js.origin_rpy, device=device, dtype=dtype)
        axis = torch.tensor(js.axis, device=device, dtype=dtype)
        
        # Rotation matrices
        from robocore.kinematics.fk_utils.batch_fk_torch import (
            _rpy_to_rotation_matrix_batch,
            _axis_angle_to_rotation_matrix_batch
        )
        
        R_origin = _rpy_to_rotation_matrix_batch(
            origin_rpy.unsqueeze(0).repeat(batch_size, 1)
        )
        R_joint = _axis_angle_to_rotation_matrix_batch(axis, theta)
        R_total = torch.matmul(R_origin, R_joint)
        
        # Build 4x4 matrix
        T_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        T_joint[:, :3, :3] = R_total
        T_joint[:, :3, 3] = origin_xyz
        
        # Accumulate
        T_batch = torch.matmul(T_batch, T_joint)
    
    return T_batch
