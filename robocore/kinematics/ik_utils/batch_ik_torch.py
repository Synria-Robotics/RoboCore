"""Batch Inverse Kinematics implementation for PyTorch.

This module provides true parallel/vectorized IK computation for batches of
target poses, leveraging PyTorch's automatic vectorization and GPU acceleration.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import torch

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


def batch_inverse_kinematics_torch(
    model: "RobotModel",
    target_poses_batch: torch.Tensor,
    q_init_batch: torch.Tensor | None = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    damping: float = 0.01,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve inverse kinematics for a batch of target poses in parallel using DLS.
    
    This is a true vectorized implementation that processes all samples
    simultaneously using PyTorch's batch operations.
    
    Parameters
    ----------
    model : RobotModel
        Robot model
    target_poses_batch : torch.Tensor
        Batch of target poses, shape [B, 4, 4] (homogeneous matrices)
    q_init_batch : torch.Tensor, optional
        Initial joint configurations, shape [B, N]. If None, uses zeros
    device : str
        PyTorch device ('cpu', 'cuda', 'mps')
    dtype : torch.dtype
        Data type for computation
    max_iterations : int
        Maximum iterations for each sample
    tolerance : float
        Convergence tolerance (position + orientation error)
    damping : float
        Damping factor for DLS (Damped Least Squares)
    verbose : bool
        Print convergence information
    
    Returns
    -------
    q_batch : torch.Tensor
        Solution joint configurations, shape [B, N]
    success_batch : torch.Tensor
        Success flags, shape [B] (bool)
    iterations_batch : torch.Tensor
        Number of iterations taken, shape [B] (int)
    
    Examples
    --------
    >>> import torch
    >>> from robocore.modeling.robot_model import RobotModel
    >>> 
    >>> model = RobotModel('path/to/robot.urdf')
    >>> target_poses = torch.randn(100, 4, 4)  # 100 target poses
    >>> q_sol, success, iters = batch_inverse_kinematics_torch(
    ...     model, target_poses, device='cuda'
    ... )
    >>> print(f"Success rate: {success.float().mean():.1%}")
    """
    from robocore.kinematics.fk_utils.batch_fk_torch import (
        batch_forward_kinematics_torch,
        extract_position_batch,
        extract_rotation_batch
    )
    from robocore.kinematics.jacobian_utils.batch_jacobian_torch import (
        batch_geometric_jacobian_torch
    )
    
    # Setup
    target_poses_batch = target_poses_batch.to(device=device, dtype=dtype)
    batch_size = target_poses_batch.shape[0]
    n_joints = model.dof()
    
    # Initialize joint configurations
    if q_init_batch is None:
        q_batch = torch.zeros(batch_size, n_joints, device=device, dtype=dtype)
    else:
        q_batch = q_init_batch.to(device=device, dtype=dtype).clone()
    
    # Extract target position and rotation
    target_pos = extract_position_batch(target_poses_batch)  # [B, 3]
    target_rot = extract_rotation_batch(target_poses_batch)  # [B, 3, 3]
    
    # Tracking
    success_batch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    iterations_batch = torch.zeros(batch_size, dtype=torch.long, device=device)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)  # Which samples still solving
    
    # Damping matrix
    damping_matrix = damping ** 2 * torch.eye(6, device=device, dtype=dtype)
    
    for iteration in range(max_iterations):
        if not active_mask.any():
            break  # All converged
        
        # Forward kinematics for active samples
        T_current = batch_forward_kinematics_torch(model, q_batch, device=device, dtype=dtype)
        current_pos = extract_position_batch(T_current)
        current_rot = extract_rotation_batch(T_current)
        
        # Compute position error [B, 3]
        pos_error = target_pos - current_pos
        
        # Compute orientation error using angle-axis [B, 3]
        ori_error = _batch_rotation_error(target_rot, current_rot)
        
        # Combined error [B, 6]
        error = torch.cat([pos_error, ori_error], dim=1)
        
        # Compute total error magnitude for each sample
        error_magnitude = torch.linalg.norm(error, dim=1)  # [B]
        
        # Check convergence for each sample
        converged = error_magnitude < tolerance
        newly_converged = converged & active_mask
        success_batch[newly_converged] = True
        iterations_batch[newly_converged] = iteration + 1
        active_mask[converged] = False
        
        if not active_mask.any():
            break
        
        # Compute Jacobian for all samples (including inactive for simplicity)
        J_batch = batch_geometric_jacobian_torch(model, q_batch, device=device, dtype=dtype)  # [B, 6, N]
        
        # Solve for joint update using DLS: dq = J^T (J J^T + λ²I)^(-1) error
        # For batch: [B, N, 6] @ [B, 6, 6]^(-1) @ [B, 6, 1] = [B, N, 1]
        
        # Compute J J^T + λ²I for each sample
        JJT = torch.matmul(J_batch, J_batch.transpose(1, 2))  # [B, 6, 6]
        JJT_damped = JJT + damping_matrix.unsqueeze(0)  # [B, 6, 6]
        
        # Solve: (J J^T + λ²I)^(-1) @ error
        try:
            error_6d = error.unsqueeze(2)  # [B, 6, 1]
            JJT_inv_error = torch.linalg.solve(JJT_damped, error_6d)  # [B, 6, 1]
            
            # dq = J^T @ (JJT)^(-1) @ error
            dq = torch.matmul(J_batch.transpose(1, 2), JJT_inv_error).squeeze(2)  # [B, N]
        except RuntimeError:
            # Singular matrix - use pseudoinverse fallback for problematic samples
            dq = torch.zeros(batch_size, n_joints, device=device, dtype=dtype)
            for b in range(batch_size):
                if active_mask[b]:
                    try:
                        J_pinv = torch.linalg.pinv(J_batch[b])  # [N, 6]
                        dq[b] = J_pinv @ error[b]
                    except RuntimeError:
                        active_mask[b] = False  # Give up on this sample
        
        # Update only active samples
        q_batch[active_mask] = q_batch[active_mask] + dq[active_mask]
        
        # Optional: Clamp joint limits here if needed
        # q_batch = torch.clamp(q_batch, min=q_min, max=q_max)
    
    # Mark remaining active samples as failed
    iterations_batch[active_mask] = max_iterations
    
    if verbose:
        success_rate = success_batch.float().mean().item()
        avg_iters = iterations_batch[success_batch].float().mean().item() if success_batch.any() else 0
        print(f"Batch IK: {success_rate:.1%} success, avg {avg_iters:.1f} iterations")
    
    return q_batch, success_batch, iterations_batch


def _batch_rotation_error(R_target: torch.Tensor, R_current: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation error in angle-axis representation.
    
    Error = log(R_target @ R_current^T) converted to angle-axis
    
    Parameters
    ----------
    R_target : torch.Tensor
        Target rotation matrices, shape [B, 3, 3]
    R_current : torch.Tensor
        Current rotation matrices, shape [B, 3, 3]
    
    Returns
    -------
    torch.Tensor
        Rotation error vectors, shape [B, 3]
    """
    # R_error = R_target @ R_current^T
    R_error = torch.matmul(R_target, R_current.transpose(1, 2))
    
    # Convert to angle-axis
    return _batch_rotation_matrix_to_axis_angle(R_error)


def _batch_rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation matrices to axis-angle representation.
    
    Parameters
    ----------
    R : torch.Tensor
        Rotation matrices, shape [B, 3, 3]
    
    Returns
    -------
    torch.Tensor
        Axis-angle vectors, shape [B, 3]
    """
    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype
    
    # Angle from trace: cos(θ) = (trace(R) - 1) / 2
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # [B]
    
    # Axis from skew-symmetric part
    axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    
    # Handle small angles (θ ≈ 0) - use linear approximation
    small_angle = angle < 1e-6
    if small_angle.any():
        # For small angles: axis-angle ≈ [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / 2
        axis[small_angle, 0] = (R[small_angle, 2, 1] - R[small_angle, 1, 2]) / 2
        axis[small_angle, 1] = (R[small_angle, 0, 2] - R[small_angle, 2, 0]) / 2
        axis[small_angle, 2] = (R[small_angle, 1, 0] - R[small_angle, 0, 1]) / 2
    
    # Handle normal angles
    normal_angle = ~small_angle
    if normal_angle.any():
        # axis = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / (2*sin(θ))
        sin_angle = torch.sin(angle[normal_angle])
        axis[normal_angle, 0] = (R[normal_angle, 2, 1] - R[normal_angle, 1, 2]) / (2 * sin_angle)
        axis[normal_angle, 1] = (R[normal_angle, 0, 2] - R[normal_angle, 2, 0]) / (2 * sin_angle)
        axis[normal_angle, 2] = (R[normal_angle, 1, 0] - R[normal_angle, 0, 1]) / (2 * sin_angle)
    
    # Axis-angle = angle * axis
    axis_angle = axis * angle.unsqueeze(1)
    
    return axis_angle
