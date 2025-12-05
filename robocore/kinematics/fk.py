"""Unified Forward Kinematics Interface

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

from __future__ import annotations

from typing import Any, Dict, Sequence, Union
import numpy as np

from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy
from robocore.utils.backend import get_backend


def _normalize_q_input(q: Any) -> tuple[Any, bool]:
    """Normalize joint configuration input for batch processing.
    
    :param q: Joint configuration (1D array, 2D array, or dict)
    :return: Tuple of (normalized_q, is_batch)
        - normalized_q: Input normalized for batch processing
        - is_batch: True if input is batch (2D), False if single (1D)
    """
    # Handle dict input (backward compatibility, not batchable)
    if isinstance(q, dict):
        return q, False

    # Convert to numpy array for shape detection
    q_arr = np.asarray(q)

    # Check dimensions
    if q_arr.ndim == 1:
        # Single configuration: [n] -> wrap to [[n]]
        return q_arr, False
    elif q_arr.ndim == 2:
        # Batch configuration: [B, n]
        return q_arr, True
    else:
        raise ValueError(f"Expected 1D or 2D array, got {q_arr.ndim}D array with shape {q_arr.shape}")


def forward_kinematics(
    model: Any,
    q: Sequence[float] | Any,
    *,
    return_end: bool = False,
    return_all_links: bool = False,
    link_names: Sequence[str] | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Dict[str, Any], Any, np.ndarray]:
    """Compute forward kinematics for single chain or multiple chains.
    
    Supports both single and batch processing. Input is automatically normalized:
    - Single: [n] -> automatically wrapped to batch=1
    - Batch: [[n], [n], ...] -> processed in parallel
    
    :param model: RobotModel instance
    :param q: Joint configuration (array-like or dict {joint_name: value})
        - Single: [n] or (n,) -> returns single 4x4 matrix
        - Batch: [[n], [n], ...] or (B, n) -> returns [B, 4, 4] array
    :param return_end: Return only end-effector pose (legacy, conflicts with return_all_links)
    :param return_all_links: Return FK for all links in the kinematic tree (uses multi-chain solver)
    :param link_names: Specific links to compute FK for (only with return_all_links=True)
    :param device: Torch device when using torch backend (uses global backend setting)
    :param dtype: Torch dtype when using torch backend (uses global backend setting)
    :return: 
        - Single mode: 4x4 pose matrix (if return_end=True) or dict {link_name: 4x4 matrix}
        - Batch mode: [B, 4, 4] array (if return_end=True) or dict {link_name: [B, 4, 4] array}
    """
    b = get_backend()

    # Multi-chain FK path (doesn't support batch yet)
    if return_all_links:
        if return_end:
            raise ValueError("Cannot specify both return_end and return_all_links")

        if b == 'numpy':
            solver = FKSolverNumPy(model)
            return solver.solve_multi_chain(q, link_names=link_names)
        elif b == 'torch':
            import torch
            from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch

            solver = FKSolverTorch(model)
            if dtype is None:
                dtype = torch.float64
            return solver.solve_multi_chain(q, link_names=link_names, device=device, dtype=dtype)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")

    # Normalize input for batch processing
    q_normalized, is_batch = _normalize_q_input(q)

    # Single chain FK path with batch support
    if b == 'numpy':
        solver = FKSolverNumPy(model)
        if is_batch:
            # Batch mode
            if return_end:
                T_batch = solver.solve_batch(q_normalized)
                return T_batch
            else:
                # For non-end mode, return dict with batch arrays
                results = {}
                for i, q_single in enumerate(q_normalized):
                    poses = solver.solve(q_single, return_end_only=False)
                    if i == 0:
                        # Initialize dict structure
                        for link_name in poses.keys():
                            results[link_name] = []
                    for link_name, T in poses.items():
                        results[link_name].append(T)
                # Stack into arrays
                return {link_name: np.stack(arrays, axis=0) for link_name, arrays in results.items()}
        else:
            # Single mode (backward compatible)
            poses = solver.solve(q_normalized, return_end_only=return_end)
            return poses['end'] if return_end else poses
    elif b == 'torch':
        import torch
        from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
        
        solver = FKSolverTorch(model)
        if dtype is None:
            dtype = torch.float64

        # Convert numpy to torch if needed
        if isinstance(q_normalized, np.ndarray):
            q_torch = torch.from_numpy(q_normalized).to(dtype=dtype, device=device)
        else:
            q_torch = q_normalized

        if is_batch:
            # Batch mode
            if return_end:
                # solve() returns [B, 4, 4] tensor directly in batch mode
                poses = solver.solve(q_torch, return_end_only=True, device=device, dtype=dtype)
                return poses.detach().cpu().numpy() if isinstance(poses, torch.Tensor) else poses
            else:
                # For non-end mode in batch, need to process each configuration
                # This is less efficient but maintains API consistency
                batch_size = q_torch.shape[0]
                results = {}
                for i in range(batch_size):
                    poses_single = solver.solve(q_torch[i], return_end_only=False, device=device, dtype=dtype)
                    if i == 0:
                        # Initialize dict structure
                        for link_name in poses_single.keys():
                            results[link_name] = []
                    for link_name, T in poses_single.items():
                        results[link_name].append(T.detach().cpu().numpy() if isinstance(T, torch.Tensor) else T)
                # Stack into arrays
                return {link_name: np.stack(arrays, axis=0) for link_name, arrays in results.items()}
        else:
            # Single mode (backward compatible)
            poses = solver.solve(q_torch, return_end_only=return_end, device=device, dtype=dtype)
            if return_end:
                result = poses['end'] if isinstance(poses, dict) else poses
                return result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else result
            else:
                return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in poses.items()}
    else:
        raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")


__all__ = ["forward_kinematics"]
