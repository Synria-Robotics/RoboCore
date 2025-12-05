"""Unified Jacobian computation interface.

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

from typing import Any, Iterable, List, Optional, Sequence, Union
import numpy as np

from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from robocore.utils.backend import get_backend


def _normalize_q_input(q: Any) -> tuple[Any, bool]:
    """Normalize joint configuration input for batch processing.
    
    :param q: Joint configuration (1D array, 2D array)
    :return: Tuple of (normalized_q, is_batch)
    """
    q_arr = np.asarray(q)
    
    if q_arr.ndim == 1:
        # Single configuration: [n]
        return q_arr, False
    elif q_arr.ndim == 2:
        # Batch configuration: [B, n]
        return q_arr, True
    else:
        raise ValueError(f"Expected 1D or 2D array, got {q_arr.ndim}D array with shape {q_arr.shape}")


def jacobian(
    model: Any,
    q: Sequence[float] | Any,
    *,
    method: str = 'analytic',
    # Local / partial options
    target_link: Optional[str] = None,
    joint_indices: Optional[Sequence[int]] = None,
    row_mask: Optional[Sequence[int | bool]] = None,
    # Numeric Jacobian options
    epsilon: float = 5e-5,
    use_central_diff: bool = True,
    # Torch-specific options
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[np.ndarray, Any]:
    """Compute Jacobian matrix for single or batch of joint configurations.
    
    Supports both single and batch processing:
    - Single: q [n] -> returns [6, n] matrix
    - Batch: q [B, n] -> returns [B, 6, n] array
    
    :param model: RobotModel instance
    :param q: Joint configuration(s) - [n] or [B, n] array
    :param method: 'analytic'|'numeric'|'autograd'
    :param target_link: Target link name
    :param joint_indices: Selected joint indices
    :param row_mask: Row selection mask (len=6)
    :param epsilon: Finite-difference step (numeric)
    :param use_central_diff: Use central difference (numeric)
    :param device: Torch device when using torch backend (uses global backend setting)
    :param dtype: Torch dtype when using torch backend (uses global backend setting)
    :return: 
        - Single mode: [6, n] Jacobian matrix
        - Batch mode: [B, 6, n] Jacobian array
    """
    b = get_backend()
    method = method.lower()

    # Normalize input for batch processing
    q_normalized, is_batch = _normalize_q_input(q)

    if b == 'numpy':
        # NumPy backend
        solver = JacobianSolverNumPy(model)
        if method in ('analytic', 'numeric'):
            if is_batch:
                # Batch mode
                J_batch = solver.solve_batch(
                    q_normalized,
                    method=method,
                    epsilon=epsilon,
                    use_central_diff=use_central_diff,
                    target_link=target_link,
                )
            else:
                # Single mode
                J_batch = solver.solve(
                    q_normalized,
                    method=method,
                    epsilon=epsilon,
                    use_central_diff=use_central_diff,
                    target_link=target_link,
                )
                J_batch = J_batch[np.newaxis, :, :]  # Add batch dimension
            
            # Apply row mask (rows) if provided
            if row_mask is not None:
                mask_bool: List[bool] = [bool(m) for m in row_mask]
                if len(mask_bool) != 6:
                    raise ValueError("row_mask must have length 6 (for 6 twist components)")
                J_batch = J_batch[:, mask_bool, :] if is_batch else J_batch[mask_bool, :]
            
            # Apply joint (column) selection if provided
            if joint_indices is not None:
                if is_batch:
                    J_batch = J_batch[:, :, list(joint_indices)]
                else:
                    J_batch = J_batch[:, list(joint_indices)]
            
            # Remove batch dimension for single mode
            if not is_batch:
                J_batch = J_batch[0]
            
            return J_batch
        elif method == 'autograd':
            raise ValueError("Autograd method requires torch backend")
        else:
            raise ValueError(f"Unknown method '{method}' for numpy backend. Use 'analytic' or 'numeric'.")
    elif b == 'torch':
        import torch
        from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch

        solver = JacobianSolverTorch(model)

        if dtype is None:
            dtype = torch.float64

        # Convert numpy to torch if needed
        if isinstance(q_normalized, np.ndarray):
            q_torch = torch.from_numpy(q_normalized).to(dtype=dtype, device=device)
        else:
            q_torch = q_normalized

        if method in ('analytic', 'numeric', 'autograd'):
            J = solver.solve(
                q_torch,
                method=method,
                epsilon=epsilon,
                use_central_diff=use_central_diff,
                device=device,
                dtype=dtype,
                target_link=target_link,
            )
            
            # Apply row mask (rows) if provided
            if row_mask is not None:
                mask_bool: List[bool] = [bool(m) for m in row_mask]
                if len(mask_bool) != 6:
                    raise ValueError("row_mask must have length 6")
                if is_batch:
                    J = J[:, mask_bool, :]
                else:
                    J = J[mask_bool, :]
            
            # Apply joint (column) selection if provided
            if joint_indices is not None:
                joint_tensor = torch.tensor(list(joint_indices), dtype=torch.long)
                if is_batch:
                    J = J[:, :, joint_tensor]
                else:
                    J = J[:, joint_tensor]
            
            # Convert to numpy for consistency
            if isinstance(J, torch.Tensor):
                J = J.detach().cpu().numpy()
            
            return J
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'analytic', 'numeric', or 'autograd'.")
    else:
        raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")


__all__ = ["jacobian"]
