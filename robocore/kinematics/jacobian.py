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


def jacobian(
    model: Any,
    q: Sequence[float] | Any,
    *,
    method: str = 'analytic',
    # Local / partial options
    target_link: Optional[str] = None,
    base_link: Optional[str] = None,
    end_link: Optional[str] = None,
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
    - Single: q [n] -> returns [6, n] matrix (or [6, nq_full] if base_link/end_link specified)
    - Batch: q [B, n] -> returns [B, 6, n] array (or [B, 6, nq_full] if base_link/end_link specified)
    
    :param model: RobotModel instance
    :param q: Joint configuration(s) - [n] or [B, n] array, or full config [num_dof] if base_link/end_link specified
    :param method: 'analytic'|'numeric'|'autograd'
    :param target_link: Target link name (legacy, use end_link instead)
    :param base_link: Base link name (if specified, uses dynamic path and returns full config space Jacobian)
    :param end_link: End link name (if specified, uses dynamic path and returns full config space Jacobian)
    :param joint_indices: Selected joint indices
    :param row_mask: Row selection mask (len=6)
    :param epsilon: Finite-difference step (numeric)
    :param use_central_diff: Use central difference (numeric)
    :param device: Torch device when using torch backend (uses global backend setting)
    :param dtype: Torch dtype when using torch backend (uses global backend setting)
    :return: 
        - Single mode: [6, n] or [6, nq_full] Jacobian matrix
        - Batch mode: [B, 6, n] or [B, 6, nq_full] Jacobian array
    """
    # Handle dynamic base_link/end_link
    use_full_config = False
    chain_indices = None
    original_end = None
    original_base = None
    
    # Check if q is full configuration (even without explicit base_link/end_link)
    q_arr = np.array(q)
    q_flat_len = len(q_arr.flatten()) if q_arr.ndim <= 2 else q_arr.shape[-1]
    
    if base_link is not None or end_link is not None:
        # Dynamic path: extract chain joint values from full configuration
        base = base_link or model.base_link
        end = end_link or (target_link or model.end_link)
        
        # Check if q is full configuration
        if hasattr(model, 'num_dof') and q_flat_len == model.num_dof:
            use_full_config = True
            chain_indices = model._get_joint_indices(base, end)
            q_chain = q_arr[chain_indices] if q_arr.ndim == 1 else q_arr[:, chain_indices]
            q = q_chain.tolist() if hasattr(q_chain, 'tolist') else list(q_chain)
        
        # Temporarily update model's end_link for jacobian computation
        original_end = model.end_link
        original_base = model.base_link
        try:
            model.end_link = end
            model.base_link = base
            # Rebuild chain for this path
            model._build_chain()
        except Exception:
            model.end_link = original_end
            model.base_link = original_base
            raise
    else:
        # No dynamic path specified, but q might be full config
        # If q length matches num_dof, extract chain config and mark for full config expansion
        if hasattr(model, 'num_dof') and q_flat_len == model.num_dof:
            use_full_config = True
            # Extract chain joint values from full configuration
            chain_indices = model._get_joint_indices(model.base_link, model.end_link)
            q_chain = q_arr[chain_indices] if q_arr.ndim == 1 else q_arr[:, chain_indices]
            q = q_chain.tolist() if hasattr(q_chain, 'tolist') else list(q_chain)
    
    b = get_backend()
    method = method.lower()

    if b == 'numpy':
        # NumPy backend
        solver = JacobianSolverNumPy(model)
        if method not in ('analytic', 'numeric'):
            if method == 'autograd':
                raise ValueError("Autograd method requires torch backend")
            else:
                raise ValueError(f"Unknown method '{method}' for numpy backend. Use 'analytic' or 'numeric'.")

        J = solver.solve(
            q,
            method=method,
            epsilon=epsilon,
            use_central_diff=use_central_diff,
            target_link=target_link,
        )

        # Apply row mask (rows) if provided
        if row_mask is not None:
            mask_bool: List[bool] = [bool(m) for m in row_mask]
            if len(mask_bool) != 6:
                raise ValueError("row_mask must have length 6 (for 6 twist components)")
            if J.ndim == 3:
                J = J[:, mask_bool, :]
            else:
                J = J[mask_bool, :]

        # Apply joint (column) selection if provided
        if joint_indices is not None:
            if J.ndim == 3:
                J = J[:, :, list(joint_indices)]
            else:
                J = J[:, list(joint_indices)]

        # Expand to full configuration space if dynamic path was used
        if use_full_config and chain_indices is not None:
            nq_full = model.num_dof
            if J.ndim == 3:
                # Batch mode: [B, 6, n_chain] -> [B, 6, nq_full]
                batch_size = J.shape[0]
                J_full = np.zeros((batch_size, 6, nq_full), dtype=J.dtype)
                J_full[:, :, chain_indices] = J
                J = J_full
            else:
                # Single mode: [6, n_chain] -> [6, nq_full]
                J_full = np.zeros((6, nq_full), dtype=J.dtype)
                J_full[:, chain_indices] = J
                J = J_full
        
        # Restore original end_link and base_link if we modified them
        if original_end is not None:
            model.end_link = original_end
            model.base_link = original_base
            model._build_chain()

        return J
    elif b == 'torch':
        import torch
        from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch

        solver = JacobianSolverTorch(model)

        if dtype is None:
            dtype = torch.float64

        # Convert numpy to torch if needed
        if isinstance(q, np.ndarray):
            q_torch = torch.from_numpy(q).to(dtype=dtype, device=device)
        else:
            q_torch = q

        if method not in ('analytic', 'numeric', 'autograd'):
            raise ValueError(f"Unknown method '{method}'. Use 'analytic', 'numeric', or 'autograd'.")

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
            if J.ndim == 3:
                J = J[:, mask_bool, :]
            else:
                J = J[mask_bool, :]

        # Apply joint (column) selection if provided
        if joint_indices is not None:
            joint_tensor = torch.tensor(list(joint_indices), dtype=torch.long)
            if J.ndim == 3:
                J = J[:, :, joint_tensor]
            else:
                J = J[:, joint_tensor]

        # Expand to full configuration space if dynamic path was used
        if use_full_config and chain_indices is not None:
            nq_full = model.num_dof
            if J.ndim == 3:
                # Batch mode: [B, 6, n_chain] -> [B, 6, nq_full]
                batch_size = J.shape[0]
                J_full = torch.zeros((batch_size, 6, nq_full), dtype=J.dtype, device=J.device)
                J_full[:, :, chain_indices] = J
                J = J_full
            else:
                # Single mode: [6, n_chain] -> [6, nq_full]
                J_full = torch.zeros((6, nq_full), dtype=J.dtype, device=J.device)
                J_full[:, chain_indices] = J
                J = J_full
        
        # Restore original end_link and base_link if we modified them
        if original_end is not None:
            model.end_link = original_end
            model.base_link = original_base
            model._build_chain()

        # Convert to numpy for consistency
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        return J
    elif b == 'cpp':
        from robocore.kinematics.jacobian_utils.jacobian_solver_cpp import JacobianSolverCpp

        try:
            import torch
            if isinstance(q, torch.Tensor):
                q_in = q.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                q_in = np.asarray(q, dtype=np.float64)
        except ImportError:
            q_in = np.asarray(q, dtype=np.float64)

        if method == 'autograd':
            raise ValueError("Autograd method requires torch backend")
        if method not in ('analytic', 'numeric'):
            raise ValueError(f"Unknown method '{method}' for cpp backend. Use 'analytic' or 'numeric'.")

        solver_cpp = JacobianSolverCpp(model)
        # Solver is built for the current model chain (including dynamic end_link);
        # passing target_link into solve would require the same target_link in __init__.
        J = solver_cpp.solve(
            q_in,
            method=method,
            epsilon=epsilon,
            use_central_diff=use_central_diff,
            target_link=None,
        )

        if row_mask is not None:
            mask_bool: List[bool] = [bool(m) for m in row_mask]
            if len(mask_bool) != 6:
                raise ValueError("row_mask must have length 6 (for 6 twist components)")
            if J.ndim == 3:
                J = J[:, mask_bool, :]
            else:
                J = J[mask_bool, :]

        if joint_indices is not None:
            if J.ndim == 3:
                J = J[:, :, list(joint_indices)]
            else:
                J = J[:, list(joint_indices)]

        if use_full_config and chain_indices is not None:
            nq_full = model.num_dof
            if J.ndim == 3:
                batch_size = J.shape[0]
                J_full = np.zeros((batch_size, 6, nq_full), dtype=J.dtype)
                J_full[:, :, chain_indices] = J
                J = J_full
            else:
                J_full = np.zeros((6, nq_full), dtype=J.dtype)
                J_full[:, chain_indices] = J
                J = J_full

        if original_end is not None:
            model.end_link = original_end
            model.base_link = original_base
            model._build_chain()

        return J
    else:
        raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")


__all__ = ["jacobian"]
