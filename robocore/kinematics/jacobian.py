"""Unified Jacobian computation interface.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union
import numpy as np

from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from robocore.utils.backend import get_backend


def _restore_model_chain(model: Any, original_base: Optional[str], original_end: Optional[str]) -> None:
    if original_end is not None:
        model.end_link = original_end
        model.base_link = original_base
        model._build_chain()


def _apply_row_and_joint_masks(J: Any, row_mask: Optional[Sequence[int | bool]], joint_indices: Optional[Sequence[int]]) -> Any:
    if row_mask is not None:
        mask_bool: List[bool] = [bool(m) for m in row_mask]
        if len(mask_bool) != 6:
            raise ValueError("row_mask must have length 6 (for 6 twist components)")
        J = J[:, mask_bool, :] if J.ndim == 3 else J[mask_bool, :]

    if joint_indices is not None:
        J = J[:, :, list(joint_indices)] if J.ndim == 3 else J[:, list(joint_indices)]

    return J


def _expand_numpy_jacobian(J: np.ndarray, nq_full: int, chain_indices: Sequence[int]) -> np.ndarray:
    if J.ndim == 3:
        J_full = np.zeros((J.shape[0], J.shape[1], nq_full), dtype=J.dtype)
        J_full[:, :, chain_indices] = J
        return J_full
    J_full = np.zeros((J.shape[0], nq_full), dtype=J.dtype)
    J_full[:, chain_indices] = J
    return J_full


def jacobian(
    model: Any,
    q: Sequence[float] | Any,
    *,
    method: str = 'analytic',
    target_link: Optional[str] = None,
    base_link: Optional[str] = None,
    end_link: Optional[str] = None,
    joint_indices: Optional[Sequence[int]] = None,
    row_mask: Optional[Sequence[int | bool]] = None,
    epsilon: float = 5e-5,
    use_central_diff: bool = True,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[np.ndarray, Any]:
    """Compute Jacobian matrix for single or batch joint configurations."""
    use_full_config = False
    chain_indices = None
    original_end = None
    original_base = None

    q_arr = np.array(q)
    q_flat_len = len(q_arr.flatten()) if q_arr.ndim <= 2 else q_arr.shape[-1]

    if base_link is not None or end_link is not None:
        base = base_link or model.base_link
        end = end_link or (target_link or model.end_link)

        if hasattr(model, 'num_dof') and q_flat_len == model.num_dof:
            use_full_config = True
            chain_indices = model._get_joint_indices(base, end)
            q_chain = q_arr[chain_indices] if q_arr.ndim == 1 else q_arr[:, chain_indices]
            q = q_chain.tolist() if hasattr(q_chain, 'tolist') else list(q_chain)

        original_end = model.end_link
        original_base = model.base_link
        model.end_link = end
        model.base_link = base
        model._build_chain()
    elif hasattr(model, 'num_dof') and q_flat_len == model.num_dof:
        use_full_config = True
        chain_indices = model._get_joint_indices(model.base_link, model.end_link)
        q_chain = q_arr[chain_indices] if q_arr.ndim == 1 else q_arr[:, chain_indices]
        q = q_chain.tolist() if hasattr(q_chain, 'tolist') else list(q_chain)

    b = get_backend()
    method = method.lower()

    try:
        if b == 'numpy':
            solver = JacobianSolverNumPy(model)
            if method == 'autograd':
                raise ValueError("Autograd method requires torch backend")
            if method not in ('analytic', 'numeric'):
                raise ValueError(f"Unknown method '{method}' for numpy backend. Use 'analytic' or 'numeric'.")

            J = solver.solve(
                q,
                method=method,
                epsilon=epsilon,
                use_central_diff=use_central_diff,
                target_link=target_link,
            )
            J = _apply_row_and_joint_masks(J, row_mask, joint_indices)
            if use_full_config and chain_indices is not None:
                J = _expand_numpy_jacobian(J, model.num_dof, chain_indices)
            return J

        if b == 'torch':
            import torch
            from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch

            if method not in ('analytic', 'numeric', 'autograd'):
                raise ValueError(f"Unknown method '{method}'. Use 'analytic', 'numeric', or 'autograd'.")
            if dtype is None:
                dtype = torch.float64

            q_torch = torch.from_numpy(q).to(dtype=dtype, device=device) if isinstance(q, np.ndarray) else q
            J = JacobianSolverTorch(model).solve(
                q_torch,
                method=method,
                epsilon=epsilon,
                use_central_diff=use_central_diff,
                device=device,
                dtype=dtype,
                target_link=target_link,
            )

            if row_mask is not None:
                mask_bool: List[bool] = [bool(m) for m in row_mask]
                if len(mask_bool) != 6:
                    raise ValueError("row_mask must have length 6 (for 6 twist components)")
                J = J[:, mask_bool, :] if J.ndim == 3 else J[mask_bool, :]

            if joint_indices is not None:
                joint_tensor = torch.tensor(list(joint_indices), dtype=torch.long, device=J.device)
                J = J[:, :, joint_tensor] if J.ndim == 3 else J[:, joint_tensor]

            if use_full_config and chain_indices is not None:
                if J.ndim == 3:
                    J_full = torch.zeros((J.shape[0], J.shape[1], model.num_dof), dtype=J.dtype, device=J.device)
                    J_full[:, :, chain_indices] = J
                    J = J_full
                else:
                    J_full = torch.zeros((J.shape[0], model.num_dof), dtype=J.dtype, device=J.device)
                    J_full[:, chain_indices] = J
                    J = J_full

            return J.detach().cpu().numpy() if isinstance(J, torch.Tensor) else J

        if b == 'cpp':
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

            J = JacobianSolverCpp(model).solve(
                q_in,
                method=method,
                epsilon=epsilon,
                use_central_diff=use_central_diff,
                target_link=None,
            )
            J = _apply_row_and_joint_masks(J, row_mask, joint_indices)
            if use_full_config and chain_indices is not None:
                J = _expand_numpy_jacobian(J, model.num_dof, chain_indices)
            return J

        raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    finally:
        _restore_model_chain(model, original_base, original_end)


__all__ = ["jacobian"]
