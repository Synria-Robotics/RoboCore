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

from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from robocore.utils.backend import get_backend


def jacobian(
    model: Any,
    q: Sequence[float] | Any,
    *,
    backend: str = 'auto',
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
) -> Union[Any, Any]:
    """
    :param model: RobotModel instance
    :param q: Joint configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param method: 'analytic'|'numeric'|'autograd'
    :param target_link: Target link name
    :param joint_indices: Selected joint indices
    :param row_mask: Row selection mask (len=6)
    :param epsilon: Finite-difference step (numeric)
    :param use_central_diff: Use central difference (numeric)
    :param device: Torch device when using torch backend
    :param dtype: Torch dtype when using torch backend
    :return: 6xn Jacobian matrix
    """
    b = get_backend() if backend == 'auto' else backend

    method = method.lower()

    if b == 'numpy':
        # NumPy backend
        solver = JacobianSolverNumPy(model)
        if method in ('analytic', 'numeric'):
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
                J = J[mask_bool, :]
            # Apply joint (column) selection if provided
            if joint_indices is not None:
                J = J[:, list(joint_indices)]
            return J
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

        if method in ('analytic', 'numeric', 'autograd'):
            J = solver.solve(
                q,
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
                    raise ValueError("row_mask must have length 6")
                J = J[mask_bool, :]
            if joint_indices is not None:
                J = J[:, torch.tensor(list(joint_indices), dtype=torch.long)]
            return J
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'analytic', 'numeric', or 'autograd'.")
    else:
        raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")


__all__ = ["jacobian"]
