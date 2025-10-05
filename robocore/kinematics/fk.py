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

from typing import Sequence, Dict, Any, Union

from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy

_HAS_TORCH = False
try:  # pragma: no cover
    from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch  # type: ignore
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # noqa: E722
    torch = None  # type: ignore
    FKSolverTorch = None  # type: ignore


def _select_backend(backend: str) -> str:
    if backend == 'auto':
        return 'torch' if _HAS_TORCH else 'numpy'
    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unsupported backend '{backend}', expected 'auto'|'numpy'|'torch'")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError("Torch backend requested but PyTorch is not available")
    return backend


def forward_kinematics(
    model: Any,
    q: Sequence[float] | Any,
    *,
    backend: str = 'auto',
    return_end: bool = False,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Dict[str, Any], Any]:
    """Compute forward kinematics for a robot model.

    Parameters
    ----------
    model : RobotModel
        Parsed robot model instance exposing attributes used internally
        (``_chain_joints``, ``_actuated``, ``base_link``, ``end_link``).
    q : sequence | tensor
        Joint configuration of length = dof.
    backend : str, default 'auto'
        'auto' picks torch if available else numpy; or explicitly 'numpy'|'torch'.
    return_end : bool, default False
        If True, return only end-effector 4x4 pose instead of full dict.
    device : torch device (torch backend only)
    dtype : torch dtype (torch backend only). Defaults to float64 if omitted.

    Returns
    -------
    dict | tensor
        Mapping link->pose or single 4x4 pose if return_end=True.
    """
    b = _select_backend(backend)
    if b == 'numpy':
        solver = FKSolverNumPy(model)
        poses = solver.solve(q, return_end_only=return_end)
        return poses['end'] if return_end else poses

    # torch path
    solver_torch = FKSolverTorch(model)  # type: ignore[misc]

    # Decide dtype default
    if dtype is None and _HAS_TORCH:  # pragma: no branch
        dtype = torch.float64  # type: ignore[attr-defined]

    poses = solver_torch.solve(q, return_end_only=return_end, device=device, dtype=dtype)
    return poses['end'] if return_end else poses


__all__ = ["forward_kinematics"]
