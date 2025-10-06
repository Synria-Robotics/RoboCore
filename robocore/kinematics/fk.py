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

from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy
from robocore.utils.backend import get_backend


def forward_kinematics(
    model: Any,
    q: Sequence[float] | Any,
    *,
    backend: str = 'auto',
    return_end: bool = False,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Dict[str, Any], Any]:
    """
    :param model: RobotModel instance
    :param q: Joint configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param return_end: Return only end-effector pose
    :param device: Torch device when using torch backend
    :param dtype: Torch dtype when using torch backend
    :return: Mapping link->pose or single 4x4 pose if return_end=True
    """
    b = get_backend() if backend == 'auto' else backend
    if b == 'numpy':
        solver = FKSolverNumPy(model)
        poses = solver.solve(q, return_end_only=return_end)
        return poses['end'] if return_end else poses
    elif b == 'torch':
        import torch
        from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
        
        solver = FKSolverTorch(model)
        if dtype is None:
            dtype = torch.float64
        poses = solver.solve(q, return_end_only=return_end, device=device, dtype=dtype)
        return poses['end'] if return_end else poses
    else:
        raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")


__all__ = ["forward_kinematics"]
