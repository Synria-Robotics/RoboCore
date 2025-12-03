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
    return_end: bool = False,
    return_all_links: bool = False,
    link_names: Sequence[str] | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Dict[str, Any], Any]:
    """Compute forward kinematics for single chain or multiple chains.
    
    :param model: RobotModel instance
    :param q: Joint configuration (array-like or dict {joint_name: value})
    :param return_end: Return only end-effector pose (legacy, conflicts with return_all_links)
    :param return_all_links: Return FK for all links in the kinematic tree (uses multi-chain solver)
    :param link_names: Specific links to compute FK for (only with return_all_links=True)
    :param device: Torch device when using torch backend (uses global backend setting)
    :param dtype: Torch dtype when using torch backend (uses global backend setting)
    :return: Mapping link->pose or single 4x4 pose if return_end=True
    """
    b = get_backend()

    # Multi-chain FK path
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

    # Single chain FK path (legacy, backward compatible)
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
