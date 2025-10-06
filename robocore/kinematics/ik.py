"""Unified inverse kinematics high-level API.

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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robocore.kinematics.ik_utils.ik_solver_numpy import IKSolverNumPy
from robocore.utils.backend import get_backend


def inverse_kinematics(
    model,
    target_pose: Sequence[Sequence[float]] | np.ndarray,
    q0: Sequence[float],
    *,
    backend: str = 'auto',
    method: str = 'dls',
    multi_start: int = 0,
    multi_noise: float = 0.3,
    random_seed: Optional[int] = None,
    # Local / partial task options
    target_link: Optional[str] = None,
    row_mask: Optional[Sequence[int | bool]] = None,
    # Redundancy / nullspace parameters
    nullspace_gain: float = 0.0,
    joint_centering: bool = True,
    joint_center_gain: float = 0.2,
    # Optional joint weights for centering (len = dof)
    joint_center_weights: Optional[Sequence[float]] = None,
    # torch specific passthrough (ignored by numpy backend)
    torch_device: Any | None = None,
    torch_dtype: Any | None = None,
    return_all: bool = False,
    **solver_kwargs,
) -> Dict[str, Any]:
    """
    :param model: RobotModel instance
    :param target_pose: 4x4 pose
    :param q0: Initial configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param method: 'pinv'|'dls'|'transpose'
    :param multi_start: Restart trials
    :param multi_noise: Gaussian noise scale (radians)
    :param random_seed: Seed for reproducibility
    :param torch_device: Torch device when using torch backend
    :param torch_dtype: Torch dtype when using torch backend
    :param return_all: Return all solutions
    :param solver_kwargs: Extra kwargs passed to solver
    :return: IK result dict
    """
    b = get_backend() if backend == 'auto' else backend

    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def _run_once(q_init):
        if b == 'numpy':
            solver = IKSolverNumPy(
                model,
                max_iters=solver_kwargs.pop('max_iters', 120),
                pos_tol=solver_kwargs.pop('pos_tol', 1e-4),
                ori_tol=solver_kwargs.pop('ori_tol', 1e-4),
            )
            res = solver.solve(
                np.asarray(target_pose),
                np.asarray(q_init),
                method=method,
                use_analytic_jacobian=solver_kwargs.pop('use_analytic_jacobian', True),
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solver_kwargs,
            )
            res['backend'] = 'numpy'
            return res
        elif b == 'torch':
            import torch

            from robocore.kinematics.ik_utils.ik_solver_torch import \
                IKSolverTorch
            dev = torch_device if torch_device is not None else 'cpu'
            solver = IKSolverTorch(
                model,
                max_iters=solver_kwargs.pop('max_iters', 120),
                pos_tol=solver_kwargs.pop('pos_tol', 1e-4),
                ori_tol=solver_kwargs.pop('ori_tol', 1e-4),
                device=dev,
                dtype=torch_dtype,
            )
            res = solver.solve(
                np.asarray(target_pose),
                q_init,
                method=method,
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solver_kwargs,
            )
            res['backend'] = 'torch'
            return res
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")

    base_res = _run_once(q0)
    if base_res.get('success') or multi_start <= 0:
        return base_res

    candidates: List[Dict[str, Any]] = [base_res]
    for _ in range(multi_start):
        noise = (rng.normal(size=len(q0)) * multi_noise) if rng else (np.random.randn(len(q0)) * multi_noise)
        q_pert = np.asarray(q0) + noise
        candidates.append(_run_once(q_pert))

    successes = [c for c in candidates if c.get('success')]
    if successes:
        successes.sort(key=lambda c: c['err_norm'])
        return successes[0]
    candidates.sort(key=lambda c: c['err_norm'])

    if return_all:
        return candidates
    else:
        return candidates[0]
