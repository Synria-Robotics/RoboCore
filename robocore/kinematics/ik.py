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

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from robocore.kinematics.ik_utils.ik_solver_numpy import IKSolverNumPy
from robocore.utils.backend import get_backend


def _normalize_ik_input(target_pose: Any, q0: Any) -> tuple[Any, Any, bool]:
    """Normalize IK input for batch processing.
    
    :param target_pose: Target pose(s) - 4x4 matrix or [B, 4, 4] array
    :param q0: Initial configuration(s) - [n] or [B, n] array
    :return: Tuple of (normalized_target_pose, normalized_q0, is_batch)
    """
    target_arr = np.asarray(target_pose)
    q0_arr = np.asarray(q0)

    # Check if batch mode
    is_batch = target_arr.ndim == 3 and target_arr.shape[1:] == (4, 4)

    if is_batch:
        # Batch mode: [B, 4, 4] and [B, n]
        if q0_arr.ndim == 1:
            # q0 is single, broadcast to batch
            batch_size = target_arr.shape[0]
            q0_arr = np.tile(q0_arr, (batch_size, 1))
        elif q0_arr.ndim != 2:
            raise ValueError(f"Expected q0 with shape [n] or [B, n], got {q0_arr.shape}")
    else:
        # Single mode: ensure 2D for target_pose
        if target_arr.ndim == 2:
            target_arr = target_arr.reshape(1, 4, 4)
        elif target_arr.ndim != 3 or target_arr.shape != (1, 4, 4):
            raise ValueError(f"Expected target_pose with shape [4, 4] or [1, 4, 4], got {target_arr.shape}")
        # q0 should be 1D
        if q0_arr.ndim == 1:
            q0_arr = q0_arr.reshape(1, -1)
        else:
            raise ValueError(f"Expected q0 with shape [n], got {q0_arr.shape}")

    return target_arr, q0_arr, is_batch


def inverse_kinematics(
    model,
    target_pose: Sequence[Sequence[float]] | np.ndarray,
    q0: Sequence[float] | np.ndarray,
    *,
    method: str = 'dls',
    multi_start: int = 0,
    multi_noise: float = 0.3,
    q0_retries: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
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
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute inverse kinematics for single or batch of target poses.
    
    Supports both single and batch processing:
    - Single: target_pose [4, 4], q0 [n] -> returns single dict
    - Batch: target_pose [B, 4, 4], q0 [B, n] or [n] -> returns list of dicts
    
    :param model: RobotModel instance
    :param target_pose: Target pose(s) - 4x4 matrix or [B, 4, 4] array
    :param q0: Initial configuration(s) - [n] or [B, n] array
    :param method: 'pinv'|'dls'|'transpose'
    :param multi_start: Restart trials (only for single mode, ignored if q0_retries is provided)
    :param multi_noise: Gaussian noise scale (radians, used only with multi_start)
    :param q0_retries: Multiple initial guesses - [R, n] array (only for single mode)
    :param random_seed: Seed for reproducibility
    :param torch_device: Torch device when using torch backend (uses global backend setting)
    :param torch_dtype: Torch dtype when using torch backend (uses global backend setting)
    :param return_all: Return all solutions (only for single mode)
    :param solver_kwargs: Extra kwargs passed to solver
    :return: IK result dict (single) or list of dicts (batch)
    """
    b = get_backend()

    # Normalize input for batch processing
    target_normalized, q0_normalized, is_batch = _normalize_ik_input(target_pose, q0)

    # Batch mode: multi_start not supported
    if is_batch and multi_start > 0:
        raise ValueError("multi_start is not supported in batch mode")

    if is_batch:
        # Batch processing
        batch_size = target_normalized.shape[0]

        if b == 'numpy':
            # Extract solver initialization parameters
            solver_init_kwargs = {}
            if 'min_damping' in solver_kwargs:
                solver_init_kwargs['min_damping'] = solver_kwargs['min_damping']
            if 'max_damping' in solver_kwargs:
                solver_init_kwargs['max_damping'] = solver_kwargs['max_damping']
            if 'base_step' in solver_kwargs:
                solver_init_kwargs['base_step'] = solver_kwargs['base_step']

            solver = IKSolverNumPy(
                model,
                max_iters=solver_kwargs.get('max_iters', 120),
                pos_tol=solver_kwargs.get('pos_tol', 1e-4),
                ori_tol=solver_kwargs.get('ori_tol', 1e-4),
                **solver_init_kwargs,
            )
            results_dict = solver.solve_batch(
                target_normalized,
                q0_normalized,
                method=method,
                use_analytic_jacobian=solver_kwargs.get('use_analytic_jacobian', True),
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **{k: v for k, v in solver_kwargs.items() if k not in ['max_iters', 'pos_tol', 'ori_tol', 'use_analytic_jacobian', 'min_damping', 'max_damping', 'base_step']},
            )
            # Convert from dict of lists to list of dicts
            results = []
            for i in range(batch_size):
                results.append({
                    'q': results_dict['q'][i],
                    'success': results_dict['success'][i],
                    'iters': results_dict['iters'][i],
                    'err_norm': results_dict['err_norm'][i],
                    'method': results_dict['method'][i],
                    'jacobian': results_dict.get('jacobian', ['analytic'] * batch_size)[i] if isinstance(results_dict.get('jacobian'), list) else results_dict.get('jacobian', 'analytic'),
                    'pos_err': results_dict.get('pos_err', [0.0] * batch_size)[i] if isinstance(results_dict.get('pos_err'), list) else results_dict.get('pos_err', 0.0),
                    'ori_err': results_dict.get('ori_err', [0.0] * batch_size)[i] if isinstance(results_dict.get('ori_err'), list) else results_dict.get('ori_err', 0.0),
                    'backend': 'numpy',
                })
            return results
        elif b == 'torch':
            import torch
            from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch

            dev = torch_device if torch_device is not None else 'cpu'
            solver = IKSolverTorch(
                model,
                max_iters=solver_kwargs.get('max_iters', 120),
                pos_tol=solver_kwargs.get('pos_tol', 1e-4),
                ori_tol=solver_kwargs.get('ori_tol', 1e-4),
                device=dev,
                dtype=torch_dtype,
            )

            # Convert to torch tensors
            target_torch = torch.from_numpy(target_normalized).to(dtype=torch_dtype or torch.float64, device=dev)
            q0_torch = torch.from_numpy(q0_normalized).to(dtype=torch_dtype or torch.float64, device=dev)

            # Use batch solver
            result_batch = solver._solve_batch(
                target_torch,
                q0_torch,
                method=method,
                pos_weight=solver_kwargs.get('pos_weight', 1.0),
                ori_weight=solver_kwargs.get('ori_weight', 1.0),
                max_step_norm=solver_kwargs.get('max_step_norm', 0.5),
                damping=solver_kwargs.get('damping', None),
            )

            # Convert to list of dicts
            results = []
            for i in range(batch_size):
                results.append({
                    'q': result_batch['q'][i].detach().cpu().numpy(),
                    'success': bool(result_batch['success'][i].item()),
                    'iterations': int(result_batch['iterations'][i].item()),
                    'method': method,
                    'pos_err': float(result_batch['pos_err'][i].item()) if 'pos_err' in result_batch else 0.0,
                    'ori_err': float(result_batch['ori_err'][i].item()) if 'ori_err' in result_batch else 0.0,
                    'backend': 'torch',
                })
            return results
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")

    # Single mode (backward compatible)
    target_single = target_normalized[0]
    q0_single = q0_normalized[0]

    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def _run_once(q_init):
        # Create a fresh copy for each call to avoid modifying the original
        solver_kwargs_local = solver_kwargs.copy()

        if b == 'numpy':
            # Extract solver initialization parameters
            solver_init_kwargs = {}
            if 'min_damping' in solver_kwargs_local:
                solver_init_kwargs['min_damping'] = solver_kwargs_local.pop('min_damping')
            if 'max_damping' in solver_kwargs_local:
                solver_init_kwargs['max_damping'] = solver_kwargs_local.pop('max_damping')
            if 'base_step' in solver_kwargs_local:
                solver_init_kwargs['base_step'] = solver_kwargs_local.pop('base_step')

            solver = IKSolverNumPy(
                model,
                max_iters=solver_kwargs_local.pop('max_iters', 120),
                pos_tol=solver_kwargs_local.pop('pos_tol', 1e-4),
                ori_tol=solver_kwargs_local.pop('ori_tol', 1e-4),
                **solver_init_kwargs,
            )
            res = solver.solve(
                target_single,
                np.asarray(q_init),
                method=method,
                use_analytic_jacobian=solver_kwargs_local.pop('use_analytic_jacobian', True),
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solver_kwargs_local,
            )
            res['backend'] = 'numpy'
            return res
        elif b == 'torch':
            import torch
            from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch

            dev = torch_device if torch_device is not None else 'cpu'
            # Extract solver initialization parameters
            solver_kwargs_init = {}
            if 'min_damping' in solver_kwargs_local:
                solver_kwargs_init['min_damping'] = solver_kwargs_local.pop('min_damping')
            if 'max_damping' in solver_kwargs_local:
                solver_kwargs_init['max_damping'] = solver_kwargs_local.pop('max_damping')
            if 'base_step' in solver_kwargs_local:
                solver_kwargs_init['base_step'] = solver_kwargs_local.pop('base_step')
            solver = IKSolverTorch(
                model,
                max_iters=solver_kwargs_local.pop('max_iters', 120),
                pos_tol=solver_kwargs_local.pop('pos_tol', 1e-4),
                ori_tol=solver_kwargs_local.pop('ori_tol', 1e-4),
                device=dev,
                dtype=torch_dtype,
                **solver_kwargs_init,
            )
            res = solver.solve(
                target_single,
                q_init,
                method=method,
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solver_kwargs_local,
            )
            res['backend'] = 'torch'
            return res
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")

    # Handle multiple initial guesses
    if q0_retries is not None:
        # Use provided retries
        q0_retries_arr = np.asarray(q0_retries)
        if q0_retries_arr.ndim != 2:
            raise ValueError(f"q0_retries must be 2D array [R, n], got shape {q0_retries_arr.shape}")
        if q0_retries_arr.shape[1] != len(q0_single):
            raise ValueError(f"q0_retries DOF mismatch: expected {len(q0_single)}, got {q0_retries_arr.shape[1]}")

        candidates: List[Dict[str, Any]] = []
        for q_retry in q0_retries_arr:
            candidates.append(_run_once(q_retry))

        # Select best: first successful, or best error if none successful
        successes = [c for c in candidates if c.get('success')]
        if successes:
            successes.sort(key=lambda c: c.get('err_norm', float('inf')))
            return successes[0]
        candidates.sort(key=lambda c: c.get('err_norm', float('inf')))
        return candidates[0] if not return_all else candidates

    # Original multi_start logic (backward compatible)
    base_res = _run_once(q0_single)
    if base_res.get('success') or multi_start <= 0:
        return base_res

    candidates: List[Dict[str, Any]] = [base_res]
    for _ in range(multi_start):
        noise = (rng.normal(size=len(q0_single)) * multi_noise) if rng else (np.random.randn(len(q0_single)) * multi_noise)
        q_pert = np.asarray(q0_single) + noise
        candidates.append(_run_once(q_pert))

    successes = [c for c in candidates if c.get('success')]
    if successes:
        successes.sort(key=lambda c: c.get('err_norm', float('inf')))
        return successes[0]
    candidates.sort(key=lambda c: c.get('err_norm', float('inf')))

    if return_all:
        return candidates
    else:
        return candidates[0]
