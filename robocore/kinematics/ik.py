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

    # Check if batch mode for multi_start validation
    target_arr = np.asarray(target_pose)
    is_batch = target_arr.ndim == 3 and target_arr.shape[1:] == (4, 4)

    # Batch mode: multi_start not supported
    if is_batch and multi_start > 0:
        raise ValueError("multi_start is not supported in batch mode")

    # Direct solver call - solver handles batch automatically
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

        # Handle single mode with multi_start/q0_retries
        if not is_batch:
            rng = np.random.default_rng(random_seed) if random_seed is not None else None

            def _run_once(q_init):
                # Filter out solver initialization parameters
                solve_kwargs = {}
                for k, v in solver_kwargs.items():
                    if k not in ['max_iters', 'pos_tol', 'ori_tol', 'min_damping', 'max_damping', 'base_step']:
                        solve_kwargs[k] = v

                res = solver.solve(
                    target_pose,
                    np.asarray(q_init),
                    method=method,
                    use_analytic_jacobian=solve_kwargs.pop('use_analytic_jacobian', True),
                    target_link=target_link,
                    row_mask=row_mask,
                    nullspace_gain=nullspace_gain,
                    joint_centering=joint_centering,
                    joint_center_gain=joint_center_gain,
                    joint_center_weights=joint_center_weights,
                    **solve_kwargs,
                )
                # Ensure result is a dict (not list)
                if isinstance(res, dict):
                    res['backend'] = 'numpy'
                return res

            # Handle multiple initial guesses
            if q0_retries is not None:
                q0_retries_arr = np.asarray(q0_retries)
                if q0_retries_arr.ndim != 2:
                    raise ValueError(f"q0_retries must be 2D array [R, n], got shape {q0_retries_arr.shape}")
                if q0_retries_arr.shape[1] != len(q0):
                    raise ValueError(f"q0_retries DOF mismatch: expected {len(q0)}, got {q0_retries_arr.shape[1]}")

                candidates: List[Dict[str, Any]] = []
                for q_retry in q0_retries_arr:
                    candidates.append(_run_once(q_retry))

                successes = [c for c in candidates if c.get('success')]
                if successes:
                    successes.sort(key=lambda c: c.get('err_norm', float('inf')))
                    return successes[0]
                candidates.sort(key=lambda c: c.get('err_norm', float('inf')))
                return candidates[0] if not return_all else candidates

            # Original multi_start logic
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
                successes.sort(key=lambda c: c.get('err_norm', float('inf')))
                return successes[0]
            candidates.sort(key=lambda c: c.get('err_norm', float('inf')))

            if return_all:
                return candidates
            else:
                return candidates[0]
        else:
            # Batch mode - direct call
            # Filter out solver initialization parameters
            solve_kwargs = {}
            for k, v in solver_kwargs.items():
                if k not in ['max_iters', 'pos_tol', 'ori_tol', 'min_damping', 'max_damping', 'base_step']:
                    solve_kwargs[k] = v

            result = solver.solve(
                target_pose,
                q0,
                method=method,
                use_analytic_jacobian=solve_kwargs.pop('use_analytic_jacobian', True),
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solve_kwargs,
            )
            # Convert from dict of lists to list of dicts for batch
            if isinstance(result, dict) and 'q' in result and isinstance(result['q'], list):
                batch_size = len(result['q'])
                results = []
                for i in range(batch_size):
                    results.append({
                        'q': result['q'][i],
                        'success': result['success'][i],
                        'iters': result['iters'][i],
                        'err_norm': result['err_norm'][i],
                        'method': result['method'][i],
                        'jacobian': result.get('jacobian', ['analytic'] * batch_size)[i] if isinstance(result.get('jacobian'), list) else result.get('jacobian', 'analytic'),
                        'pos_err': result.get('pos_err', [0.0] * batch_size)[i] if isinstance(result.get('pos_err'), list) else result.get('pos_err', 0.0),
                        'ori_err': result.get('ori_err', [0.0] * batch_size)[i] if isinstance(result.get('ori_err'), list) else result.get('ori_err', 0.0),
                        'backend': 'numpy',
                    })
                return results
            return result
    elif b == 'torch':
        import torch
        from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch

        dev = torch_device if torch_device is not None else 'cpu'
        solver_kwargs_init = {}
        if 'min_damping' in solver_kwargs:
            solver_kwargs_init['min_damping'] = solver_kwargs.pop('min_damping')
        if 'max_damping' in solver_kwargs:
            solver_kwargs_init['max_damping'] = solver_kwargs.pop('max_damping')
        if 'base_step' in solver_kwargs:
            solver_kwargs_init['base_step'] = solver_kwargs.pop('base_step')

        solver = IKSolverTorch(
            model,
            max_iters=solver_kwargs.pop('max_iters', 120),
            pos_tol=solver_kwargs.pop('pos_tol', 1e-4),
            ori_tol=solver_kwargs.pop('ori_tol', 1e-4),
            device=dev,
            dtype=torch_dtype,
            **solver_kwargs_init,
        )

        # Convert to torch tensors if needed
        if isinstance(target_pose, np.ndarray):
            target_torch = torch.from_numpy(target_pose).to(dtype=torch_dtype or torch.float64, device=dev)
        else:
            target_torch = target_pose
        if isinstance(q0, np.ndarray):
            q0_torch = torch.from_numpy(q0).to(dtype=torch_dtype or torch.float64, device=dev)
        else:
            q0_torch = q0

        result = solver.solve(
            target_torch,
            q0_torch,
            method=method,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
            **solver_kwargs,
        )

        # Convert to numpy and handle batch format
        if isinstance(result, dict):
            # Check if batch result: 'q' is tensor with ndim==2
            if 'q' in result and isinstance(result['q'], torch.Tensor) and result['q'].ndim == 2:
                # Batch result
                batch_size = result['q'].shape[0]
                results = []
                for i in range(batch_size):
                    results.append({
                        'q': result['q'][i].detach().cpu().numpy(),
                        'success': bool(result['success'][i].item()) if isinstance(result['success'], torch.Tensor) else result['success'][i],
                        'iters': int(result['iters'][i].item() if isinstance(result['iters'], torch.Tensor) else result['iters'][i]),
                        'method': method,
                        'pos_err': float(result['pos_err'][i].item()) if isinstance(result['pos_err'], torch.Tensor) else result['pos_err'][i],
                        'ori_err': float(result['ori_err'][i].item()) if isinstance(result['ori_err'], torch.Tensor) else result['ori_err'][i],
                        'backend': 'torch',
                    })
                return results
            else:
                # Single result
                result_np = {}
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result_np[k] = v.detach().cpu().numpy() if v.numel() == 1 else v.detach().cpu().tolist()
                    else:
                        result_np[k] = v
                result_np['backend'] = 'torch'
                return result_np
        return result
    else:
        raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
