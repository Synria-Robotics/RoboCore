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
from robocore.kinematics.ik_utils.initial_guess import generate_initial_guesses
from robocore.utils.backend import get_backend


def inverse_kinematics(
    model,
    target_pose: Sequence[Sequence[float]] | np.ndarray,
    q0: Optional[Sequence[float] | np.ndarray] = None,
    *,
    method: str = 'dls',
    num_initial_guesses: int = 1,
    initial_guess_strategy: str = 'random',
    initial_guess_scale: float = 1.0,
    random_seed: Optional[int] = None,
    # Local / partial task options
    target_link: Optional[str] = None,
    row_mask: Optional[Sequence[int | bool]] = None,
    # Redundancy / nullspace parameters
    nullspace_gain: float = 0.0,
    joint_centering: bool = True,
    joint_center_gain: float = 0.2,
    joint_center_weights: Optional[Sequence[float]] = None,
    # torch specific passthrough
    torch_device: Any | None = None,
    torch_dtype: Any | None = None,
    **solver_kwargs,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute inverse kinematics for single or batch of target poses.
    
    :param model: RobotModel instance
    :param target_pose: Target pose(s) - 4x4 matrix or [B, 4, 4] array
    :param q0: Initial configuration (optional, used as base for noise-based strategies)
    :param method: 'pinv'|'dls'|'transpose'
    :param num_initial_guesses: Number of initial guesses to try (default: 1)
    :param initial_guess_strategy: Strategy - 'zero'|'random'|'sobol'|'latin'|'center'|'uniform'
    :param initial_guess_scale: Scale factor for joint limits (0.0 to 1.0)
    :param random_seed: Seed for reproducibility
    :param solver_kwargs: Extra kwargs passed to solver
    :return: IK result dict (single) or list of dicts (batch)
    """
    backend = get_backend()

    # Parse input dimensions
    target_arr = np.asarray(target_pose)
    is_batch = target_arr.ndim == 3 and target_arr.shape[1:] == (4, 4)
    batch_size = target_arr.shape[0] if is_batch else 1

    # Prepare q0 (optional, used as base for some strategies)
    n_dof = model.num_chain_dof
    base_q0 = np.asarray(q0) if q0 is not None else np.zeros(n_dof)
    if base_q0.ndim == 2:
        base_q0 = base_q0[0]  # Use first config as base

    # Generate initial guesses - ALWAYS use strategy
    initial_guesses = generate_initial_guesses(
        model,
        num_initial_guesses,
        strategy=initial_guess_strategy,
        seed=random_seed,
        scale=initial_guess_scale,
        base_q0=base_q0 if initial_guess_strategy in ('random', 'uniform') else None,
    )

    # Route to backend
    if backend == 'numpy':
        return _solve_numpy(
            model, target_arr, initial_guesses, is_batch,
            method=method,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
            **solver_kwargs,
        )
    elif backend == 'torch':
        return _solve_torch(
            model, target_arr, initial_guesses, is_batch,
            method=method,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            **solver_kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'")


def _solve_numpy(
    model, target_arr, initial_guesses, is_batch, *, method, target_link, row_mask,
    nullspace_gain, joint_centering, joint_center_gain, joint_center_weights, **solver_kwargs
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """NumPy backend IK solver."""
    # Extract solver init params
    solver_init = {k: solver_kwargs.pop(k) for k in ['min_damping', 'max_damping', 'base_step']
                   if k in solver_kwargs}

    solver = IKSolverNumPy(
        model,
        max_iters=solver_kwargs.pop('max_iters', 200),
        pos_tol=solver_kwargs.pop('pos_tol', 1e-3),
        ori_tol=solver_kwargs.pop('ori_tol', 1e-3),
        **solver_init,
    )

    common_kwargs = dict(
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

    def run_once(target, q_init):
        res = solver.solve(target, np.asarray(q_init), **common_kwargs)
        if isinstance(res, dict):
            res['backend'] = 'numpy'
        return res

    def best_result(candidates):
        """Select best from candidates: first successful, else lowest error."""
        successes = [c for c in candidates if c.get('success')]
        if successes:
            return min(successes, key=lambda c: c.get('err_norm', float('inf')))
        return min(candidates, key=lambda c: c.get('err_norm', float('inf')))

    num_guesses = len(initial_guesses)

    if not is_batch:
        # Single target
        if num_guesses == 1:
            return run_once(target_arr, initial_guesses[0])
        candidates = [run_once(target_arr, q0) for q0 in initial_guesses]
        return best_result(candidates)

    # Batch mode - use vectorized solver for efficiency
    batch_size = target_arr.shape[0]

    if num_guesses == 1:
        # Single guess per target - use true batch solver
        q0_batch = np.tile(initial_guesses[0], (batch_size, 1))
        res = solver.solve(target_arr, q0_batch, **common_kwargs)
        # Convert batch result to list of dicts
        if isinstance(res, dict):
            # Batch result format: {'q': [[...], [...]], 'success': [True, False], ...}
            results = []
            for i in range(batch_size):
                result_dict = {}
                for k, v in res.items():
                    if isinstance(v, list) and len(v) == batch_size:
                        result_dict[k] = v[i]
                    elif k == 'backend':
                        result_dict[k] = v
                    else:
                        result_dict[k] = v
                results.append(result_dict)
            return results
        return res

    # Multiple guesses - try each guess for each target
    results = []
    for target in target_arr:
        candidates = []
        for q0 in initial_guesses:
            res = run_once(target, q0)
            if res.get('success'):
                candidates.append(res)
                break  # Early exit on success
            candidates.append(res)
        results.append(best_result(candidates))
    return results


def _solve_torch(
    model, target_arr, initial_guesses, is_batch, *, method, target_link, row_mask,
    nullspace_gain, joint_centering, joint_center_gain, joint_center_weights,
    torch_device, torch_dtype, **solver_kwargs
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """PyTorch backend IK solver."""
    import torch
    from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch

    device = torch_device or 'cpu'
    dtype = torch_dtype or torch.float64

    # Extract solver init params
    solver_init = {k: solver_kwargs.pop(k) for k in ['min_damping', 'max_damping', 'base_step']
                   if k in solver_kwargs}

    solver = IKSolverTorch(
        model,
        max_iters=solver_kwargs.pop('max_iters', 200),
        pos_tol=solver_kwargs.pop('pos_tol', 1e-3),
        ori_tol=solver_kwargs.pop('ori_tol', 1e-3),
        device=device,
        dtype=dtype,
        **solver_init,
    )

    # Filter solve kwargs
    solve_kwargs = {k: v for k, v in solver_kwargs.items()
                    if k not in ['max_iters', 'pos_tol', 'ori_tol', 'min_damping', 'max_damping', 'base_step']}

    def to_torch(arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=dtype, device=device)
        return torch.from_numpy(np.asarray(arr)).to(dtype=dtype, device=device)

    def to_result_dict(res, is_batch_result=False):
        """Convert torch result to numpy dict."""
        if not isinstance(res, dict):
            return res
        out = {}
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                if is_batch_result and v.ndim >= 1:
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = v.detach().cpu().numpy() if v.numel() > 1 else v.item()
            else:
                out[k] = v
        out['backend'] = 'torch'
        return out

    def run_single(target, q_init):
        """Run single-mode solve."""
        res = solver.solve(
            to_torch(target), to_torch(q_init),
            method=method,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
            **solve_kwargs,
        )
        return to_result_dict(res)

    def best_result(candidates):
        successes = [c for c in candidates if c.get('success')]
        if successes:
            return min(successes, key=lambda c: c.get('err_norm', float('inf')))
        return min(candidates, key=lambda c: c.get('err_norm', float('inf')))

    num_guesses = len(initial_guesses)

    if not is_batch:
        # Single target
        if num_guesses == 1:
            return run_single(target_arr, initial_guesses[0])
        candidates = [run_single(target_arr, q0) for q0 in initial_guesses]
        return best_result(candidates)

    # Batch mode - use vectorized solver for efficiency
    batch_size = target_arr.shape[0]

    if num_guesses == 1:
        # Single guess per target - use true batch solver
        q0_batch = np.tile(initial_guesses[0], (batch_size, 1))
        target_torch = to_torch(target_arr)
        q0_torch = to_torch(q0_batch)

        res = solver.solve(
            target_torch, q0_torch,
            method=method,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
            **solve_kwargs,
        )

        # Convert batch result to list of dicts - optimize by batch converting tensors first
        if 'q' in res and isinstance(res['q'], torch.Tensor) and res['q'].ndim == 2:
            # Batch convert all tensors to numpy first (much faster than per-item conversion)
            q_np = res['q'].detach().cpu().numpy()
            if isinstance(res['success'], torch.Tensor):
                success_np = res['success'].detach().cpu().numpy()
            else:
                success_np = np.asarray(res['success'])
            if isinstance(res['iters'], torch.Tensor):
                iters_np = res['iters'].detach().cpu().numpy()
            else:
                iters_np = np.asarray(res['iters'])
            if isinstance(res['pos_err'], torch.Tensor):
                pos_err_np = res['pos_err'].detach().cpu().numpy()
            else:
                pos_err_np = np.asarray(res['pos_err'])
            if isinstance(res['ori_err'], torch.Tensor):
                ori_err_np = res['ori_err'].detach().cpu().numpy()
            else:
                ori_err_np = np.asarray(res['ori_err'])

            # Build result list from pre-converted arrays
            results = []
            for i in range(batch_size):
                results.append({
                    'q': q_np[i].tolist(),
                    'success': bool(success_np[i]),
                    'iters': int(iters_np[i]),
                    'method': method,
                    'pos_err': float(pos_err_np[i]),
                    'ori_err': float(ori_err_np[i]),
                    'backend': 'torch',
                })
            return results
        return to_result_dict(res, is_batch_result=True)

    # Multiple guesses - iterate per target (less efficient but more accurate)
    results = []
    for target in target_arr:
        candidates = []
        for q0 in initial_guesses:
            res = run_single(target, q0)
            if res.get('success'):
                candidates.append(res)
                break
            candidates.append(res)
        results.append(best_result(candidates))
    return results
