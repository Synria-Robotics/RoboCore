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
    target_pose: Sequence[Sequence[float]] | np.ndarray | Dict[str, Sequence[Sequence[float]] | np.ndarray] = None,
    q0: Optional[Sequence[float] | np.ndarray] = None,
    *,
    method: str = 'dls',
    num_initial_guesses: int = 1,
    initial_guess_strategy: str = 'random',
    initial_guess_scale: float = 1.0,
    random_seed: Optional[int] = None,
    # Local / partial task options
    target_link: Optional[str] = None,
    base_link: Optional[str] = None,
    end_link: Optional[str] = None,
    # Multi-chain options
    targets: Optional[Dict[str, Sequence[Sequence[float]] | np.ndarray]] = None,
    end_links: Optional[Sequence[str]] = None,
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
    
    Supports both single-chain and multi-chain modes:
    - Single-chain: target_pose (4x4 or [B, 4, 4]), end_link (optional)
    - Multi-chain: targets={end_link: pose}, end_links=[...], base_link (optional)
    
    :param model: RobotModel instance
    :param target_pose: Target pose(s) - 4x4 matrix or [B, 4, 4] array (single-chain mode)
    :param q0: Initial configuration - full config vector [num_dof] or chain config (optional)
    :param method: 'pinv'|'dls'|'transpose'
    :param num_initial_guesses: Number of initial guesses to try (default: 1)
    :param initial_guess_strategy: Strategy - 'zero'|'random'|'sobol'|'latin'|'center'|'uniform'
    :param initial_guess_scale: Scale factor for joint limits (0.0 to 1.0)
    :param random_seed: Seed for reproducibility
    :param target_link: Target link name (legacy, use end_link instead)
    :param base_link: Base link name (if specified, uses dynamic path and full config space)
    :param end_link: End link name (single-chain mode, if specified, uses dynamic path and full config space)
    :param targets: Multi-chain mode - dict mapping end_link to target pose {end_link: 4x4 or [B, 4, 4]}
    :param end_links: Multi-chain mode - list of end link names
    :param solver_kwargs: Extra kwargs passed to solver
    :return: IK result dict (single) or list of dicts (batch), with 'q' in full config space
    """
    backend = get_backend()
    
    # Check if multi-chain mode is requested
    if targets is not None or end_links is not None:
        # Multi-chain mode
        if targets is None or end_links is None:
            raise ValueError("Multi-chain mode requires both 'targets' and 'end_links' parameters")
        
        return _solve_multichain_ik(
            model, targets, end_links, q0,
            base_link=base_link,
            method=method,
            num_initial_guesses=num_initial_guesses,
            initial_guess_strategy=initial_guess_strategy,
            initial_guess_scale=initial_guess_scale,
            random_seed=random_seed,
            **solver_kwargs
        )
    
    # Single-chain mode
    if target_pose is None:
        raise ValueError("Either 'target_pose' (single-chain) or 'targets' (multi-chain) must be provided")
    
    # Handle dynamic base_link/end_link
    use_full_config = False
    chain_indices = None
    original_end = None
    original_base = None
    
    if base_link is not None or end_link is not None:
        # Dynamic path: extract chain joint values from full configuration
        base = base_link or model.base_link
        end = end_link or (target_link or model.end_link)
        use_full_config = True
        chain_indices = model._get_joint_indices(base, end)
        
        # Temporarily update model's end_link for IK computation
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
        
        # Extract chain joint values from full configuration if q0 is full config
        if q0 is not None:
            q0_arr = np.asarray(q0)
            if len(q0_arr.flatten()) == model.num_dof:
                q0_chain = q0_arr[chain_indices] if q0_arr.ndim == 1 else q0_arr[:, chain_indices]
                q0 = q0_chain.tolist() if hasattr(q0_chain, 'tolist') else list(q0_chain)

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
    try:
        if backend == 'numpy':
            result = _solve_numpy(
                model, target_arr, initial_guesses, is_batch,
                method=method,
                target_link=target_link or end_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                **solver_kwargs,
            )
        elif backend == 'torch':
            result = _solve_torch(
                model, target_arr, initial_guesses, is_batch,
                method=method,
                target_link=target_link or end_link,
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
        
        # Expand result to full configuration space if dynamic path was used
        if use_full_config and chain_indices is not None:
            if isinstance(result, list):
                # Batch mode
                for res in result:
                    if 'q' in res:
                        q_chain = np.array(res['q'])
                        q_full = np.zeros(model.num_dof)
                        q_full[chain_indices] = q_chain
                        res['q'] = q_full.tolist()
            else:
                # Single mode
                if 'q' in result:
                    q_chain = np.array(result['q'])
                    q_full = np.zeros(model.num_dof)
                    q_full[chain_indices] = q_chain
                    result['q'] = q_full.tolist()
        
        return result
    finally:
        # Restore original end_link and base_link if we modified them
        if original_end is not None:
            model.end_link = original_end
            model.base_link = original_base
            model._build_chain()


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

    # Multiple guesses - smart batch processing with progressive refinement
    # Strategy:
    # 1. Batch process all targets with first guess (fast, vectorized)
    # 2. For failed targets, try remaining guesses in batches
    # 3. This combines batch efficiency with early exit benefits
    num_guesses = len(initial_guesses)

    # Initialize results with first guess for all targets
    q0_first = np.tile(initial_guesses[0], (batch_size, 1))
    first_results = solver.solve(target_arr, q0_first, **common_kwargs)

    # Convert to list format if needed
    if isinstance(first_results, dict):
        first_results_list = []
        for i in range(batch_size):
            result_dict = {}
            for k, v in first_results.items():
                if isinstance(v, list) and len(v) == batch_size:
                    result_dict[k] = v[i]
                elif k == 'backend':
                    result_dict[k] = v
                else:
                    result_dict[k] = v
            first_results_list.append(result_dict)
        first_results = first_results_list

    # Track which targets need more guesses
    results = []
    failed_targets = []  # List of (target_idx, candidates) tuples
    failed_indices = []

    for target_idx in range(batch_size):
        first_res = first_results[target_idx] if isinstance(first_results, list) else first_results
        candidates = [first_res]

        if first_res.get('success'):
            # First guess succeeded, use it
            results.append(first_res)
        else:
            # First guess failed, need to try other guesses
            failed_targets.append(target_arr[target_idx])
            failed_indices.append(target_idx)
            results.append(None)  # Placeholder, will be filled later

    # If all targets succeeded, return early
    if not failed_targets:
        return results

    # Process failed targets with remaining guesses in batches
    # Strategy: For each remaining guess, batch process all failed targets
    for guess_idx in range(1, num_guesses):
        if not failed_targets:
            break  # All targets succeeded

        # Batch process all failed targets with this guess
        failed_targets_arr = np.array(failed_targets)
        q0_batch = np.tile(initial_guesses[guess_idx], (len(failed_targets), 1))
        batch_results = solver.solve(failed_targets_arr, q0_batch, **common_kwargs)

        # Convert to list format if needed
        if isinstance(batch_results, dict):
            batch_results_list = []
            for i in range(len(failed_targets)):
                result_dict = {}
                for k, v in batch_results.items():
                    if isinstance(v, list) and len(v) == len(failed_targets):
                        result_dict[k] = v[i]
                    elif k == 'backend':
                        result_dict[k] = v
                    else:
                        result_dict[k] = v
                batch_results_list.append(result_dict)
            batch_results = batch_results_list

        # Update results and remove successful targets from failed list
        new_failed_targets = []
        new_failed_indices = []
        new_candidates_map = {}  # Map from original index to candidates list

        for i, failed_idx in enumerate(failed_indices):
            guess_res = batch_results[i] if isinstance(batch_results, list) else batch_results
            # Get existing candidates for this target
            if failed_idx not in new_candidates_map:
                # Find the first result for this target
                first_res = first_results[failed_idx] if isinstance(first_results, list) else first_results
                new_candidates_map[failed_idx] = [first_res]
            new_candidates_map[failed_idx].append(guess_res)

            if guess_res.get('success'):
                # This guess succeeded, use best from all candidates
                results[failed_idx] = best_result(new_candidates_map[failed_idx])
            else:
                # Still failed, keep trying
                new_failed_targets.append(failed_targets[i])
                new_failed_indices.append(failed_idx)

        failed_targets = new_failed_targets
        failed_indices = new_failed_indices

    # Fill in any remaining failed targets with best result from all guesses
    for failed_idx in failed_indices:
        if results[failed_idx] is None:
            candidates = new_candidates_map.get(failed_idx, [])
            if candidates:
                results[failed_idx] = best_result(candidates)
            else:
                # Fallback
                results[failed_idx] = {'q': [0.0] * model.num_chain_dof, 'success': False}

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


def _solve_multichain_ik(
    model: Any,
    targets: Dict[str, Sequence[Sequence[float]] | np.ndarray],
    end_links: Sequence[str],
    q0: Optional[Sequence[float] | np.ndarray] = None,
    *,
    base_link: Optional[str] = None,
    method: str = 'dls',
    num_initial_guesses: int = 1,
    initial_guess_strategy: str = 'random',
    initial_guess_scale: float = 1.0,
    random_seed: Optional[int] = None,
    **solver_kwargs,
) -> Dict[str, Any]:
    """Solve multi-chain inverse kinematics using unified configuration space.
    
    :param model: RobotModel instance
    :param targets: Dict mapping end_link to target pose {end_link: 4x4 or [B, 4, 4]}
    :param end_links: List of end link names
    :param q0: Initial full configuration vector [num_dof] (optional)
    :param base_link: Base link name (if None, uses model.base_link)
    :param method: IK method ('dls', 'pinv', 'transpose')
    :param num_initial_guesses: Number of initial guesses
    :param initial_guess_strategy: Strategy for generating guesses
    :param initial_guess_scale: Scale factor for joint limits
    :param random_seed: Random seed
    :param solver_kwargs: Extra solver parameters
    :return: IK result dict with 'q' in full config space
    """
    base = base_link or model.base_link
    
    # Create chain views for each end_link
    chain_views = {}
    for end_link in end_links:
        try:
            chain_views[end_link] = model.get_chain(end_link, base)
        except ValueError as e:
            raise ValueError(f"Cannot create chain for {end_link}: {e}")
    
    # Detect shared joints (joints that appear in all chains)
    if len(chain_views) >= 2:
        chain_joint_sets = [
            set(chain_view._chain_joint_names)
            for chain_view in chain_views.values()
        ]
        shared_joint_names = set.intersection(*chain_joint_sets)
        shared_joint_indices = [
            model._dof_name_to_index[j]
            for j in shared_joint_names
            if j in model._dof_name_to_index
        ]
    else:
        shared_joint_names = set()
        shared_joint_indices = []
    
    # Prepare initial configuration
    if q0 is None:
        q0 = np.zeros(model.num_dof)
    else:
        q0 = np.asarray(q0)
        if len(q0) != model.num_dof:
            raise ValueError(f"Expected q0 of length {model.num_dof} (unified config space), got {len(q0)}")
    
    # Extract solver parameters
    max_iters = solver_kwargs.pop('max_iters', 200)
    pos_tol = solver_kwargs.pop('pos_tol', 1e-3)
    ori_tol = solver_kwargs.pop('ori_tol', 1e-3)
    
    # Multi-chain IK iteration
    q = q0.copy()
    
    for it in range(max_iters):
        # Compute current poses and errors
        errors = []
        jacobians = []
        
        for end_link in end_links:
            chain_view = chain_views[end_link]
            target = np.asarray(targets[end_link])
            
            # Current pose
            T_cur = chain_view.forward_kinematics(q)
            if hasattr(T_cur, 'detach'):
                T_cur = T_cur.detach().cpu().numpy()
            T_cur = np.asarray(T_cur)
            
            # Error (position + orientation)
            e_pos = target[:3, 3] - T_cur[:3, 3]
            
            # Orientation error (axis-angle)
            R_cur = T_cur[:3, :3]
            R_tgt = target[:3, :3]
            R_err = R_tgt @ R_cur.T
            
            trace = np.trace(R_err)
            theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            if theta < 1e-6:
                e_ori = np.zeros(3)
            else:
                axis = np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1]
                ]) / (2 * np.sin(theta))
                e_ori = theta * axis
            
            error = np.concatenate([e_pos, e_ori])
            errors.append(error)
            
            # Jacobian (6 x nq_full)
            J = chain_view.jacobian(q)
            if hasattr(J, 'detach'):
                J = J.detach().cpu().numpy()
            J = np.asarray(J)
            jacobians.append(J)
        
        # Stack errors and Jacobians
        e_total = np.concatenate(errors)
        J_total = np.vstack(jacobians)
        
        # Check convergence
        if np.linalg.norm(e_total) < pos_tol + ori_tol:
            return {
                'q': q.tolist(),
                'success': True,
                'iters': it + 1,
                'pos_err': np.linalg.norm(e_total[:3*len(end_links)]),
                'ori_err': np.linalg.norm(e_total[3*len(end_links):]),
            }
        
        # Solve for joint update
        if method == 'dls':
            # Damped least squares
            damping = solver_kwargs.get('damping', 1e-3)
            JT = J_total.T
            A = J_total @ JT + (damping ** 2) * np.eye(J_total.shape[0])
            dq = JT @ np.linalg.solve(A, e_total)
        elif method == 'pinv':
            # Pseudoinverse
            dq = np.linalg.pinv(J_total) @ e_total
        elif method == 'transpose':
            # Transpose method
            alpha = solver_kwargs.get('alpha', 0.1)
            dq = alpha * J_total.T @ e_total
        else:
            raise ValueError(f"Unknown method '{method}'")
        
        # Step limit
        step_limit = solver_kwargs.get('step_limit', 0.2)
        dq_norm = np.linalg.norm(dq)
        if dq_norm > step_limit:
            dq = dq * (step_limit / dq_norm)
        
        # Update configuration
        q = q + dq
        
        # Apply joint limits (use DOF limits for unified config space)
        if hasattr(model, 'dof_limits_min') and hasattr(model, 'dof_limits_max'):
            q = np.clip(q, model.dof_limits_min, model.dof_limits_max)
    
    # Max iterations reached
    return {
        'q': q.tolist(),
        'success': False,
        'iters': max_iters,
        'pos_err': np.linalg.norm(e_total[:3*len(end_links)]),
        'ori_err': np.linalg.norm(e_total[3*len(end_links):]),
    }
