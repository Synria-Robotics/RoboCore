"""Dual-arm cooperative kinematics (Phase 0/1 MVP + Phase 2.1 Extensions).

Features:
Phase 0/1:
- Block-diagonal Jacobian assembly for two independent chains (same base world).
- Simultaneous absolute dual-end IK solving (both arms track their own 6D pose).
- Minimal dependency: reuse existing single-chain jacobian()/fk() and iterative IK logic style.

Phase 2.1:
- Relative pose task constraints (T_rel = T_left^-1 @ T_right)
- Relative pose error computation (6D in left arm frame)
- Relative Jacobian construction (analytical via adjoint transform)
- Weighted task composition (absolute + relative mixed)

Future:
- Hierarchical / nullspace priority tasks
- Joint-level redundancy optimization
- Trajectory-level synchronization

Author: Synria Robotics Team
License: GPL-3.0
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, List
from dataclasses import dataclass
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.transform.conversions import matrix_to_axis_angle
from .jacobian import jacobian as single_jacobian
from .fk import forward_kinematics as single_fk

# ----------------------- FK / Jacobian combined -----------------------

def dual_fk(left: RobotModel, right: RobotModel, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'numpy') -> Dict[str, Any]:
    """Compute FK for both arms.
    Returns dict with 'left', 'right' 4x4 poses (end-effector) and full link maps if needed later.

    :param left: left arm chain model
    :param right: right arm chain model
    :param q_left: left joint values
    :param q_right: right joint values
    :param backend: backend selection
    """
    T_l = left.fk(q_left, backend=backend, return_end=True)
    T_r = right.fk(q_right, backend=backend, return_end=True)
    return {"left": T_l, "right": T_r}

def dual_block_jacobian(left: RobotModel, right: RobotModel, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'auto') -> Any:
    """Assemble block-diagonal 12 x (nL + nR) Jacobian for independent absolute tasks.
    J = [[J_L, 0],[0, J_R]].
    """
    J_L = single_jacobian(left, q_left, backend=backend)
    J_R = single_jacobian(right, q_right, backend=backend)
    # Normalize to numpy for MVP (if torch backend requested, user can still get torch per-arm matrices)
    try:
        import torch  # noqa
        if hasattr(J_L, 'detach'):
            J_L_np = J_L.detach().cpu().numpy()
        else:
            J_L_np = np.array(J_L)
        if hasattr(J_R, 'detach'):
            J_R_np = J_R.detach().cpu().numpy()
        else:
            J_R_np = np.array(J_R)
    except Exception:  # torch not present
        J_L_np = np.array(J_L)
        J_R_np = np.array(J_R)
    nL = J_L_np.shape[1]
    nR = J_R_np.shape[1]
    J = np.zeros((12, nL + nR))
    J[0:6, 0:nL] = J_L_np
    J[6:12, nL:] = J_R_np
    return J

# ----------------------- Dual absolute IK -----------------------------

def _pose_error(T_current, T_target):
    """Compute 6D pose error (pos diff + axis-angle orientation error)."""
    # Position
    p_c = T_current[0:3, 3]
    p_t = T_target[0:3, 3]
    e_pos = p_t - p_c
    # Orientation
    R_c = T_current[0:3, 0:3]
    R_t = T_target[0:3, 0:3]
    R_err = R_t @ R_c.T
    # axis-angle from rotation matrix
    angle = np.arccos(max(-1.0, min(1.0, (np.trace(R_err) - 1.0) * 0.5)))
    if angle < 1e-12:
        e_ori = np.zeros(3)
    else:
        wx = R_err[2, 1] - R_err[1, 2]
        wy = R_err[0, 2] - R_err[2, 0]
        wz = R_err[1, 0] - R_err[0, 1]
        axis = np.array([wx, wy, wz]) / (2.0 * np.sin(angle) + 1e-12)
        e_ori = axis * angle
    return np.concatenate([e_pos, e_ori])

def dual_ik(
    left: RobotModel,
    right: RobotModel,
    target_left: Sequence[Sequence[float]] | np.ndarray | None,
    target_right: Sequence[Sequence[float]] | np.ndarray | None,
    q0_left: Optional[Sequence[float]] = None,
    q0_right: Optional[Sequence[float]] = None,
    max_iters: int = 120,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-4,
    method: str = 'pinv',
    backend: str = 'auto',
    check_workspace: bool = True,  # NEW: pre-check reachability
    workspace_tolerance: float = 0.08,  # 8cm tolerance for workspace check
    workspace_samples: int = 5000,  # samples for auto workspace computation
    **solver_kwargs,
) -> Dict[str, Any]:
    """Dual-arm absolute IK MVP (independent tasks via existing single-arm solvers).

    This function simply calls each arm's IK solver separately (because there is
    no coupling yet) and aggregates results into one dictionary. It is a stable
    baseline before introducing coupled relative constraints.

    :param left: left arm chain model
    :param right: right arm chain model
    :param target_left: 4x4 target pose (or None to skip)
    :param target_right: 4x4 target pose (or None to skip)
    :param q0_left: initial guess for left
    :param q0_right: initial guess for right
    :param max_iters: max IK iterations (per arm)
    :param pos_tol: position tolerance
    :param ori_tol: orientation tolerance
    :param method: 'pinv'|'dls'|'transpose' (passed through)
    :param backend: kinematics backend
    :param check_workspace: if True, verify targets are in reachable workspace before IK
    :param workspace_tolerance: distance threshold for workspace check (meters)
    :param workspace_samples: number of samples for workspace computation if not cached
    :param solver_kwargs: additional solver parameters
    :return: dict with keys q_left, q_right, success, pos_err_left/right, ori_err_left/right, iters, method
    """
    if target_left is None and target_right is None:
        raise ValueError("At least one of target_left or target_right must be provided")

    # Pre-check workspace reachability (optional but recommended)
    if check_workspace:
        if target_left is not None:
            T_left = np.array(target_left)
            p_left = T_left[0:3, 3]
            if not left.is_point_reachable(p_left, tolerance=workspace_tolerance,
                                            auto_compute=True, num_samples=workspace_samples):
                return {
                    'q_left': q0_left or [0.0] * left.num_dof(),
                    'q_right': q0_right or [0.0] * right.num_dof(),
                    'iters': 0,
                    'success': False,
                    'pos_err_left': float('inf'),
                    'ori_err_left': float('inf'),
                    'pos_err_right': 0.0,
                    'ori_err_right': 0.0,
                    'method': method,
                    'error': 'left_target_unreachable',
                }
        if target_right is not None:
            T_right = np.array(target_right)
            p_right = T_right[0:3, 3]
            if not right.is_point_reachable(p_right, tolerance=workspace_tolerance,
                                             auto_compute=True, num_samples=workspace_samples):
                return {
                    'q_left': q0_left or [0.0] * left.num_dof(),
                    'q_right': q0_right or [0.0] * right.num_dof(),
                    'iters': 0,
                    'success': False,
                    'pos_err_left': 0.0,
                    'ori_err_left': 0.0,
                    'pos_err_right': float('inf'),
                    'ori_err_right': float('inf'),
                    'method': method,
                    'error': 'right_target_unreachable',
                }

    if q0_left is None:
        q0_left = [0.0] * left.num_dof()
    if q0_right is None:
        q0_right = [0.0] * right.num_dof()

    base_kwargs = dict(
        backend=backend,
        method=method,
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )
    # merge user overrides
    base_kwargs.update(solver_kwargs)

    def solve_single(model: RobotModel, target, q0):
        target_list = target.tolist() if hasattr(target, 'tolist') else target
        res = model.ik(target_list, q_initial=q0, **base_kwargs)
        if res.get('success', False):
            return res
        # Fallback 1: enable multi-start if not provided
        if base_kwargs.get('multi_start', 0) == 0:
            fb_kwargs = base_kwargs.copy()
            fb_kwargs['multi_start'] = 2
            fb_kwargs.setdefault('multi_noise', 0.3)
            res2 = model.ik(target_list, q_initial=q0, **fb_kwargs)
            if res2.get('success', False):
                res2['fallback'] = 'multi_start'
                return res2
            res = res2  # keep worst so far
        # Fallback 2: switch method if available
        alt_method = 'dls' if base_kwargs.get('method', method) == 'pinv' else 'pinv'
        if alt_method != base_kwargs.get('method'):
            fb_kwargs2 = base_kwargs.copy()
            fb_kwargs2['method'] = alt_method
            res3 = model.ik(target_list, q_initial=q0, **fb_kwargs2)
            if res3.get('success', False):
                res3['fallback'] = f"method_switch({alt_method})"
                return res3
        return res

    if target_left is not None:
        res_left = solve_single(left, target_left, q0_left)
    else:
        res_left = {'q': q0_left, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}

    if target_right is not None:
        res_right = solve_single(right, target_right, q0_right)
    else:
        res_right = {'q': q0_right, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}

    # Compose final metrics (reuse res fields; also recompute with _pose_error for consistency)
    q_left_final = res_left['q']
    q_right_final = res_right['q']

    if target_left is not None:
        T_left_cur = left.fk(q_left_final, backend='numpy', return_end=True)
        if hasattr(T_left_cur, 'detach'):
            T_left_cur = T_left_cur.detach().cpu().numpy()
        err_left_vec = _pose_error(T_left_cur, np.array(target_left))
        pos_err_left = float(np.linalg.norm(err_left_vec[:3]))
        ori_err_left = float(np.linalg.norm(err_left_vec[3:]))
    else:
        pos_err_left = 0.0
        ori_err_left = 0.0

    if target_right is not None:
        T_right_cur = right.fk(q_right_final, backend='numpy', return_end=True)
        if hasattr(T_right_cur, 'detach'):
            T_right_cur = T_right_cur.detach().cpu().numpy()
        err_right_vec = _pose_error(T_right_cur, np.array(target_right))
        pos_err_right = float(np.linalg.norm(err_right_vec[:3]))
        ori_err_right = float(np.linalg.norm(err_right_vec[3:]))
    else:
        pos_err_right = 0.0
        ori_err_right = 0.0

    success = res_left['success'] and res_right['success']
    iters = max(res_left['iters'], res_right['iters'])
    return {
        'q_left': q_left_final,
        'q_right': q_right_final,
        'iters': iters,
        'success': success,
        'pos_err_left': pos_err_left,
        'ori_err_left': ori_err_left,
        'pos_err_right': pos_err_right,
        'ori_err_right': ori_err_right,
        'method': method,
    }


# =====================================================================
# Phase 2.1: Relative Pose Tasks & Cooperative Control
# =====================================================================

def relative_pose_error(T_left: np.ndarray, T_right: np.ndarray, T_rel_desired: np.ndarray) -> np.ndarray:
    """Compute 6D error for relative pose constraint.
    
    Constraint: T_rel = T_left^{-1} @ T_right should equal T_rel_desired
    Error is computed in the left arm's coordinate frame.
    
    :param T_left: Left arm end-effector pose (4x4)
    :param T_right: Right arm end-effector pose (4x4)
    :param T_rel_desired: Desired relative transformation (4x4)
    :return: 6D error [e_pos (3), e_ori (3)] in left arm frame
    """
    # Current relative transform
    T_left_inv = np.linalg.inv(T_left)
    T_rel_current = T_left_inv @ T_right
    
    # Position error (in left arm frame)
    p_rel_current = T_rel_current[0:3, 3]
    p_rel_desired = T_rel_desired[0:3, 3]
    e_pos = p_rel_desired - p_rel_current
    
    # Orientation error (axis-angle in left arm frame)
    R_rel_current = T_rel_current[0:3, 0:3]
    R_rel_desired = T_rel_desired[0:3, 0:3]
    R_error = R_rel_desired @ R_rel_current.T
    
    # Convert rotation matrix to axis-angle
    axis, angle = matrix_to_axis_angle(R_error)
    e_ori = axis * angle
    
    return np.concatenate([e_pos, e_ori])


def adjoint_matrix(T: np.ndarray) -> np.ndarray:
    """Compute 6x6 adjoint matrix for SE(3) transformation.
    
    Ad(T) maps twist coordinates: V' = Ad(T) @ V
    Structure: Ad = [[R, 0], [p_skew @ R, R]]
    
    :param T: 4x4 homogeneous transformation matrix
    :return: 6x6 adjoint matrix
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    
    # Skew-symmetric matrix from position vector
    p_skew = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = R
    Ad[3:6, 3:6] = R
    Ad[3:6, 0:3] = p_skew @ R
    
    return Ad


def relative_jacobian(
    left: RobotModel,
    right: RobotModel,
    q_left: Sequence[float],
    q_right: Sequence[float],
    backend: str = 'numpy'
) -> np.ndarray:
    """Construct Jacobian for relative pose task (numerical differentiation).
    
    Maps joint velocities to relative pose velocity:
        ė_rel = J_rel @ q̇   where q̇ = [q̇_left; q̇_right]
    
    For stability and correctness, this uses numerical differentiation
    rather than analytical adjoint formulation.
    
    :param left: Left arm robot model
    :param right: Right arm robot model
    :param q_left: Left arm joint configuration
    :param q_right: Right arm joint configuration
    :param backend: Backend for computation ('numpy' or 'auto')
    :return: 6 x (nL + nR) relative Jacobian matrix
    """
    eps = 1e-7
    
    def rel_pose_6d(qL, qR):
        """Compute relative pose as 6D vector (position + axis-angle)."""
        TL = left.fk(qL, backend=backend, return_end=True)
        TR = right.fk(qR, backend=backend, return_end=True)
        
        # Ensure numpy
        if hasattr(TL, 'detach'):
            TL = TL.detach().cpu().numpy()
            TR = TR.detach().cpu().numpy()
        else:
            TL = np.array(TL)
            TR = np.array(TR)
        
        T_rel = np.linalg.inv(TL) @ TR
        
        p_rel = T_rel[0:3, 3]
        R_rel = T_rel[0:3, 0:3]
        
        axis, angle = matrix_to_axis_angle(R_rel)
        ori_rel = axis * angle
        
        return np.concatenate([p_rel, ori_rel])
    
    # Base configuration
    q_left_np = np.array(q_left, dtype=float)
    q_right_np = np.array(q_right, dtype=float)
    q_combined = np.concatenate([q_left_np, q_right_np])
    
    nL = len(q_left_np)
    nR = len(q_right_np)
    
    # Numerical Jacobian
    J_rel = np.zeros((6, nL + nR))
    p0 = rel_pose_6d(q_left_np, q_right_np)
    
    for i in range(nL + nR):
        q_pert = q_combined.copy()
        q_pert[i] += eps
        
        qL_p = q_pert[:nL]
        qR_p = q_pert[nL:]
        
        p_plus = rel_pose_6d(qL_p, qR_p)
        J_rel[:, i] = (p_plus - p0) / eps
    
    return J_rel


@dataclass
class Task:
    """Task descriptor for weighted/hierarchical dual-arm IK.
    
    :param type: Task type - 'absolute_left', 'absolute_right', or 'relative'
    :param target: Target transformation (4x4 matrix)
    :param weight: Task weight for weighted composition (default: 1.0)
    :param priority: Priority level for hierarchical solving (0=highest, default: 0)
    :param row_mask: Optional boolean mask to select task dimensions [6,] (default: all True)
                     Example: [1,1,1,0,0,0] constrains only position (xyz)
    """
    type: str  # 'absolute_left' | 'absolute_right' | 'relative'
    target: np.ndarray  # 4x4 transformation matrix
    weight: float = 1.0
    priority: int = 0
    row_mask: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate task configuration."""
        if self.type not in ['absolute_left', 'absolute_right', 'relative']:
            raise ValueError(f"Invalid task type: {self.type}")
        if self.target.shape != (4, 4):
            raise ValueError(f"Target must be 4x4, got {self.target.shape}")
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}")
        if self.row_mask is not None:
            mask = np.array(self.row_mask, dtype=bool)
            if mask.shape != (6,):
                raise ValueError(f"row_mask must be (6,), got {mask.shape}")


def dual_ik_weighted(
    left: RobotModel,
    right: RobotModel,
    tasks: List[Task],
    q0_left: Sequence[float],
    q0_right: Sequence[float],
    max_iters: int = 100,
    tol: float = 1e-3,
    damping: float = 1e-3,
    step_limit: float = 0.2,
    backend: str = 'numpy',
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solve dual-arm IK with weighted task composition.
    
    Combines multiple tasks (absolute + relative) using weighted least-squares:
        min Σ wᵢ ||eᵢ||²   subject to damped least squares
    
    :param left: Left arm robot model
    :param right: Right arm robot model
    :param tasks: List of Task objects defining objectives
    :param q0_left: Initial left arm configuration
    :param q0_right: Initial right arm configuration
    :param max_iters: Maximum iterations
    :param tol: Convergence tolerance (total weighted error norm)
    :param damping: Damping factor for DLS
    :param step_limit: Maximum joint step per iteration (rad)
    :param backend: Computation backend
    :param verbose: Print iteration details
    :return: Dict with 'q_left', 'q_right', 'success', 'iters', 'residual'
    """
    # Initialize joint vector
    q_left = np.array(q0_left, dtype=float)
    q_right = np.array(q0_right, dtype=float)
    q = np.concatenate([q_left, q_right])
    
    nL = len(q_left)
    nR = len(q_right)
    
    for it in range(max_iters):
        # Build combined error vector and Jacobian
        errors = []
        jacobians = []
        
        for task in tasks:
            # Compute task-specific error and Jacobian
            if task.type == 'absolute_left':
                T_current = left.fk(q[:nL], backend=backend, return_end=True)
                if hasattr(T_current, 'detach'):
                    T_current = T_current.detach().cpu().numpy()
                else:
                    T_current = np.array(T_current)
                
                e = _pose_error(T_current, task.target)
                J = single_jacobian(left, q[:nL], backend=backend)
                if hasattr(J, 'detach'):
                    J = J.detach().cpu().numpy()
                else:
                    J = np.array(J)
                
                # Pad with zeros for right arm
                J_padded = np.hstack([J, np.zeros((6, nR))])
            
            elif task.type == 'absolute_right':
                T_current = right.fk(q[nL:], backend=backend, return_end=True)
                if hasattr(T_current, 'detach'):
                    T_current = T_current.detach().cpu().numpy()
                else:
                    T_current = np.array(T_current)
                
                e = _pose_error(T_current, task.target)
                J = single_jacobian(right, q[nL:], backend=backend)
                if hasattr(J, 'detach'):
                    J = J.detach().cpu().numpy()
                else:
                    J = np.array(J)
                
                # Pad with zeros for left arm
                J_padded = np.hstack([np.zeros((6, nL)), J])
            
            elif task.type == 'relative':
                T_left = left.fk(q[:nL], backend=backend, return_end=True)
                T_right = right.fk(q[nL:], backend=backend, return_end=True)
                if hasattr(T_left, 'detach'):
                    T_left = T_left.detach().cpu().numpy()
                    T_right = T_right.detach().cpu().numpy()
                else:
                    T_left = np.array(T_left)
                    T_right = np.array(T_right)
                
                e = relative_pose_error(T_left, T_right, task.target)
                J_padded = relative_jacobian(left, right, q[:nL], q[nL:], backend=backend)
            
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            # Apply row mask if specified
            if task.row_mask is not None:
                mask = np.array(task.row_mask, dtype=bool)
                e = e[mask]
                J_padded = J_padded[mask, :]
            
            # Apply weighting (sqrt for proper least-squares)
            sqrt_w = np.sqrt(task.weight)
            errors.append(sqrt_w * e)
            jacobians.append(sqrt_w * J_padded)
        
        # Stack all tasks
        e_total = np.concatenate(errors)
        J_total = np.vstack(jacobians)
        
        # Damped least squares solve
        JT = J_total.T
        A = J_total @ JT + (damping ** 2) * np.eye(len(e_total))
        dq = JT @ np.linalg.solve(A, e_total)
        
        # Step limiting
        dq = np.clip(dq, -step_limit, step_limit)
        q += dq
        
        # Compute residual
        residual = np.linalg.norm(e_total)
        
        if verbose:
            print(f"Iter {it+1}: residual = {residual:.6f}, |dq| = {np.linalg.norm(dq):.6f}")
        
        # Convergence check
        if residual < tol:
            if verbose:
                print(f"✓ Converged in {it+1} iterations")
            break
    
    return {
        'q_left': q[:nL].tolist(),
        'q_right': q[nL:].tolist(),
        'success': residual < tol,
        'iters': it + 1,
        'residual': float(residual),
    }


# =====================================================================
# Phase 2.2: Hierarchical Task Priority (Nullspace Projection)
# =====================================================================

def nullspace_projector(J: np.ndarray, damping: float = 1e-6) -> np.ndarray:
    """Compute nullspace projection matrix for a Jacobian.
    
    Projects vectors into the nullspace of J:
        N = I - J⁺ @ J
    
    where J⁺ is the Moore-Penrose pseudoinverse.
    
    :param J: Jacobian matrix (m × n)
    :param damping: Regularization for pseudoinverse (default: 1e-6)
    :return: Nullspace projector (n × n)
    """
    n = J.shape[1]
    
    # Compute pseudoinverse with damping
    J_pinv = np.linalg.pinv(J, rcond=damping)
    
    # Nullspace projector
    N = np.eye(n) - J_pinv @ J
    
    return N


def dual_ik_hierarchical(
    left: RobotModel,
    right: RobotModel,
    task_groups: List[List[Task]],
    q0_left: Sequence[float],
    q0_right: Sequence[float],
    max_iters: int = 150,
    tol_primary: float = 1e-3,
    tol_secondary: float = 5e-3,
    damping: float = 1e-3,
    step_limit: float = 0.15,
    backend: str = 'numpy',
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solve dual-arm IK with hierarchical task priorities using nullspace projection.
    
    Implements strict priority: higher priority tasks are satisfied exactly,
    lower priority tasks are optimized in the nullspace of higher tasks.
    
    Algorithm:
        1. Solve primary task: Δq₁ = J₁⁺ e₁
        2. Compute nullspace: N₁ = I - J₁⁺ J₁
        3. Project secondary task: Δq₂ = N₁ (J₂N₁)⁺ (e₂ - J₂ Δq₁)
        4. Repeat for additional levels
    
    :param left: Left arm robot model
    :param right: Right arm robot model
    :param task_groups: List of task lists, ordered by priority (0=highest)
                        Example: [[primary_tasks], [secondary_tasks], [tertiary_tasks]]
    :param q0_left: Initial left arm configuration
    :param q0_right: Initial right arm configuration
    :param max_iters: Maximum iterations
    :param tol_primary: Convergence tolerance for primary tasks
    :param tol_secondary: Convergence tolerance for secondary tasks
    :param damping: Damping factor for pseudoinverse
    :param step_limit: Maximum joint step per iteration (rad)
    :param backend: Computation backend
    :param verbose: Print iteration details
    :return: Dict with 'q_left', 'q_right', 'success', 'iters', 'residuals_by_priority'
    """
    # Initialize joint vector
    q_left = np.array(q0_left, dtype=float)
    q_right = np.array(q0_right, dtype=float)
    q = np.concatenate([q_left, q_right])
    
    nL = len(q_left)
    nR = len(q_right)
    n_total = nL + nR
    
    num_levels = len(task_groups)
    
    for it in range(max_iters):
        dq_total = np.zeros(n_total)
        N = np.eye(n_total)  # Initial nullspace = full space
        
        residuals = []
        
        for priority_level, tasks in enumerate(task_groups):
            # Build error vector and Jacobian for this priority level
            errors_level = []
            jacobians_level = []
            
            for task in tasks:
                # Compute task-specific error and Jacobian
                if task.type == 'absolute_left':
                    T_current = left.fk(q[:nL], backend=backend, return_end=True)
                    if hasattr(T_current, 'detach'):
                        T_current = T_current.detach().cpu().numpy()
                    else:
                        T_current = np.array(T_current)
                    
                    e = _pose_error(T_current, task.target)
                    J = single_jacobian(left, q[:nL], backend=backend)
                    if hasattr(J, 'detach'):
                        J = J.detach().cpu().numpy()
                    else:
                        J = np.array(J)
                    
                    J_padded = np.hstack([J, np.zeros((6, nR))])
                
                elif task.type == 'absolute_right':
                    T_current = right.fk(q[nL:], backend=backend, return_end=True)
                    if hasattr(T_current, 'detach'):
                        T_current = T_current.detach().cpu().numpy()
                    else:
                        T_current = np.array(T_current)
                    
                    e = _pose_error(T_current, task.target)
                    J = single_jacobian(right, q[nL:], backend=backend)
                    if hasattr(J, 'detach'):
                        J = J.detach().cpu().numpy()
                    else:
                        J = np.array(J)
                    
                    J_padded = np.hstack([np.zeros((6, nL)), J])
                
                elif task.type == 'relative':
                    T_left = left.fk(q[:nL], backend=backend, return_end=True)
                    T_right = right.fk(q[nL:], backend=backend, return_end=True)
                    if hasattr(T_left, 'detach'):
                        T_left = T_left.detach().cpu().numpy()
                        T_right = T_right.detach().cpu().numpy()
                    else:
                        T_left = np.array(T_left)
                        T_right = np.array(T_right)
                    
                    e = relative_pose_error(T_left, T_right, task.target)
                    J_padded = relative_jacobian(left, right, q[:nL], q[nL:], backend=backend)
                
                else:
                    raise ValueError(f"Unknown task type: {task.type}")
                
                # Apply row mask if specified
                if task.row_mask is not None:
                    mask = np.array(task.row_mask, dtype=bool)
                    e = e[mask]
                    J_padded = J_padded[mask, :]
                
                errors_level.append(e)
                jacobians_level.append(J_padded)
            
            # Stack all tasks for this priority level
            e_level = np.concatenate(errors_level)
            J_level = np.vstack(jacobians_level)
            
            residuals.append(float(np.linalg.norm(e_level)))
            
            if priority_level == 0:
                # Primary task: use damped pseudoinverse (Moore-Penrose) to get
                # a minimal-norm regularized solution. This keeps the primary
                # solution consistent with the nullspace projector computed
                # from the same pseudoinverse formulation.
                J_pinv = np.linalg.pinv(J_level, rcond=damping)
                dq_prim = J_pinv @ e_level
                dq_total += dq_prim

                # Compute nullspace of primary task (consistent with J_pinv)
                N = np.eye(n_total) - J_pinv @ J_level
            else:
                # Secondary/tertiary tasks: project into nullspace
                J_projected = J_level @ N
                
                # Residual error after accounting for primary motion
                e_residual = e_level - J_level @ dq_total
                
                # Solve in nullspace using pseudoinverse of the projected Jacobian.
                # Using pinv here keeps the solution consistent and tends to
                # avoid numerical leakage that can degrade higher-priority tasks.
                # Note: J_projected may be rank-deficient; pinv handles that.
                J_proj_pinv = np.linalg.pinv(J_projected, rcond=damping)
                dq_sec = J_proj_pinv @ e_residual

                # Project back to ensure it stays in the higher-priority nullspace
                dq_sec_projected = N @ dq_sec
                dq_total += dq_sec_projected

                # Update nullspace (compound projection)
                N = N @ (np.eye(n_total) - J_proj_pinv @ J_projected)
        
        # Apply step limiting
        dq_total = np.clip(dq_total, -step_limit, step_limit)
        q += dq_total
        
        if verbose:
            residual_str = ", ".join([f"L{i}={r:.4f}" for i, r in enumerate(residuals)])
            print(f"Iter {it+1}: {residual_str}, |dq|={np.linalg.norm(dq_total):.4f}")
        
        # Convergence check
        primary_converged = residuals[0] < tol_primary
        secondary_converged = all(r < tol_secondary for r in residuals[1:]) if len(residuals) > 1 else True
        
        # Stop if primary converged AND (no secondary tasks OR secondary also converged OR dq very small)
        if primary_converged:
            if len(residuals) == 1:
                # Only primary task - done
                if verbose:
                    print(f"✓ Primary task converged in {it+1} iterations")
                break
            elif secondary_converged:
                # All tasks converged
                if verbose:
                    print(f"✓ All tasks converged in {it+1} iterations")
                break
            elif np.linalg.norm(dq_total) < 1e-4:
                # Motion stopped (likely at local minimum for secondary)
                if verbose:
                    print(f"✓ Primary converged, secondary at local optimum ({it+1} iterations)")
                break
    
    # Final convergence status
    primary_converged = residuals[0] < tol_primary
    secondary_converged = all(r < tol_secondary for r in residuals[1:]) if len(residuals) > 1 else True
    
    return {
        'q_left': q[:nL].tolist(),
        'q_right': q[nL:].tolist(),
        'success': residuals[0] < tol_primary,
        'secondary_success': secondary_converged,
        'iters': it + 1,
        'residuals': residuals,
        'primary_residual': residuals[0] if residuals else float('inf'),
    }


__all__ = [
    'dual_fk',
    'dual_block_jacobian',
    'dual_ik',
    'relative_pose_error',
    'relative_jacobian',
    'Task',
    'dual_ik_weighted',
    'dual_ik_hierarchical',
    'nullspace_projector',
]
