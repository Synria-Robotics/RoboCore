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
from robocore.utils.backend import get_backend
from typing import Dict, Any, Optional, Sequence, List
from dataclasses import dataclass
import numpy as np

from .jacobian import jacobian as single_jacobian
from .fk import forward_kinematics as single_fk
from .utils import relative_pose_error, relative_jacobian


def bimanual_forward_kinematics(
    left_model,
    right_model,
    q_left,
    q_right,
    *,
    backend: str = 'auto',
    return_end: bool = True,
    mode: str = 'indep',
):
    """
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param q_left: Left joint configuration
    :param q_right: Right joint configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param return_end: Return only end-effector poses
    :param mode: 'indep'|'relative'|'mirror'
    :return: {'left': T_left, 'right': T_right} or with additional fields
    """
    b = get_backend() if backend == 'auto' else backend
    if mode == 'indep':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiIndependentFKSolverNumpy

            solver = BiIndependentFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, backend='numpy', return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiIndependentFKSolverTorch
            import torch  # type: ignore

            solver = BiIndependentFKSolverTorch(left_model, right_model)
            return solver.fk(q_left, q_right, backend='torch', return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif mode == 'relative':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiRelativeFKSolverNumpy

            solver = BiRelativeFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, backend='numpy', return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiRelativeFKSolverTorch
            import torch  # type: ignore

            solver = BiRelativeFKSolverTorch(left_model, right_model)
            return solver.fk(q_left, q_right, backend='torch', return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif mode == 'mirror':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiMirrorFKSolverNumpy

            solver = BiMirrorFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, backend='numpy', return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiMirrorFKSolverTorch
            import torch  # type: ignore

            solver = BiMirrorFKSolverTorch(left_model, right_model)
            return solver.fk(q_left, q_right, backend='torch', return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    else:
        raise ValueError("Unknown mode, expected 'indep'|'relative'|'mirror'")


def bimanual_inverse_kinematics(
    left_model,
    right_model,
    *,
    target_left=None,
    target_right=None,
    q0_left=None,
    q0_right=None,
    backend: str = 'auto',
    method: str = 'dls',
    coordination: str = 'indep',
    # Optional advanced params
    T_rel_grasp=None,
    T_left_initial=None,
    T_right_initial=None,
    return_all: bool = False,
    **solver_kwargs,
):
    """
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param target_left: Left target pose
    :param target_right: Right target pose
    :param q0_left: Initial left configuration
    :param q0_right: Initial right configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param method: 'dls'|'pinv'|'transpose'
    :param coordination: 'indep'|'relative_pose'|'relative_pos'|'relative_ori'|'mirror'
    :param return_all: Return all candidates if available
    :return: Result dict
    """
    b = get_backend() if backend == 'auto' else backend
    if coordination == 'indep':
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiIndependentIKSolverNumpy

            solver = BiIndependentIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, method=method, **solver_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiIndependentIKSolverTorch
            solver = BiIndependentIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, method=method, **solver_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif coordination in ('relative_pose', 'relative_pos', 'relative_ori'):
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiRelativeIKSolverNumpy
            solver = BiRelativeIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, constraint_type=('pose' if coordination == 'relative_pose' else 'position' if coordination == 'relative_pos' else 'orientation'), T_rel_grasp=T_rel_grasp, **solver_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiRelativeIKSolverTorch
            solver = BiRelativeIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, constraint_type=('pose' if coordination == 'relative_pose' else 'position' if coordination == 'relative_pos' else 'orientation'), backend='torch', **solver_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif coordination == 'mirror':
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiMirrorIKSolverNumpy
            solver = BiMirrorIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, T_left_initial=T_left_initial, T_right_initial=T_right_initial, **solver_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiMirrorIKSolverTorch
            solver = BiMirrorIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, backend='torch', **solver_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    else:
        raise ValueError("Unknown coordination mode")


def bimanual_jacobian(
    left_model,
    right_model,
    q_left,
    q_right,
    *,
    backend: str = 'auto',
    mode: str = 'indep',
    row_mask=None,
):
    """
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param q_left: Left joint configuration
    :param q_right: Right joint configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param mode: 'indep'|'relative'|'mirror'
    :param row_mask: Optional row selection mask
    :return: Jacobian matrix
    """
    b = get_backend() if backend == 'auto' else backend
    if mode == 'indep':
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiIndependentJacobianSolverNumpy
            import numpy as np

            solver = BiIndependentJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right, backend='numpy')
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiIndependentJacobianSolverTorch
            import torch  # type: ignore

            solver = BiIndependentJacobianSolverTorch(left_model, right_model)
            J = solver.compute(q_left, q_right, backend='torch')
            if row_mask is not None:
                mask = torch.tensor([bool(m) for m in row_mask], dtype=torch.bool)
                J = J[mask, :]
            return J
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif mode == 'relative':
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiRelativeJacobianSolverNumpy
            import numpy as np

            solver = BiRelativeJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right, backend='numpy')
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiRelativeJacobianSolverTorch
            import torch  # type: ignore

            solver = BiRelativeJacobianSolverTorch(left_model, right_model)
            J = solver.compute(q_left, q_right, backend='torch')
            if row_mask is not None:
                mask = torch.tensor([bool(m) for m in row_mask], dtype=torch.bool)
                J = J[mask, :]
            return J
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    elif mode == 'mirror':
        raise NotImplementedError("Jacobian mode not implemented yet: mirror")
    
    
__all__ = [
    'relative_pose_error',
    'relative_jacobian',
    'bimanual_forward_kinematics',
    'bimanual_jacobian',
    'bimanual_inverse_kinematics',
]
