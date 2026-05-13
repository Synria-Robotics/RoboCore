"""Dual-arm cooperative kinematics.

Author: Synria Robotics Team
License: MIT
"""
from __future__ import annotations
from robocore.utils.backend import get_backend, set_backend
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
    return_end: bool = True,
    mode: str = 'indep',
    device: Any | None = None,
    dtype: Any | None = None,
):
    """Compute bimanual forward kinematics.
    
    DEPRECATED: Use robot.fk(q_full, base_link=..., end_link=...) with unified config space instead.
    This function is kept for backward compatibility with existing examples.
    
    Supports both single and batch processing. Inputs are automatically detected:
    - Single: [n] -> single 4x4 matrix
    - Batch: [B, n] -> [B, 4, 4] array
    
    :param left_model: Left arm RobotModel (or single RobotModel with left end_link)
    :param right_model: Right arm RobotModel (or single RobotModel with right end_link)
    :param q_left: Left joint configuration(s) - [n] or [B, n], or full config [num_dof] if using unified space
    :param q_right: Right joint configuration(s) - [n] or [B, n], or full config [num_dof] if using unified space
    :param return_end: Return only end-effector poses
    :param mode: 'indep'|'relative'|'mirror'
    :param device: Torch device (only for torch backend)
    :param dtype: Torch dtype (only for torch backend)
    :return: {'left': T_left, 'right': T_right} or with additional fields
    """
    b = get_backend()
    if mode == 'indep':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiIndependentFKSolverNumpy

            solver = BiIndependentFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiIndependentFKSolverTorch
            import torch  # type: ignore

            solver = BiIndependentFKSolverTorch(left_model, right_model)
            if dtype is None:
                dtype = torch.float64
            return solver.fk(q_left, q_right, return_end=return_end, device=device, dtype=dtype)
        elif b == 'cpp':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_cpp import BiIndependentFKSolverCpp

            solver = BiIndependentFKSolverCpp(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif mode == 'relative':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiRelativeFKSolverNumpy

            solver = BiRelativeFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiRelativeFKSolverTorch
            import torch  # type: ignore

            solver = BiRelativeFKSolverTorch(left_model, right_model)
            if dtype is None:
                dtype = torch.float64
            return solver.fk(q_left, q_right, return_end=return_end, device=device, dtype=dtype)
        elif b == 'cpp':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_cpp import BiRelativeFKSolverCpp

            solver = BiRelativeFKSolverCpp(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif mode == 'mirror':
        if b == 'numpy':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_numpy import BiMirrorFKSolverNumpy

            solver = BiMirrorFKSolverNumpy(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        elif b == 'torch':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_torch import BiMirrorFKSolverTorch
            import torch  # type: ignore

            solver = BiMirrorFKSolverTorch(left_model, right_model)
            if dtype is None:
                dtype = torch.float64
            return solver.fk(q_left, q_right, return_end=return_end, device=device, dtype=dtype)
        elif b == 'cpp':
            from robocore.kinematics.fk_utils.bimanual_fk_solver_cpp import BiMirrorFKSolverCpp

            solver = BiMirrorFKSolverCpp(left_model, right_model)
            return solver.fk(q_left, q_right, return_end=return_end)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
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
    method: str = 'dls',
    coordination: str = 'indep',
    backend: str | None = None,
    # Initial guess parameters (new system)
    num_initial_guesses: int = 1,
    initial_guess_strategy: str = 'random',
    initial_guess_scale: float = 1.0,
    random_seed: Optional[int] = None,
    # Optional advanced params
    T_rel_grasp=None,
    T_left_initial=None,
    T_right_initial=None,
    return_all: bool = False,
    # Torch specific
    torch_device: Any | None = None,
    torch_dtype: Any | None = None,
    **solver_kwargs,
):
    """Compute bimanual inverse kinematics.
    
    Supports both single and batch processing. Inputs are automatically detected:
    - Single: 4x4 matrix -> single result
    - Batch: [B, 4, 4] -> list of results
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param target_left: Left target pose(s) - 4x4 or [B, 4, 4]
    :param target_right: Right target pose(s) - 4x4 or [B, 4, 4]
    :param q0_left: Initial left configuration (optional, used as base for strategies)
    :param q0_right: Initial right configuration (optional, used as base for strategies)
    :param method: 'dls'|'pinv'|'transpose'
    :param coordination: 'indep'|'relative_pose'|'relative_pos'|'relative_ori'|'mirror'
    :param backend: Optional backend override ('numpy', 'torch', or 'cpp'). If provided,
        this is forwarded to the global backend manager so that IK solvers run
        on the requested backend (matches usage in higher-level helpers such as
        ``BimanualRobotModel.ik`` and interactive demos).
    :param num_initial_guesses: Number of initial guesses to try (default: 1)
    :param initial_guess_strategy: Strategy - 'zero'|'random'|'sobol'|'latin'|'center'|'uniform'
    :param initial_guess_scale: Scale factor for joint limits (0.0 to 1.0)
    :param random_seed: Seed for reproducibility
    :param T_rel_grasp: Relative grasp transform (for relative modes)
    :param T_left_initial: Left initial reference pose (for mirror mode)
    :param T_right_initial: Right initial reference pose (for mirror mode)
    :param return_all: Return all candidates if available
    :param torch_device: Torch device (only for torch backend)
    :param torch_dtype: Torch dtype (only for torch backend)
    :return: Result dict
    """
    # Allow callers (e.g. InteractiveDualArmIK, BimanualRobotModel.ik) to
    # explicitly select the backend. If backend is None we keep the existing
    # global backend setting.
    if backend is not None:
        set_backend(backend)

    b = get_backend()
    # Prepare IK kwargs with new initial guess system
    ik_kwargs = {
        'method': method,
        'num_initial_guesses': num_initial_guesses,
        'initial_guess_strategy': initial_guess_strategy,
        'initial_guess_scale': initial_guess_scale,
        'random_seed': random_seed,
        **solver_kwargs,
    }
    if torch_device is not None:
        ik_kwargs['torch_device'] = torch_device
    if torch_dtype is not None:
        ik_kwargs['torch_dtype'] = torch_dtype

    if coordination == 'indep':
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiIndependentIKSolverNumpy

            solver = BiIndependentIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiIndependentIKSolverTorch
            solver = BiIndependentIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)
        elif b == 'cpp':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiIndependentIKSolverNumpy

            solver = BiIndependentIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif coordination in ('relative_pose', 'relative_pos', 'relative_ori'):
        constraint_type = 'pose' if coordination == 'relative_pose' else ('position' if coordination == 'relative_pos' else 'orientation')
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiRelativeIKSolverNumpy
            solver = BiRelativeIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                constraint_type=constraint_type, T_rel_grasp=T_rel_grasp, **ik_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiRelativeIKSolverTorch
            solver = BiRelativeIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                constraint_type=constraint_type, T_rel_grasp=T_rel_grasp, **ik_kwargs)
        elif b == 'cpp':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiRelativeIKSolverNumpy
            solver = BiRelativeIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                constraint_type=constraint_type, T_rel_grasp=T_rel_grasp, **ik_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif coordination == 'mirror':
        if b == 'numpy':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiMirrorIKSolverNumpy
            solver = BiMirrorIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                T_left_initial=T_left_initial, T_right_initial=T_right_initial, **ik_kwargs)
        elif b == 'torch':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_torch import BiMirrorIKSolverTorch
            solver = BiMirrorIKSolverTorch(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                T_left_initial=T_left_initial, T_right_initial=T_right_initial, **ik_kwargs)
        elif b == 'cpp':
            from robocore.kinematics.ik_utils.bimanual_ik_solver_numpy import BiMirrorIKSolverNumpy
            solver = BiMirrorIKSolverNumpy(left_model, right_model)
            return solver.solve(target_left, target_right, q0_left, q0_right,
                                T_left_initial=T_left_initial, T_right_initial=T_right_initial, **ik_kwargs)
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    else:
        raise ValueError("Unknown coordination mode")


def bimanual_jacobian(
    left_model,
    right_model,
    q_left,
    q_right,
    *,
    mode: str = 'indep',
    row_mask=None,
    device: Any | None = None,
    dtype: Any | None = None,
):
    """Compute bimanual Jacobian matrix.
    
    Supports both single and batch processing. Inputs are automatically detected:
    - Single: [n] -> [12, nL+nR] or [6, nL+nR] matrix (depending on mode)
    - Batch: [B, n] -> [B, 12, nL+nR] or [B, 6, nL+nR] array
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param q_left: Left joint configuration(s) - [n] or [B, n]
    :param q_right: Right joint configuration(s) - [n] or [B, n]
    :param mode: 'indep'|'relative'|'mirror'
    :param row_mask: Optional row selection mask
    :param device: Torch device (only for torch backend)
    :param dtype: Torch dtype (only for torch backend)
    :return: Jacobian matrix - [12, nL+nR] or [6, nL+nR] (single) or [B, 12, nL+nR] / [B, 6, nL+nR] (batch)
    """
    b = get_backend()
    if mode == 'indep':
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiIndependentJacobianSolverNumpy
            import numpy as np

            solver = BiIndependentJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiIndependentJacobianSolverTorch
            import torch  # type: ignore

            solver = BiIndependentJacobianSolverTorch(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = torch.tensor([bool(m) for m in row_mask], dtype=torch.bool)
                J = J[mask, :]
            return J
        elif b == 'cpp':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiIndependentJacobianSolverNumpy
            import numpy as np

            solver = BiIndependentJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif mode == 'relative':
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiRelativeJacobianSolverNumpy
            import numpy as np

            solver = BiRelativeJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiRelativeJacobianSolverTorch
            import torch  # type: ignore

            solver = BiRelativeJacobianSolverTorch(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = torch.tensor([bool(m) for m in row_mask], dtype=torch.bool)
                J = J[mask, :]
            return J
        elif b == 'cpp':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiRelativeJacobianSolverNumpy
            import numpy as np

            solver = BiRelativeJacobianSolverNumpy(left_model, right_model)
            J = solver.compute(q_left, q_right)
            if row_mask is not None:
                mask = np.array([bool(m) for m in row_mask])
                J = J[mask, :]
            return J
        else:
            raise ValueError("Unsupported backend, expected 'numpy'|'torch'|'cpp'")
    elif mode == 'mirror':
        raise NotImplementedError("Jacobian mode not implemented yet: mirror")


# ---------------------------------------------------------------------------
# Re-export Task so callers can do: from robocore.kinematics.bimanual import Task
# ---------------------------------------------------------------------------
from robocore.kinematics.task import Task  # noqa: E402


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def dual_fk(left_model, right_model, q_left, q_right, **kwargs):
    """Positional-argument alias for :func:`bimanual_forward_kinematics`."""
    return bimanual_forward_kinematics(left_model, right_model, q_left, q_right, **kwargs)


def dual_ik(left_model, right_model, target_left, target_right, **kwargs):
    """Positional-argument alias for :func:`bimanual_inverse_kinematics`.

    Accepts ``check_workspace`` for compatibility with older call sites but
    silently discards it (not a parameter of the underlying solver).
    """
    kwargs.pop('check_workspace', None)
    return bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left,
        target_right=target_right,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Screw-theory helpers (Phase 2.1)
# ---------------------------------------------------------------------------

def adjoint_matrix(T):
    """Compute the 6×6 spatial adjoint matrix Ad(T) of an SE(3) transform.

    .. math::
        \\mathrm{Ad}(T) = \\begin{bmatrix} R & 0 \\\\ [p]_\\times R & R \\end{bmatrix}

    where ``[p]_×`` is the skew-symmetric matrix of the translation ``p``.

    :param T: 4×4 homogeneous transformation matrix
    :return: 6×6 adjoint matrix (numpy array)
    """
    R = np.asarray(T[:3, :3], dtype=float)
    p = np.asarray(T[:3, 3], dtype=float)
    px = np.array([
        [ 0.0,  -p[2],  p[1]],
        [ p[2],  0.0,  -p[0]],
        [-p[1],  p[0],  0.0 ],
    ])
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = px @ R
    Ad[3:, 3:] = R
    return Ad


# ---------------------------------------------------------------------------
# Null-space helpers (Phase 2.2)
# ---------------------------------------------------------------------------

def nullspace_projector(J):
    """Compute the null-space projector  **N = I − J⁺J**.

    :param J: Jacobian matrix of shape ``(m, n)``
    :return: Null-space projector of shape ``(n, n)``
    """
    Jpinv = np.linalg.pinv(J)
    n = J.shape[1]
    return np.eye(n) - Jpinv @ J


def dual_ik_hierarchical(left_model, right_model, tasks, q0_left=None, q0_right=None, **kwargs):
    """Hierarchical task-priority IK for dual-arm systems.

    .. note::
        Planned for Phase 2.2.  Not yet implemented.
    """
    raise NotImplementedError(
        "dual_ik_hierarchical is a Phase 2.2 feature and has not been implemented yet."
    )


__all__ = [
    'relative_pose_error',
    'relative_jacobian',
    'bimanual_forward_kinematics',
    'bimanual_jacobian',
    'bimanual_inverse_kinematics',
    # aliases
    'dual_fk',
    'dual_ik',
    # screw-theory helpers
    'adjoint_matrix',
    # null-space helpers
    'nullspace_projector',
    'dual_ik_hierarchical',
    # task abstraction
    'Task',
]
