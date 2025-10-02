"""Unified inverse kinematics high-level API.

Goals of this refactor:
1. Consistent naming across backends (numpy / torch)
2. Single public entry: `inverse_kinematics(model, target_pose, q0, *, backend='auto', method='pinv', ...)`
3. Optional multi-start + backend selection
4. Hide legacy ad-hoc solver objects from user-facing surface

Returned result dict normalized fields:
  q, success, iters, err_norm, pos_err, ori_err, method, backend, jacobian
"""
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List
import numpy as np

from robocore.kinematics.ik_utils.ik_solver_numpy import IKSolverNumPy

_HAS_TORCH = False
try:  # pragma: no cover
    from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
    _HAS_TORCH = True
except Exception:  # noqa: E722
    pass


def _select_backend(backend: str) -> str:
    if backend == 'auto':
        return 'torch' if _HAS_TORCH else 'numpy'
    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unknown backend '{backend}'")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError('Torch backend requested but torch not available')
    return backend


def inverse_kinematics(
    model,
    target_pose: Sequence[Sequence[float]] | np.ndarray,
    q0: Sequence[float],
    *,
    backend: str = 'auto',
    method: str = 'pinv',
    multi_start: int = 0,
    multi_noise: float = 0.3,
    random_seed: Optional[int] = None,
    # torch specific passthrough (ignored by numpy backend)
    torch_device: Any | None = None,
    torch_dtype: Any | None = None,
    **solver_kwargs,
) -> Dict[str, Any]:
    """Unified IK entry.

    :param model: RobotModel
    :param target_pose: 4x4 pose (list or ndarray)
    :param q0: initial configuration
    :param backend: 'auto'|'numpy'|'torch'
    :param method: 'pinv'|'dls'|'transpose'
    :param multi_start: extra random restarts count (0 disable)
    :param multi_noise: gaussian noise scale (radians) for restarts
    :param random_seed: seed for reproducibility
    :param torch_device: specify torch device when backend='torch' (e.g. 'cuda', 'mps', 'cpu')
    :param torch_dtype: specify torch dtype (e.g. torch.float32) when backend='torch'
    :param solver_kwargs: forwarded to concrete solver (e.g. max_iters, pos_tol, ori_tol, ...)
    """
    b = _select_backend(backend)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def _run_once(q_init):
        if b == 'numpy':
            solver = IKSolverNumPy(model, max_iters=solver_kwargs.pop('max_iters', 120), pos_tol=solver_kwargs.pop('pos_tol', 1e-4), ori_tol=solver_kwargs.pop('ori_tol', 1e-4))
            res = solver.solve(np.asarray(target_pose), np.asarray(q_init), method=method, use_analytic_jacobian=solver_kwargs.pop('use_analytic_jacobian', True), **solver_kwargs)
            res['backend'] = 'numpy'
            return res
        # torch backend
        solver = IKSolverTorch(
            model,
            max_iters=solver_kwargs.pop('max_iters', 120),
            pos_tol=solver_kwargs.pop('pos_tol', 1e-4),
            ori_tol=solver_kwargs.pop('ori_tol', 1e-4),
            device=torch_device,
            dtype=torch_dtype,
        )  # type: ignore
        res = solver.solve(np.asarray(target_pose), q_init, method=method, **solver_kwargs)  # type: ignore[arg-type]
        res['backend'] = 'torch'
        return res

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
    return candidates[0]
