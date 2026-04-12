"""Multi-chain DLS IK via native Eigen loop (unified configuration space).

Requires built ``_multichain_ik_core`` extension. Used from ``inverse_kinematics``
when backend is ``cpp`` and ``method`` is ``dls``.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from robocore.kinematics.fk_utils.fk_solver_cpp import chain_arrays_global_config

from . import _multichain_ik_core


def solve_multichain_ik_cpp(
    model: Any,
    targets: Dict[str, np.ndarray],
    end_links: Sequence[str],
    q0: Optional[Sequence[float] | np.ndarray] = None,
    *,
    base_link: Optional[str] = None,
    method: str = "dls",
    max_iters: int = 200,
    pos_tol: float = 1e-3,
    ori_tol: float = 1e-3,
    damping: float = 1e-3,
    step_limit: float = 0.2,
    **_: Any,
) -> Dict[str, Any]:
    """Run stacked-task DLS IK in C++ (same iteration as Python ``_solve_multichain_ik``).

    :param model: RobotModel instance
    :param targets: end_link -> (4, 4) target pose
    :param end_links: ordered list of end links (must match stacking order)
    :param q0: initial full configuration (length num_dof)
    :param base_link: base link (default model.base_link)
    :param method: only ``dls`` supported
    :param max_iters: maximum iterations
    :param pos_tol: convergence tolerance (combined with ori on error norm)
    :param ori_tol: added to pos_tol for convergence check (Python compatibility)
    :param damping: DLS damping lambda (A += lambda^2 I on J J^T)
    :param step_limit: max norm of dq per iteration
    :return: dict with q, success, iters, pos_err, ori_err
    """
    if str(method).lower() != "dls":
        raise ValueError("multichain C++ IK only supports method='dls'")

    base = base_link or model.base_link
    nq = model.num_dof

    if q0 is None:
        q0_arr = np.zeros(nq, dtype=np.float64)
    else:
        q0_arr = np.asarray(q0, dtype=np.float64).reshape(-1)
        if q0_arr.shape[0] != nq:
            raise ValueError(f"Expected q0 of length {nq}, got {len(q0_arr)}")

    K = len(end_links)
    targets_arr = np.zeros((K, 4, 4), dtype=np.float64)
    counts: list[int] = []
    types_parts: list[np.ndarray] = []
    qidx_parts: list[np.ndarray] = []
    T_parts: list[np.ndarray] = []
    ax_parts: list[np.ndarray] = []

    for k, end_link in enumerate(end_links):
        if end_link not in targets:
            raise KeyError(f"targets missing key {end_link!r}")
        targets_arr[k] = np.asarray(targets[end_link], dtype=np.float64).reshape(4, 4)
        tys, qi, Tr, ax = chain_arrays_global_config(model, base, end_link)
        counts.append(int(tys.shape[0]))
        types_parts.append(tys)
        qidx_parts.append(qi)
        T_parts.append(Tr)
        ax_parts.append(ax)

    types_flat = np.concatenate(types_parts, axis=0)
    q_idx_flat = np.concatenate(qidx_parts, axis=0)
    T_flat = np.concatenate(T_parts, axis=0)
    axes_flat = np.concatenate(ax_parts, axis=0)
    joint_counts = np.array(counts, dtype=np.int32)

    q_lo = np.full(nq, -1e300, dtype=np.float64)
    q_hi = np.full(nq, 1e300, dtype=np.float64)
    has_lo = np.zeros(nq, dtype=np.int32)
    has_hi = np.zeros(nq, dtype=np.int32)
    if hasattr(model, "dof_limits_min") and hasattr(model, "dof_limits_max"):
        q_lo = np.asarray(model.dof_limits_min, dtype=np.float64).reshape(-1)
        q_hi = np.asarray(model.dof_limits_max, dtype=np.float64).reshape(-1)
        if q_lo.shape[0] == nq and q_hi.shape[0] == nq:
            has_lo[:] = 1
            has_hi[:] = 1

    out = _multichain_ik_core.solve_multichain_dls(
        q0_arr,
        targets_arr,
        joint_counts,
        types_flat,
        q_idx_flat,
        T_flat,
        axes_flat,
        int(nq),
        int(max_iters),
        float(pos_tol),
        float(ori_tol),
        float(damping),
        float(step_limit),
        q_lo,
        q_hi,
        has_lo,
        has_hi,
    )
    return {
        "q": list(out["q"]),
        "success": bool(out["success"]),
        "iters": int(out["iters"]),
        "pos_err": float(out["pos_err"]),
        "ori_err": float(out["ori_err"]),
    }


__all__ = ["solve_multichain_ik_cpp"]
