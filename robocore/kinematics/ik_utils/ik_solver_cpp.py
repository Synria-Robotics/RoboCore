"""Eigen + pybind11 DLS IK (required native extension).

Requires a built ``_ik_chain_core`` module (Eigen3 + pybind11; see ``pip install -e .`` / ``setup.py``).
Importing this module fails if the extension is missing.

The C++ solver follows the core of ``IKSolverNumPy._solve_single`` (DLS, analytic Jacobian,
adaptive damping/step, joint clamp). Multi-target ``solve`` calls native ``solve_batch`` (C++ loop).
It does **not** implement:

- ``_project_jacobian_for_limits`` (limit-aware Jacobian column zeroing)
- large step / cumulative jump heuristics near limits
- ``pinv`` / ``transpose`` methods, nullspace, ``row_mask``, ``refine``

For difficult limits or near-singular poses, results may differ slightly from full NumPy IK.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from robocore.kinematics.fk_utils.fk_solver_cpp import chain_arrays_from_robot_model
from robocore.kinematics.jacobian_utils.jacobian_solver_cpp import stop_chain_index
from robocore.kinematics.utils import ensure_batch, restore_single

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

from . import _ik_chain_core


class IKSolverCpp:
    """DLS IK using compiled chain. Batch ``solve`` uses native ``solve_batch`` (one pybind call)."""

    def __init__(
        self,
        model: "RobotModel",
        *,
        target_link: str | None = None,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        ori_tol: float = 1e-3,
        min_damping: float = 1e-4,
        max_damping: float = 5e-2,
        base_step: float = 1.0,
    ):
        """Initialize IK solver.

        :param model: robot model.
        :param target_link: optional partial chain (same as Jacobian/FK cpp).
        :param max_iters: maximum iterations.
        :param pos_tol: position tolerance (m).
        :param ori_tol: orientation tolerance (rad).
        :param min_damping: minimum damping.
        :param max_damping: maximum damping.
        :param base_step: base step scale (adaptive_step multiplies relative to this).
        """
        self.model = model
        self.n = model.num_chain_dof
        self._target_link = target_link
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.base_step = base_step

        t, qi, To, ax, n_dof = chain_arrays_from_robot_model(model)
        if n_dof != self.n:
            raise ValueError("internal DOF mismatch")
        stop = np.int32(stop_chain_index(model, target_link))
        lo = np.array(model.chain_joint_limit_min, dtype=np.float64, copy=True)
        hi = np.array(model.chain_joint_limit_max, dtype=np.float64, copy=True)
        hlo = np.array(model.chain_joint_limit_has_lower, dtype=np.int32, copy=True)
        hhi = np.array(model.chain_joint_limit_has_upper, dtype=np.int32, copy=True)
        self._ik = _ik_chain_core.ChainIKDls(
            t, qi, To, ax, np.int32(n_dof), stop, lo, hi, hlo, hhi
        )

    def solve(
        self,
        target_pose: np.ndarray,
        q0: np.ndarray,
        *,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        adaptive_damping: bool = True,
        adaptive_step: bool = False,
        method: str = "dls",
        max_step_norm: float = 0.5,
        target_link: str | None = None,
        **_: Any,
    ) -> Dict[str, Any] | Dict[str, list]:
        """Solve IK (same signature subset as ``IKSolverNumPy.solve``).

        :param target_pose: (4,4) or (B,4,4).
        :param q0: (n,) or (B,n).
        :param pos_weight: position row weights.
        :param ori_weight: orientation row weights.
        :param adaptive_damping: use adaptive damping.
        :param adaptive_step: use adaptive step and ``max_step_norm`` clipping.
        :param method: only ``dls`` supported.
        :param max_step_norm: step clip when ``adaptive_step`` is True.
        :param target_link: must match constructor.
        :return: result dict (single or batch lists like NumPy).
        """
        if method.lower() != "dls":
            raise NotImplementedError("IKSolverCpp only implements method='dls'")
        if target_link is not None and target_link != self._target_link:
            raise ValueError("target_link must match the value passed to IKSolverCpp(...)")

        target_pose = np.asarray(target_pose, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)

        if target_pose.ndim == 2:
            target_pose = target_pose.reshape(1, 4, 4)
        elif target_pose.ndim != 3:
            raise ValueError(f"target_pose must be (4,4) or (B,4,4), got {target_pose.shape}")

        was_single_t = target_pose.shape[0] == 1
        q0, was_single_q = ensure_batch(q0)
        was_single = was_single_t and was_single_q

        if q0.shape[1] != self.n:
            raise ValueError(f"Expected q0 with {self.n} elements, got {q0.shape[1]}")
        if target_pose.shape[0] != q0.shape[0]:
            raise ValueError("Batch size mismatch: target_pose vs q0")

        B = target_pose.shape[0]
        if B == 1:
            r = self._ik.solve(
                target_pose[0],
                q0[0],
                int(self.max_iters),
                float(self.pos_tol),
                float(self.ori_tol),
                float(self.min_damping),
                float(self.max_damping),
                float(self.base_step),
                float(pos_weight),
                float(ori_weight),
                bool(adaptive_damping),
                bool(adaptive_step),
                float(max_step_norm),
            )
            out_q = [r["q"]]
            out_success = [bool(r["success"])]
            out_iters = [int(r["iters"])]
            out_err = [float(r["err_norm"])]
            out_pos = [float(r["pos_err"])]
            out_ori = [float(r["ori_err"])]
        else:
            raw = self._ik.solve_batch(
                target_pose,
                q0,
                int(self.max_iters),
                float(self.pos_tol),
                float(self.ori_tol),
                float(self.min_damping),
                float(self.max_damping),
                float(self.base_step),
                float(pos_weight),
                float(ori_weight),
                bool(adaptive_damping),
                bool(adaptive_step),
                float(max_step_norm),
            )
            qm = np.asarray(raw["q"], dtype=np.float64)
            succ_u8 = np.asarray(raw["success"], dtype=np.uint8)
            it_arr = np.asarray(raw["iters"], dtype=np.int32)
            en_arr = np.asarray(raw["err_norm"], dtype=np.float64)
            pe_arr = np.asarray(raw["pos_err"], dtype=np.float64)
            oe_arr = np.asarray(raw["ori_err"], dtype=np.float64)
            out_q = [qm[i].tolist() for i in range(B)]
            out_success = [bool(succ_u8[i]) for i in range(B)]
            out_iters = [int(it_arr[i]) for i in range(B)]
            out_err = [float(en_arr[i]) for i in range(B)]
            out_pos = [float(pe_arr[i]) for i in range(B)]
            out_ori = [float(oe_arr[i]) for i in range(B)]

        result = {
            "q": out_q,
            "success": out_success,
            "iters": out_iters,
            "err_norm": out_err,
            "method": ["dls"] * B,
            "jacobian": ["analytic"] * B,
            "pos_err": out_pos,
            "ori_err": out_ori,
        }
        return restore_single(result, was_single)


__all__ = ["IKSolverCpp"]
