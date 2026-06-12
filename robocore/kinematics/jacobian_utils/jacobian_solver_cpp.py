"""Eigen + pybind11 geometric Jacobian (required native extension).

Requires a built ``_jacobian_chain_core`` module (Eigen3 + pybind11; ``pip install -e .``).
Importing this module fails if the extension is missing.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal

import numpy as np

from robocore.kinematics.fk_utils.fk_solver_cpp import chain_arrays_from_robot_model
from robocore.kinematics.utils import ensure_batch, restore_single

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

_jacobian_chain_core = import_module("robocore.kinematics.jacobian_utils._jacobian_chain_core")


def stop_chain_index(model: "RobotModel", target_link: str | None) -> int:
    """Chain joint index to stop after (inclusive), or -1 for full chain.

    :param model: robot model.
    :param target_link: child link name at stop, or None.
    :return: index in ``model._chain_joints`` or -1.
    """
    if target_link is None:
        return -1
    for i, j in enumerate(model._chain_joints):
        if j.child == target_link:
            return int(i)
    raise ValueError(f"target_link {target_link!r} not found on chain to {model.end_link!r}")


class JacobianSolverCpp:
    """Geometric Jacobian in world frame (6 x n), compiled chain."""

    def __init__(self, model: "RobotModel", *, target_link: str | None = None):
        """Initialize Jacobian solver.

        :param model: robot model.
        :param target_link: optional link to treat as end-effector (partial chain).
        """
        self.model = model
        self.n = model.num_chain_dof
        self._target_link = target_link
        t, qi, To, ax, n_dof = chain_arrays_from_robot_model(model)
        if n_dof != self.n:
            raise ValueError("internal DOF mismatch")
        stop = np.int32(stop_chain_index(model, target_link))
        self._jac = _jacobian_chain_core.ChainJacobian(t, qi, To, ax, np.int32(n_dof), stop)

    def solve(
        self,
        q: np.ndarray,
        method: Literal["analytic", "numeric"] = "analytic",
        epsilon: float = 5e-5,
        use_central_diff: bool = True,
        target_link: str | None = None,
    ) -> np.ndarray:
        """Compute Jacobian via native chain (analytic geometric or numeric finite differences).

        :param q: configuration (n,) or (B, n).
        :param method: ``analytic`` or ``numeric`` (FK-based FD, same convention as NumPy solver).
        :param epsilon: finite-difference step (numeric).
        :param use_central_diff: central vs forward difference (numeric).
        :param target_link: must match constructor if partial chain was used; ignored if None in both.
        :return: (6, n) or (B, 6, n).
        """
        if method not in ("analytic", "numeric"):
            raise NotImplementedError("JacobianSolverCpp only implements method='analytic' or 'numeric'")
        if target_link is not None and target_link != self._target_link:
            raise ValueError(
                "JacobianSolverCpp was built for a fixed target_link; "
                "create another solver or pass target_link=None in both places."
            )
        q = np.asarray(q, dtype=np.float64)
        q, was_single = ensure_batch(q)
        if q.shape[1] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[1]}")
        if method == "analytic":
            out = self._jac.jacobian_analytic(q)
        else:
            out = self._jac.jacobian_numeric(q, float(epsilon), bool(use_central_diff))
        return restore_single(out, was_single)


__all__ = ["JacobianSolverCpp", "stop_chain_index"]
