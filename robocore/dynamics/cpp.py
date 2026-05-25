"""C++/Eigen dynamics backend wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from robocore.dynamics.model import (
    DynamicsModel,
    _make_transform,
    _rpy_to_matrix,
    build_dynamics_model,
)
from robocore.kinematics.utils import restore_single

from . import _dynamics_core


def arrays_from_dynamics_model(model: DynamicsModel):
    """Pack :class:`DynamicsModel` into arrays consumed by the native backend."""
    nb = len(model.bodies)
    parents = np.empty(nb, dtype=np.int32)
    joint_types = np.zeros(nb, dtype=np.int32)
    q_indices = np.empty(nb, dtype=np.int32)
    T_rows = np.zeros((nb, 16), dtype=np.float64)
    axes = np.zeros((nb, 3), dtype=np.float64)
    masses = np.zeros(nb, dtype=np.float64)
    coms = np.zeros((nb, 3), dtype=np.float64)
    inertias = np.zeros((nb, 9), dtype=np.float64)

    for i, body in enumerate(model.bodies):
        parents[i] = int(body.parent)
        if body.joint_type == "revolute":
            joint_types[i] = 1
        elif body.joint_type == "prismatic":
            joint_types[i] = 2
        else:
            joint_types[i] = 0
        q_indices[i] = int(body.q_index)
        R = _rpy_to_matrix(
            float(body.joint_origin_rpy[0]),
            float(body.joint_origin_rpy[1]),
            float(body.joint_origin_rpy[2]),
        )
        T_origin = _make_transform(R, np.asarray(body.joint_origin_xyz, dtype=np.float64))
        T_rows[i, :] = np.asarray(T_origin, dtype=np.float64).flatten(order="F")
        axes[i, :] = np.asarray(body.joint_axis, dtype=np.float64)
        masses[i] = float(body.mass)
        coms[i, :] = np.asarray(body.com_xyz, dtype=np.float64)
        inertias[i, :] = np.asarray(body.inertia, dtype=np.float64).reshape(9)

    return (
        parents,
        joint_types,
        q_indices,
        T_rows,
        axes,
        masses,
        coms,
        inertias,
        np.asarray(model.gravity, dtype=np.float64),
        np.int32(model.nq),
    )


class DynamicsSolverCpp:
    """Native fixed-base rigid-body dynamics solver."""

    def __init__(self, robot_model: Any):
        self.robot_model = robot_model
        self.model = build_dynamics_model(robot_model)
        self.nq = self.model.nq
        self._core = _dynamics_core.ChainDynamics(*arrays_from_dynamics_model(self.model))

    @staticmethod
    def _single_q(x: Any) -> bool:
        return np.asarray(x).ndim == 1

    def inverse_dynamics(self, q: Any, v: Any, a: Any) -> np.ndarray:
        was_single = self._single_q(q)
        out = np.asarray(
            self._core.rnea(
                np.ascontiguousarray(q, dtype=np.float64),
                np.ascontiguousarray(v, dtype=np.float64),
                np.ascontiguousarray(a, dtype=np.float64),
            )
        )
        return restore_single(out, was_single)

    def mass_matrix(self, q: Any) -> np.ndarray:
        was_single = self._single_q(q)
        out = np.asarray(self._core.crba(np.ascontiguousarray(q, dtype=np.float64)))
        return restore_single(out, was_single)

    def forward_dynamics(self, q: Any, v: Any, tau: Any) -> np.ndarray:
        was_single = self._single_q(q)
        out = np.asarray(
            self._core.forward_dynamics(
                np.ascontiguousarray(q, dtype=np.float64),
                np.ascontiguousarray(v, dtype=np.float64),
                np.ascontiguousarray(tau, dtype=np.float64),
            )
        )
        return restore_single(out, was_single)

    def gravity(self, q: Any) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64)
        zeros = np.zeros_like(q_arr, dtype=np.float64)
        return self.inverse_dynamics(q_arr, zeros, zeros)

    def nonlinear_effects(self, q: Any, v: Any) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64)
        return self.inverse_dynamics(q_arr, np.asarray(v, dtype=np.float64), np.zeros_like(q_arr, dtype=np.float64))

    def mass_matrix_inverse(self, q: Any) -> np.ndarray:
        M = self.mass_matrix(q)
        if M.ndim == 2:
            return np.linalg.inv(M)
        return np.stack([np.linalg.inv(Mi) for Mi in M], axis=0)


def get_cpp_dynamics_solver(robot_model: Any) -> DynamicsSolverCpp:
    solver = getattr(robot_model, "_dynamics_cpp_solver", None)
    if solver is None:
        solver = DynamicsSolverCpp(robot_model)
        robot_model._dynamics_cpp_solver = solver
    return solver


__all__ = ["DynamicsSolverCpp", "arrays_from_dynamics_model", "get_cpp_dynamics_solver"]
