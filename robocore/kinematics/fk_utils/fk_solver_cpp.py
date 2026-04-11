"""Eigen + pybind11 forward kinematics (required native extension).

Requires a built ``_fk_chain_core`` module (Eigen3 + pybind11; ``pip install -e .`` / ``setup.py``).
Importing this module fails if the extension is missing.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from robocore.kinematics.utils import ensure_batch, restore_single

from . import _fk_chain_core


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX extrinsic R = Rz * Ry * Rx (matches robocore.transform.rpy_to_matrix, NumPy only)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def chain_arrays_from_robot_model(model: "RobotModel"):
    """Pack RobotModel chain into arrays for :class:`ChainFK`.

    :param model: loaded robot model.
    :return: tuple (joint_types int32, q_indices int32, T_origin_f T_origin (n,16) column-major rows, axes (n,3), n_dof).
    """
    name_to_q = {js.name: js.index for js in model._chain_dof_list}
    joints = model._chain_joints
    n_j = len(joints)
    types = np.zeros(n_j, dtype=np.int32)
    q_indices = np.full(n_j, -1, dtype=np.int32)
    T_rows = np.zeros((n_j, 16), dtype=np.float64)
    axes = np.zeros((n_j, 3), dtype=np.float64)

    for i, j in enumerate(joints):
        if j.joint_type == "fixed":
            types[i] = 0
        elif j.joint_type == "revolute":
            types[i] = 1
        elif j.joint_type == "prismatic":
            types[i] = 2
        else:
            types[i] = 0

        if j.joint_type in ("revolute", "prismatic"):
            if j.name not in name_to_q:
                raise KeyError(f"actuated joint {j.name!r} missing from chain DOF list")
            q_indices[i] = name_to_q[j.name]

        R0 = _rpy_to_R(j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2])
        T_origin = _make_T(R0, np.asarray(j.origin_xyz, dtype=np.float64))
        # Column-major flatten for Eigen::Map<Matrix4d> per row i
        T_rows[i, :] = np.asarray(T_origin, dtype=np.float64).flatten(order="F")
        axes[i, :] = np.asarray(j.axis, dtype=np.float64)

    n_dof = model.num_chain_dof
    return types, q_indices, T_rows, axes, n_dof


class FKSolverCpp:
    """Forward kinematics using compiled Eigen chain (end-effector only, batch supported)."""

    def __init__(self, model: "RobotModel"):
        """Initialize FK solver.

        :param model: robot model.
        """
        self.model = model
        self.n = model.num_chain_dof
        t, qi, To, ax, n_dof = chain_arrays_from_robot_model(model)
        if n_dof != self.n:
            raise ValueError("internal DOF mismatch")
        self._chain = _fk_chain_core.ChainFK(t, qi, To, ax, np.int32(n_dof))
        self.end_link = model.end_link

    def solve(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        return_end_only: bool = True,
    ) -> Dict[str, np.ndarray] | np.ndarray:
        """Compute FK (only return_end_only=True supported).

        :param q: configuration (n,) or (B, n).
        :param return_end_only: must be True.
        :return: (4,4) or (B,4,4) pose.
        """
        if not return_end_only:
            raise NotImplementedError("FKSolverCpp only implements return_end_only=True")
        q = np.asarray(q, dtype=np.float64)
        q, was_single = ensure_batch(q)
        if q.shape[1] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[1]}")
        out = self._chain.fk_end(q)
        return restore_single(out, was_single)


__all__ = ["FKSolverCpp", "chain_arrays_from_robot_model"]
