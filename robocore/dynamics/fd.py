"""Forward dynamics and mass matrix utilities.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from robocore.dynamics.model import DynamicsModel, DynamicsBody
from robocore.dynamics.id import (
    rnea,
    _rpy_to_matrix,
    _axis_angle_to_matrix,
    _make_transform,
    _skew,
    _spatial_inertia,
    _ad,
    _ad_T,
    _X,
    _X_T,
    _motion_subspace,
)


def crba(model: DynamicsModel, q: np.ndarray) -> np.ndarray:
    """Composite Rigid Body Algorithm: mass matrix M(q).

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :return: M (nq, nq), symmetric positive definite.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    nq = model.nq
    bodies = model.bodies
    n_bodies = len(bodies)

    # Build T_parent_child and S, I for each body (same as RNEA)
    T_parent_child = []
    S_list = []
    I_list = []
    for i, b in enumerate(bodies):
        R_orig = _rpy_to_matrix(b.joint_origin_rpy[0], b.joint_origin_rpy[1], b.joint_origin_rpy[2])
        p_orig = np.asarray(b.joint_origin_xyz, dtype=np.float64)
        if b.joint_type == "revolute":
            qi = q[b.q_index] if b.q_index >= 0 else 0.0
            R_motion = _axis_angle_to_matrix(b.joint_axis, qi)
            p_motion = np.zeros(3)
        elif b.joint_type == "prismatic":
            qi = q[b.q_index] if b.q_index >= 0 else 0.0
            R_motion = np.eye(3)
            p_motion = (b.joint_axis * qi).ravel()[:3]
        else:
            R_motion = np.eye(3)
            p_motion = np.zeros(3)
        T_orig = _make_transform(R_orig, p_orig)
        T_motion = _make_transform(R_motion, p_motion)
        T_parent_child.append(T_orig @ T_motion)
        S_list.append(_motion_subspace(b.joint_type, b.joint_axis))
        I_list.append(_spatial_inertia(b.mass, b.com_xyz, b.inertia))

    # Composite inertias from tip to base (Featherstone: Ic[parent] += X^T Ic[child] X)
    Ic = [I_list[i].copy() for i in range(n_bodies)]
    for j in range(n_bodies - 1, 0, -1):
        p = bodies[j].parent
        Ic[p] = Ic[p] + _X_T(T_parent_child[j]) @ Ic[j] @ _X(T_parent_child[j])

    # Fill upper triangle of M: for each j (actuated), F = Ic[j] @ S_j, then propagate to parents
    M = np.zeros((nq, nq), dtype=np.float64)
    for j in range(n_bodies):
        b = bodies[j]
        if b.q_index < 0:
            continue
        from_body = j
        F = Ic[j] @ S_list[j]
        M[b.q_index, b.q_index] = S_list[j].T @ F
        while bodies[from_body].parent >= 0:
            to_body = bodies[from_body].parent
            F = _X_T(T_parent_child[from_body]) @ F
            if bodies[to_body].q_index >= 0:
                M[bodies[to_body].q_index, b.q_index] = S_list[to_body].T @ F
            from_body = to_body
    # Symmetrize
    M = (M + M.T) - np.diag(np.diag(M))
    return M


def aba(
    model: DynamicsModel,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    fext: Optional[list] = None,
) -> np.ndarray:
    """Forward dynamics by mass-matrix solve.

    Uses M from CRBA and nle from RNEA, then solves M @ ddq = tau - nle.

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param tau: Joint torques (nq,).
    :param fext: Optional external forces per body.
    :return: Joint accelerations (nq,).
    """
    nle = rnea(model, q, v, np.zeros(model.nq), fext=fext)
    M = crba(model, q)
    ddq = np.linalg.solve(M, np.asarray(tau, dtype=np.float64).ravel() - nle)
    return ddq


def mass_matrix_inverse(model: DynamicsModel, q: np.ndarray) -> np.ndarray:
    """Inverse of mass matrix M^{-1}(q) via Cholesky of M.

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :return: Minv (nq, nq).
    """
    M = crba(model, q)
    return np.linalg.inv(M)
