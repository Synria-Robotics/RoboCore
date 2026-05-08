"""Dynamics utilities: coriolis_matrix, static_torque.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from robocore.dynamics.model import DynamicsModel
from robocore.dynamics.id import nonlinear_effects, gravity
from robocore.dynamics.fd import crba


def coriolis_matrix(
    model: DynamicsModel,
    q: np.ndarray,
    v: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Coriolis matrix C(q,v) such that C @ v = nle - g. Computed by finite difference of nle w.r.t. v.

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param epsilon: Finite difference step for v.
    :return: C (nq, nq).
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nq = model.nq
    g_q = gravity(model, q)
    nle_qv = nonlinear_effects(model, q, v)
    # C @ v = nle - g  =>  C[i,:] = d(nle_i - g_i) / d(v)  approx by (nle(q, v + eps*ej) - nle(q,v)) / eps - 0
    C = np.zeros((nq, nq), dtype=np.float64)
    for j in range(nq):
        v_plus = v.copy()
        v_plus[j] += epsilon
        nle_plus = nonlinear_effects(model, q, v_plus)
        C[:, j] = (nle_plus - nle_qv) / epsilon
    return C


def static_torque(
    robot_model: Any,
    q: np.ndarray,
    fext: Optional[np.ndarray] = None,
    end_link: Optional[str] = None,
) -> np.ndarray:
    """Static torque g(q) - J^T f_ext at end-effector. Requires jacobian from kinematics.

    :param robot_model: RobotModel (for FK and Jacobian).
    :param q: Joint positions (nq or num_dof).
    :param fext: External 6D force at end-effector in world frame (6,) or None for g(q) only.
    :param end_link: Link name for J and f_ext; default robot_model.end_link.
    :return: Joint torques (nq,).
    """
    from robocore.dynamics.model import build_dynamics_model
    from robocore.dynamics.id import gravity
    from robocore.kinematics.jacobian import jacobian

    dyn_model = build_dynamics_model(robot_model)
    q = np.asarray(q, dtype=np.float64).ravel()
    g_q = gravity(dyn_model, q)
    if fext is None or np.allclose(fext, 0):
        return g_q
    fext = np.asarray(fext, dtype=np.float64).ravel()[:6]
    link = end_link or getattr(robot_model, "end_link", None)
    if link is None:
        return g_q
    J = jacobian(robot_model, q, end_link=link)
    J = np.asarray(J, dtype=np.float64)
    if J.ndim == 3:
        J = J[0]
    return g_q - J.T @ fext
