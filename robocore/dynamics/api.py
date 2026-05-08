"""Unified dynamics API: inverse_dynamics, forward_dynamics, mass_matrix, etc.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from robocore.dynamics.model import DynamicsModel, build_dynamics_model
from robocore.dynamics.id import rnea, gravity as _gravity, nonlinear_effects as _nonlinear_effects
from robocore.dynamics.fd import crba, aba, mass_matrix_inverse as _mass_matrix_inverse
from robocore.dynamics.utils import coriolis_matrix as _coriolis_matrix, static_torque as _static_torque


def _get_dynamics_model(robot_model: Any) -> DynamicsModel:
    """Get or build DynamicsModel for robot_model (cached on robot_model._dynamics_model)."""
    if getattr(robot_model, "_dynamics_model", None) is None:
        robot_model._dynamics_model = build_dynamics_model(robot_model)
    return robot_model._dynamics_model


def inverse_dynamics(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    a: Union[Sequence[float], np.ndarray],
    fext: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Inverse dynamics: tau = RNEA(q, v, a).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param a: Joint accelerations (nq,).
    :param fext: Optional list of 6D forces per body (in body frame).
    :return: Joint torques (nq,).
    """
    dm = _get_dynamics_model(model)
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    return rnea(dm, q, v, a, fext=fext)


def forward_dynamics(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    tau: Union[Sequence[float], np.ndarray],
    fext: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Forward dynamics: ddq = ABA(q, v, tau).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param tau: Joint torques (nq,).
    :param fext: Optional external forces per body.
    :return: Joint accelerations (nq,).
    """
    dm = _get_dynamics_model(model)
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    tau = np.asarray(tau, dtype=np.float64).ravel()
    return aba(dm, q, v, tau, fext=fext)


def mass_matrix(model: Any, q: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Mass matrix M(q) via CRBA.

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: M (nq, nq).
    """
    dm = _get_dynamics_model(model)
    return crba(dm, np.asarray(q, dtype=np.float64).ravel())


def mass_matrix_inverse(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Inverse mass matrix M^{-1}(q).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: Minv (nq, nq).
    """
    dm = _get_dynamics_model(model)
    return _mass_matrix_inverse(dm, np.asarray(q, dtype=np.float64).ravel())


def gravity(model: Any, q: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Generalized gravity g(q).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: Joint torques due to gravity (nq,).
    """
    dm = _get_dynamics_model(model)
    return _gravity(dm, np.asarray(q, dtype=np.float64).ravel())


def nonlinear_effects(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Nonlinear effects nle = C(q,v)*v + g(q) = RNEA(q, v, 0).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :return: nle (nq,).
    """
    dm = _get_dynamics_model(model)
    return _nonlinear_effects(dm, np.asarray(q, dtype=np.float64).ravel(), np.asarray(v, dtype=np.float64).ravel())


def coriolis_matrix(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Coriolis matrix C(q,v) such that C @ v = nle - g.

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param epsilon: Finite difference step.
    :return: C (nq, nq).
    """
    dm = _get_dynamics_model(model)
    return _coriolis_matrix(dm, np.asarray(q, dtype=np.float64).ravel(), np.asarray(v, dtype=np.float64).ravel(), epsilon=epsilon)


def static_torque(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    fext: Optional[np.ndarray] = None,
    end_link: Optional[str] = None,
) -> np.ndarray:
    """Static torque g(q) - J^T f_ext.

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param fext: 6D force at end-effector in world frame (6,) or None for g(q).
    :param end_link: Link name for Jacobian; default model.end_link.
    :return: Joint torques (nq,).
    """
    return _static_torque(model, np.asarray(q, dtype=np.float64).ravel(), fext=fext, end_link=end_link)
