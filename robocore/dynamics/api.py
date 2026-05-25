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
from robocore.utils.backend import get_backend


def _get_dynamics_model(robot_model: Any) -> DynamicsModel:
    """Get or build DynamicsModel for robot_model (cached on robot_model._dynamics_model)."""
    if getattr(robot_model, "_dynamics_model", None) is None:
        robot_model._dynamics_model = build_dynamics_model(robot_model)
    return robot_model._dynamics_model


def _resolve_dynamics_backend(backend: Optional[str]) -> str:
    selected = backend or get_backend()
    if selected == "cpp":
        return "cpp"
    if selected in ("numpy", "torch"):
        return "numpy"
    raise ValueError("Dynamics backend must be 'numpy' or 'cpp'")


def _get_cpp_solver(robot_model: Any):
    from robocore.dynamics.cpp import get_cpp_dynamics_solver

    return get_cpp_dynamics_solver(robot_model)


def inverse_dynamics(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    a: Union[Sequence[float], np.ndarray],
    fext: Optional[List[np.ndarray]] = None,
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Inverse dynamics: tau = RNEA(q, v, a).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param a: Joint accelerations (nq,).
    :param fext: Optional list of 6D forces per body (in body frame).
    :return: Joint torques (nq,).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp" and fext is None:
        return _get_cpp_solver(model).inverse_dynamics(q_arr, v_arr, a_arr)
    q = q_arr.ravel()
    v = v_arr.ravel()
    a = a_arr.ravel()
    dm = _get_dynamics_model(model)
    return rnea(dm, q, v, a, fext=fext)


def forward_dynamics(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    tau: Union[Sequence[float], np.ndarray],
    fext: Optional[List[np.ndarray]] = None,
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Forward dynamics: solve M(q) @ ddq = tau - nle(q, v).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param tau: Joint torques (nq,).
    :param fext: Optional external forces per body.
    :return: Joint accelerations (nq,).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp" and fext is None:
        return _get_cpp_solver(model).forward_dynamics(q_arr, v_arr, tau_arr)
    q = q_arr.ravel()
    v = v_arr.ravel()
    tau = tau_arr.ravel()
    dm = _get_dynamics_model(model)
    return aba(dm, q, v, tau, fext=fext)


def mass_matrix(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Mass matrix M(q) via CRBA.

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: M (nq, nq).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp":
        return _get_cpp_solver(model).mass_matrix(q_arr)
    dm = _get_dynamics_model(model)
    return crba(dm, q_arr.ravel())


def mass_matrix_inverse(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Inverse mass matrix M^{-1}(q).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: Minv (nq, nq).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp":
        return _get_cpp_solver(model).mass_matrix_inverse(q_arr)
    dm = _get_dynamics_model(model)
    return _mass_matrix_inverse(dm, q_arr.ravel())


def gravity(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Generalized gravity g(q).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :return: Joint torques due to gravity (nq,).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp":
        return _get_cpp_solver(model).gravity(q_arr)
    dm = _get_dynamics_model(model)
    return _gravity(dm, q_arr.ravel())


def nonlinear_effects(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Nonlinear effects nle = C(q,v)*v + g(q) = RNEA(q, v, 0).

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :return: nle (nq,).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if _resolve_dynamics_backend(backend) == "cpp":
        return _get_cpp_solver(model).nonlinear_effects(q_arr, v_arr)
    dm = _get_dynamics_model(model)
    return _nonlinear_effects(dm, q_arr.ravel(), v_arr.ravel())


def coriolis_matrix(
    model: Any,
    q: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
    epsilon: float = 1e-7,
    backend: Optional[str] = None,
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
    backend: Optional[str] = None,
) -> np.ndarray:
    """Static torque g(q) - J^T f_ext.

    :param model: RobotModel instance.
    :param q: Joint positions (nq,).
    :param fext: 6D force at end-effector in world frame (6,) or None for g(q).
    :param end_link: Link name for Jacobian; default model.end_link.
    :return: Joint torques (nq,).
    """
    return _static_torque(model, np.asarray(q, dtype=np.float64).ravel(), fext=fext, end_link=end_link)
