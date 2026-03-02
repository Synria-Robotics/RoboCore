"""Inverse dynamics: RNEA and derived quantities (gravity, nonlinear_effects).

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from robocore.dynamics.model import DynamicsModel, DynamicsBody


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from RPY (intrinsic XYZ)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    return R


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64).ravel()
    n = np.linalg.norm(a)
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    a = a / n
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 transform from R (3x3) and p (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64).ravel()[:3]
    return T


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of 3-vector."""
    v = np.asarray(v, dtype=np.float64).ravel()[:3]
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def _spatial_inertia(mass: float, com: np.ndarray, I_com: np.ndarray) -> np.ndarray:
    """6x6 spatial inertia in body frame (origin at link origin). I_com at COM."""
    c = np.asarray(com, dtype=np.float64).ravel()[:3]
    I = np.asarray(I_com, dtype=np.float64)
    if I.shape != (3, 3):
        I = np.eye(3) * 1e-6
    I_orig = I + mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    mc = mass * c
    top = np.hstack([I_orig, _skew(mc)])
    bot = np.hstack([_skew(mc).T, mass * np.eye(3)])
    return np.vstack([top, bot])


def _ad(T: np.ndarray) -> np.ndarray:
    """6x6 Lie-group adjoint Ad(T). Maps twist from child frame to parent frame
    when T = [R,p;0,1] takes child coords to parent coords (URDF convention).
    Note: NOT the Featherstone parent->child transform. See _X / _X_T for that.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    P = _skew(p)
    top = np.hstack([R, np.zeros((3, 3))])
    bot = np.hstack([P @ R, R])
    return np.vstack([top, bot])


def _ad_T(T: np.ndarray) -> np.ndarray:
    """Transpose of Ad(T). Legacy helper kept for compatibility."""
    return _ad(T).T


def _X(T: np.ndarray) -> np.ndarray:
    """Featherstone Pluecker transform: spatial velocity from parent to child frame.

    T = [R, p; 0, 1] where R rotates child->parent, p is child origin in parent.
    Equivalent to Ad(T^{-1}).
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Rt = R.T
    top = np.hstack([Rt, np.zeros((3, 3))])
    bot = np.hstack([-Rt @ _skew(p), Rt])
    return np.vstack([top, bot])


def _X_T(T: np.ndarray) -> np.ndarray:
    """Transpose of Pluecker transform: spatial force from child to parent frame."""
    return _X(T).T


def _ad_vec(v: np.ndarray) -> np.ndarray:
    """ad(v) 6x6: transform for acceleration bias ad(V) * (S*qd). v = [omega; vel]."""
    v = np.asarray(v, dtype=np.float64).ravel()[:6]
    w, u = v[:3], v[3:6]
    W, U = _skew(w), _skew(u)
    top = np.hstack([W, np.zeros((3, 3))])
    bot = np.hstack([U, W])
    return np.vstack([top, bot])


def _ad_dual(v: np.ndarray) -> np.ndarray:
    """ad*(v) for spatial force bias: F_bias = ad(V)^T I V."""
    v = np.asarray(v, dtype=np.float64).ravel()[:6]
    w, u = v[:3], v[3:6]
    W, U = _skew(w), _skew(u)
    top = np.hstack([-W, -U])
    bot = np.hstack([np.zeros((3, 3)), -W])
    return np.vstack([top, bot])


def _motion_subspace(joint_type: str, axis: np.ndarray) -> np.ndarray:
    """6x1 motion subspace in body frame. Revolute: [axis; 0]. Prismatic: [0; axis]."""
    a = np.asarray(axis, dtype=np.float64).ravel()[:3]
    n = np.linalg.norm(a)
    if n < 1e-10:
        a = np.array([0, 0, 1], dtype=np.float64)
    else:
        a = a / n
    if joint_type == "revolute":
        return np.concatenate([a, np.zeros(3)])
    elif joint_type == "prismatic":
        return np.concatenate([np.zeros(3), a])
    else:
        return np.zeros(6)


def rnea(
    model: DynamicsModel,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    fext: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Recursive Newton-Euler algorithm: inverse dynamics.

    :param model: DynamicsModel from build_dynamics_model(robot_model).
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :param a: Joint accelerations (nq,).
    :param fext: Optional list of 6D forces per body (in body frame); len = n_bodies.
    :return: Joint torques (nq,).
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    nq = model.nq
    if len(q) != nq or len(v) != nq or len(a) != nq:
        raise ValueError(f"q,v,a must have length nq={nq}")
    bodies = model.bodies
    n_bodies = len(bodies)
    gravity_world = np.asarray(model.gravity, dtype=np.float64).ravel()[:3]
    # Base acceleration in world (gravity)
    a0 = np.concatenate([np.zeros(3), -gravity_world])
    # Build T_parent_child and S for each body
    T_world: List[np.ndarray] = [np.eye(4)] * n_bodies
    T_parent_child: List[np.ndarray] = [np.eye(4)] * n_bodies
    S_list: List[np.ndarray] = []
    I_list: List[np.ndarray] = []

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
        T_p_c = T_orig @ T_motion
        T_parent_child[i] = T_p_c
        if i == 0:
            T_world[i] = np.eye(4)
        else:
            T_world[i] = T_world[b.parent] @ T_p_c
        S_list.append(_motion_subspace(b.joint_type, b.joint_axis))
        I_list.append(_spatial_inertia(b.mass, b.com_xyz, b.inertia))

    # Forward: spatial velocities and accelerations in body frame (Featherstone convention)
    V: List[np.ndarray] = [np.zeros(6)] * n_bodies
    A: List[np.ndarray] = [np.zeros(6)] * n_bodies
    for i in range(n_bodies):
        b = bodies[i]
        Xi = _X(T_parent_child[i])
        if i == 0:
            V[i] = np.zeros(6)
            A[i] = Xi @ a0
        else:
            qd = v[b.q_index] if b.q_index >= 0 else 0.0
            ddq = a[b.q_index] if b.q_index >= 0 else 0.0
            V[i] = Xi @ V[b.parent] + S_list[i] * qd
            A[i] = Xi @ A[b.parent] + S_list[i] * ddq + _ad_vec(V[i]) @ (S_list[i] * qd)

    # Backward: forces and torques (two passes so child contributions are not overwritten)
    F: List[np.ndarray] = [np.zeros(6)] * n_bodies
    tau = np.zeros(nq, dtype=np.float64)
    if fext is None:
        fext = [np.zeros(6)] * n_bodies
    for i in range(n_bodies):
        b = bodies[i]
        f_ext_i = np.asarray(fext[i], dtype=np.float64).ravel()[:6] if i < len(fext) else np.zeros(6)
        # Featherstone: f = I*a + v ×* (I*v) where v ×* = -ad(v)^T
        F[i] = I_list[i] @ A[i] - _ad_vec(V[i]).T @ (I_list[i] @ V[i]) - f_ext_i
    for i in range(n_bodies - 1, -1, -1):
        b = bodies[i]
        if b.parent >= 0:
            F[b.parent] += _X_T(T_parent_child[i]) @ F[i]
        if b.q_index >= 0:
            tau[b.q_index] = S_list[i].T @ F[i]

    return tau


def gravity(model: DynamicsModel, q: np.ndarray) -> np.ndarray:
    """Generalized gravity vector g(q) = RNEA(q, 0, 0).

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :return: Joint torques due to gravity (nq,).
    """
    v = np.zeros(model.nq, dtype=np.float64)
    a = np.zeros(model.nq, dtype=np.float64)
    return rnea(model, q, v, a)


def nonlinear_effects(model: DynamicsModel, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nonlinear effects nle = C(q,v)*v + g(q) = RNEA(q, v, 0).

    :param model: DynamicsModel.
    :param q: Joint positions (nq,).
    :param v: Joint velocities (nq,).
    :return: nle (nq,).
    """
    a = np.zeros(model.nq, dtype=np.float64)
    return rnea(model, q, v, a)
