#!/usr/bin/env python3
"""Temporal DLS prior: smooth_weight=0 matches old DLS; w>0 reduces joint chatter."""

from __future__ import annotations

import numpy as np
import pytest

from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.ik_utils.ik_solver_numpy import (
    IKSolverNumPy,
    apply_wrist_singularity_lock,
    coerce_smooth_weights,
)
from robocore.modeling.robot_model import RobotModel


@pytest.fixture(scope="module")
def robot_model(alicia_urdf_path):
    return RobotModel(alicia_urdf_path, end_link="tool0")


def _backends():
    backends = ["numpy"]
    try:
        import robocore.kinematics.ik_utils.ik_solver_cpp  # noqa: F401
    except Exception:
        return backends
    backends.append("cpp")
    return backends


def _ik_kwargs(backend: str, **extra):
    kw = dict(
        backend=backend,
        method="dls",
        max_iters=80,
        pos_tol=1e-3,
        ori_tol=1e-3,
        num_initial_guesses=1,
        joint_centering=False,
        use_analytic_jacobian=True,
    )
    kw.update(extra)
    return kw


def _sign_flip_rate(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    dq = np.diff(q, axis=0)
    moving = np.abs(dq) > eps
    prod = dq[1:] * dq[:-1]
    both = moving[1:] & moving[:-1]
    flips = (prod < 0) & both
    denom = np.maximum(both.sum(axis=0), 1)
    return flips.sum(axis=0) / denom


def test_dls_zero_weight_matches_task_space_form():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 6))
    err = rng.normal(size=6)
    damping = 0.02
    solver = IKSolverNumPy.__new__(IKSolverNumPy)
    dq_old = solver._solve_dls(J, err, damping)
    dq_w0 = solver._solve_dls(J, err, damping, q=np.zeros(6), q_prev=np.ones(6), smooth_weight=0.0)
    A = J @ J.T + (damping ** 2) * np.eye(6)
    dq_ref = J.T @ np.linalg.solve(A, err)
    assert np.allclose(dq_old, dq_ref, atol=1e-12)
    assert np.allclose(dq_w0, dq_ref, atol=1e-12)


def test_dls_smooth_pulls_toward_q_prev():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(6, 6))
    err = rng.normal(size=6)
    damping = 0.05
    q = rng.normal(size=6)
    q_prev = q + rng.normal(size=6) * 0.2
    solver = IKSolverNumPy.__new__(IKSolverNumPy)
    dq0 = solver._solve_dls(J, err, damping, q=q, q_prev=q_prev, smooth_weight=0.0)
    dq1 = solver._solve_dls(J, err, damping, q=q, q_prev=q_prev, smooth_weight=0.1)
    dist0 = np.linalg.norm((q + dq0) - q_prev)
    dist1 = np.linalg.norm((q + dq1) - q_prev)
    assert dist1 < dist0


@pytest.mark.parametrize("backend", _backends())
def test_smooth_weight_zero_matches_default(robot_model, backend):
    q_target = robot_model.random_q(seed=0, scale=0.4)
    target = forward_kinematics(robot_model, q_target, return_end=True, backend=backend)
    q0 = robot_model.random_q(seed=1, scale=0.4)
    a = inverse_kinematics(robot_model, target, q0, **_ik_kwargs(backend))
    b = inverse_kinematics(robot_model, target, q0, **_ik_kwargs(backend, smooth_weight=0.0))
    assert np.allclose(np.asarray(a["q"]), np.asarray(b["q"]), atol=1e-12)


@pytest.mark.parametrize("backend", _backends())
def test_smooth_fail_returns_q0(robot_model, backend):
    target = np.eye(4)
    target[:3, 3] = 50.0
    q0 = np.zeros(robot_model.num_chain_dof)
    res = inverse_kinematics(
        robot_model,
        target,
        q0,
        **_ik_kwargs(backend, smooth_weight=0.1, max_iters=8),
    )
    assert res["success"] is False
    assert np.allclose(np.asarray(res["q"]), q0, atol=1e-12)


@pytest.mark.parametrize("backend", _backends())
def test_smooth_weight_reduces_joint_delta(robot_model, backend):
    n = 36
    q_a = robot_model.random_q(seed=10, scale=0.35)
    q_b = robot_model.random_q(seed=11, scale=0.35)
    alpha = np.linspace(0.0, 1.0, n)[:, None]
    q_path = (1.0 - alpha) * q_a + alpha * q_b
    poses = [forward_kinematics(robot_model, q, return_end=True, backend=backend) for q in q_path]

    def replay(smooth_weight: float) -> np.ndarray:
        q_prev = q_path[0].copy()
        out = []
        for T in poses:
            res = inverse_kinematics(
                robot_model,
                T,
                q_prev,
                **_ik_kwargs(
                    backend,
                    pos_weight=10.0,
                    ori_weight=1.0,
                    ori_tol=1e-2,
                    smooth_weight=smooth_weight,
                    q_prev=q_prev,
                    max_iters=120,
                ),
            )
            q_prev = np.asarray(res["q"], dtype=np.float64)
            out.append(q_prev)
        return np.asarray(out)

    q_raw = replay(0.0)
    q_smooth = replay(0.1)
    mean_dq_raw = float(np.mean(np.abs(np.diff(q_raw, axis=0))))
    mean_dq_smooth = float(np.mean(np.abs(np.diff(q_smooth, axis=0))))
    assert mean_dq_smooth <= mean_dq_raw + 1e-9
    j3 = 3 if q_raw.shape[1] > 3 else 0
    assert _sign_flip_rate(q_smooth)[j3] <= _sign_flip_rate(q_raw)[j3] + 1e-9


def test_coerce_smooth_weights_broadcasts_scalar():
    w = coerce_smooth_weights(0.1, 6)
    assert w.shape == (6,)
    assert np.allclose(w, 0.1)
    v = coerce_smooth_weights([0.0, 0.0, 0.0, 0.4, 0.1, 0.4], 6)
    assert np.allclose(v[3], 0.4)


def test_wrist_lock_peaks_at_j4_zero():
    base = np.full(6, 0.05)
    w0 = apply_wrist_singularity_lock(base, np.array([0, 0, 0, -1.0, 0.0, 0.2]), 0.5)
    w_far = apply_wrist_singularity_lock(base, np.array([0, 0, 0, -1.0, 1.0, 0.2]), 0.5)
    assert w0[3] > w_far[3] + 0.2
    assert w0[5] > w_far[5] + 0.2
    assert np.allclose(w0[0], 0.05)
    assert np.allclose(w_far[3], 0.05, atol=1e-3)


def test_per_joint_weight_pins_selected_axis():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(6, 6))
    err = rng.normal(size=6)
    q = rng.normal(size=6)
    q_prev = q.copy()
    q_prev[3] += 0.3
    solver = IKSolverNumPy.__new__(IKSolverNumPy)
    dq_uni = solver._solve_dls(J, err, 0.05, q=q, q_prev=q_prev, smooth_weight=0.05)
    w = np.array([0.05, 0.05, 0.05, 2.0, 0.05, 0.05])
    dq_pin = solver._solve_dls(J, err, 0.05, q=q, q_prev=q_prev, smooth_weight=w)
    assert abs((q[3] + dq_pin[3]) - q_prev[3]) < abs((q[3] + dq_uni[3]) - q_prev[3])


@pytest.mark.parametrize("backend", _backends())
def test_wrist_lock_reduces_j3_chatter_near_singularity(robot_model, backend):
    n = robot_model.num_chain_dof
    q_mid = np.zeros(n, dtype=np.float64)
    q_mid[1] = -1.4
    q_mid[2] = -1.1
    q_mid[3] = -1.05
    q_mid[4] = 0.02
    q_mid[5] = 0.2
    n_steps = 48
    t = np.linspace(0.0, 1.0, n_steps)
    q_path = np.repeat(q_mid[None, :], n_steps, axis=0)
    q_path[:, 0] += 0.08 * np.sin(2.0 * np.pi * t)
    q_path[:, 5] += 0.05 * np.sin(4.0 * np.pi * t)
    poses = [forward_kinematics(robot_model, q, return_end=True, backend=backend) for q in q_path]

    def replay(**extra) -> np.ndarray:
        q_prev = q_path[0].copy()
        out = []
        for T in poses:
            res = inverse_kinematics(
                robot_model,
                T,
                q_prev,
                **_ik_kwargs(
                    backend,
                    pos_weight=10.0,
                    ori_weight=0.5,
                    ori_tol=2e-2,
                    q_prev=q_prev,
                    max_iters=80,
                    **extra,
                ),
            )
            q_prev = np.asarray(res["q"], dtype=np.float64)
            out.append(q_prev)
        return np.asarray(out)

    q_raw = replay(smooth_weight=0.0, wrist_singularity_gain=0.0)
    q_lock = replay(smooth_weight=0.05, wrist_singularity_gain=0.15)
    assert int(np.isfinite(q_lock).all()) == 1
    j3 = 3 if q_raw.shape[1] > 3 else 0
    assert _sign_flip_rate(q_lock)[j3] <= _sign_flip_rate(q_raw)[j3] + 1e-9
    assert float(np.max(np.abs(np.diff(q_lock[:, j3])))) <= float(
        np.max(np.abs(np.diff(q_raw[:, j3])))
    ) + 1e-9

