"""Unit tests for dynamics module: RNEA, CRBA, ABA, self-consistency.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import pytest
from pathlib import Path

from robocore.modeling.robot_model import RobotModel
from robocore import dynamics


def _urdf_path():
    try:
        from synriard import get_model_path
        return Path(get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf"))
    except Exception:
        pass
    p = Path(__file__).resolve().parent.parent.parent / "robocore" / "assets" / "robot_descriptions" / "urdf" / "Alicia-D_v5_5" / "alicia_duo_with_gripper.urdf"
    if not p.exists():
        p = Path("robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf")
    return p


@pytest.fixture(scope="module")
def robot_model():
    urdf_path = _urdf_path()
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return RobotModel(str(urdf_path), end_link="tool0")


def test_inverse_dynamics_shape(robot_model):
    nq = robot_model.num_chain_dof
    q = np.zeros(nq)
    v = np.zeros(nq)
    a = np.zeros(nq)
    tau = dynamics.inverse_dynamics(robot_model, q, v, a)
    assert tau.shape == (nq,)
    assert tau.dtype == np.float64


def test_gravity_shape(robot_model):
    nq = robot_model.num_chain_dof
    q = np.zeros(nq)
    g = dynamics.gravity(robot_model, q)
    assert g.shape == (nq,)


def test_nonlinear_effects_shape(robot_model):
    nq = robot_model.num_chain_dof
    q = np.zeros(nq)
    v = np.zeros(nq)
    nle = dynamics.nonlinear_effects(robot_model, q, v)
    assert nle.shape == (nq,)


def test_mass_matrix_shape(robot_model):
    nq = robot_model.num_chain_dof
    q = np.zeros(nq)
    M = dynamics.mass_matrix(robot_model, q)
    assert M.shape == (nq, nq)
    assert np.allclose(M, M.T), "M should be symmetric"


def test_forward_dynamics_shape(robot_model):
    nq = robot_model.num_chain_dof
    q = np.zeros(nq)
    v = np.zeros(nq)
    tau = np.zeros(nq)
    ddq = dynamics.forward_dynamics(robot_model, q, v, tau)
    assert ddq.shape == (nq,)


def test_tau_equals_M_ddq_plus_nle(robot_model):
    """Self-consistency: tau = M(q) @ ddq + nle(q, v)."""
    nq = robot_model.num_chain_dof
    rng = np.random.default_rng(42)
    q = rng.uniform(-0.5, 0.5, nq)
    v = rng.uniform(-0.1, 0.1, nq)
    a = rng.uniform(-0.1, 0.1, nq)
    tau = dynamics.inverse_dynamics(robot_model, q, v, a)
    M = dynamics.mass_matrix(robot_model, q)
    nle = dynamics.nonlinear_effects(robot_model, q, v)
    tau_reconstructed = M @ a + nle
    assert np.allclose(tau, tau_reconstructed, atol=1e-10), (
        f"tau = M@ddq + nle failed: max |tau - (M@a + nle)| = {np.max(np.abs(tau - tau_reconstructed))}"
    )


def test_forward_inverse_roundtrip(robot_model):
    """Self-consistency: ddq = FD(q,v,tau) then tau' = ID(q,v,ddq) should equal tau."""
    nq = robot_model.num_chain_dof
    rng = np.random.default_rng(43)
    q = rng.uniform(-0.5, 0.5, nq)
    v = rng.uniform(-0.1, 0.1, nq)
    tau = rng.uniform(-1.0, 1.0, nq)
    ddq = dynamics.forward_dynamics(robot_model, q, v, tau)
    tau_back = dynamics.inverse_dynamics(robot_model, q, v, ddq)
    assert np.allclose(tau, tau_back, atol=1e-9), (
        f"ID-FD roundtrip failed: max |tau - tau'| = {np.max(np.abs(tau - tau_back))}"
    )


def test_gravity_equals_rnea_zero_vel_acc(robot_model):
    """gravity(q) = inverse_dynamics(q, 0, 0)."""
    nq = robot_model.num_chain_dof
    q = np.random.default_rng(44).uniform(-0.3, 0.3, nq)
    g = dynamics.gravity(robot_model, q)
    tau_zero = dynamics.inverse_dynamics(robot_model, q, np.zeros(nq), np.zeros(nq))
    assert np.allclose(g, tau_zero, atol=1e-12)
