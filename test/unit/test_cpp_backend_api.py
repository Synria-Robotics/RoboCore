"""C++ backend public API coverage."""

from __future__ import annotations

import numpy as np
import pytest

import robocore
from robocore.kinematics import forward_kinematics, inverse_kinematics, jacobian
from robocore.modeling import RobotModel


pytest.importorskip("robocore.kinematics.fk_utils.fk_solver_cpp")
pytest.importorskip("robocore.kinematics.jacobian_utils.jacobian_solver_cpp")
pytest.importorskip("robocore.kinematics.ik_utils.ik_solver_cpp")


@pytest.fixture(scope="module")
def model(alicia_urdf_path):
    return RobotModel(alicia_urdf_path, base_link="base_link", end_link="link6")


def test_backend_argument_does_not_leak_global_backend(model):
    robocore.set_backend("numpy")
    q = model.random_q(seed=0, scale=0.5)
    T_cpp = forward_kinematics(model, q, return_end=True, backend="cpp")
    T_np = forward_kinematics(model, q, return_end=True, backend="numpy")
    assert robocore.get_backend() == "numpy"
    assert np.allclose(T_cpp, T_np, atol=1e-9)


def test_cpp_fk_and_jacobian_batch_match_numpy(model):
    q_batch = model.random_q_batch(5, seed=1, scale=0.5)
    T_cpp = forward_kinematics(model, q_batch, return_end=True, backend="cpp")
    T_np = forward_kinematics(model, q_batch, return_end=True, backend="numpy")
    J_cpp = jacobian(model, q_batch, backend="cpp", method="analytic")
    J_np = jacobian(model, q_batch, backend="numpy", method="analytic")
    assert T_cpp.shape == (5, 4, 4)
    assert J_cpp.shape == (5, 6, model.num_chain_dof)
    assert np.allclose(T_cpp, T_np, atol=1e-9)
    assert np.allclose(J_cpp, J_np, atol=1e-9)


def test_cpp_target_link_jacobian_matches_numpy(model):
    q = model.random_q(seed=2, scale=0.5)
    J_cpp = jacobian(model, q, target_link="link4", backend="cpp", method="analytic")
    J_np = jacobian(model, q, target_link="link4", backend="numpy", method="analytic")
    assert J_cpp.shape == J_np.shape
    assert np.allclose(J_cpp, J_np, atol=1e-9)


def test_cpp_ik_result_has_stable_status_fields(model):
    q_target = model.random_q(seed=3, scale=0.4)
    target = forward_kinematics(model, q_target, return_end=True, backend="cpp")
    q0 = model.random_q(seed=4, scale=0.4)
    result = inverse_kinematics(
        model,
        target,
        q0,
        backend="cpp",
        method="dls",
        max_iters=100,
        pos_tol=1e-3,
        ori_tol=1e-3,
    )
    for key in ("q", "success", "iters", "pos_err", "ori_err", "status", "message"):
        assert key in result
    assert result["status"] in {"success", "max_iters", "limit", "singular", "unreachable"}


def test_unreachable_ik_reports_failure_status(model):
    target = np.eye(4)
    target[:3, 3] = [100.0, 100.0, 100.0]
    q0 = np.zeros(model.num_chain_dof)
    result = inverse_kinematics(model, target, q0, backend="cpp", method="dls", max_iters=5)
    assert result["success"] is False
    assert result["status"] in {"max_iters", "limit", "singular", "unreachable"}
    assert isinstance(result["message"], str) and result["message"]
