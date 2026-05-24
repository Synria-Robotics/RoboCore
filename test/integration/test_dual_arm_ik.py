"""Integration test for dual-arm absolute IK (Phase 1).

Validates that simultaneous solving reaches both targets within tolerances.
"""
from __future__ import annotations
import pytest
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import dual_fk, dual_ik


def test_dual_arm_absolute_ik(bessica_urdf_path):
    base = RobotModel(bessica_urdf_path)
    leaves = base.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper_left_finger' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper_left_finger' in l)
    left = base.spawn_chain(left_end)
    right = base.spawn_chain(right_end)

    qL0 = left.random_q(scale=0.2)
    qR0 = right.random_q(scale=0.2)
    Tcur = dual_fk(left, right, qL0, qR0)

    # Targets: very small offsets to ensure they stay within workspace
    T_left_target = Tcur['left'].copy()
    T_left_target[0:3, 3] += np.array([0.005, -0.004, 0.006])  # ~0.9cm offset
    T_right_target = Tcur['right'].copy()
    T_right_target[0:3, 3] += np.array([-0.006, 0.005, -0.004])  # ~0.9cm offset

    res = dual_ik(left, right, T_left_target, T_right_target, q0_left=qL0, q0_right=qR0,
                  max_iters=150, pos_tol=4e-3, ori_tol=4e-3, method='dls',
                  check_workspace=False)  # Disable workspace check for convergence test
    assert res['success'], f"Dual IK did not converge: {res}"
    assert res['pos_err_left'] < 4e-3 and res['ori_err_left'] < 4e-3
    assert res['pos_err_right'] < 4e-3 and res['ori_err_right'] < 4e-3
