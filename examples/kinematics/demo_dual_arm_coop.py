"""Dual-arm cooperative IK demo (Phase 1).

Showcases:
- Single URDF parse, spawn left/right chains.
- Simultaneous absolute IK: move both end-effectors to nearby offsets.

Run:
    python examples/kinematics/demo_dual_arm_coop.py
"""
from __future__ import annotations
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print
from robocore.kinematics.bimanual import dual_fk, dual_ik

URDF_REL = "assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"

def main():
    beauty_print("Dual-Arm Cooperative IK Demo (with workspace check)")
    base = RobotModel(get_robocore_path(URDF_REL))
    leaves = base.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper_left_finger' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper_left_finger' in l)

    left = base.spawn_chain(left_end)
    right = base.spawn_chain(right_end)

    # Pre-compute workspaces (optional but recommended for multiple IK calls)
    beauty_print("Pre-computing workspaces...")
    left.compute_workspace(num_samples=4000, verbose=True)
    right.compute_workspace(num_samples=4000, verbose=True)

    # Initial random configs
    qL0 = left.random_q(scale=0.2)
    qR0 = right.random_q(scale=0.2)

    # Current poses
    Tcur = dual_fk(left, right, qL0, qR0)
    T_left = Tcur['left'].copy()
    T_right = Tcur['right'].copy()

    # Create slight target offsets (simulate cooperative reach)
    # Reduced to stay within workspace
    T_left_target = T_left.copy()
    T_left_target[0:3, 3] += np.array([0.015, 0.025, 0.008])
    T_right_target = T_right.copy()
    T_right_target[0:3, 3] += np.array([-0.02, 0.015, -0.01])

    # Check reachability before IK
    p_left = T_left_target[0:3, 3]
    p_right = T_right_target[0:3, 3]
    reach_left = left.is_point_reachable(p_left, tolerance=0.05)
    reach_right = right.is_point_reachable(p_right, tolerance=0.05)
    beauty_print(f"Target reachability: Left={reach_left}, Right={reach_right}")

    result = dual_ik(
        left, right,
        target_left=T_left_target,
        target_right=T_right_target,
        q0_left=qL0, q0_right=qR0,
        max_iters=100,
        pos_tol=5e-4,
        ori_tol=5e-4,
        method='dls',
        check_workspace=True,  # Enable automatic workspace check
        workspace_tolerance=0.08,
    )

    beauty_print(f"IK success={result['success']} iters={result['iters']}")
    beauty_print(f"Left pos_err={result['pos_err_left']:.2e} ori_err={result['ori_err_left']:.2e}")
    beauty_print(f"Right pos_err={result['pos_err_right']:.2e} ori_err={result['ori_err_right']:.2e}")

    # Quick forward check
    qL_sol = result['q_left']
    qR_sol = result['q_right']
    T_after = dual_fk(left, right, qL_sol, qR_sol)
    print("Left final translation:", T_after['left'][0:3, 3])
    print("Right final translation:", T_after['right'][0:3, 3])

    beauty_print("Demo complete.")

if __name__ == '__main__':
    main()
