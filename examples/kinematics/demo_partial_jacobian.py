"""Demo: Local / Partial Jacobian usage

Shows how to compute:
1. Full end-effector Jacobian
2. Intermediate link Jacobian (e.g. elbow)
3. Position-only Jacobian (mask rows)
4. Subset joints (e.g. wrist joints)
5. Combined local + masked + subset

Run:
    python examples/kinematics/demo_partial_jacobian.py --urdf <path> --elbow-link <linkname>
"""

import argparse
import numpy as np
from robotcore.modeling.robot_model import RobotModel
from robotcore.utils.path import get_robocore_path
from robotcore.utils.beauty_logger import beauty_print, beauty_print_array


def main(args):
    model = RobotModel(args.urdf, end_link=args.end_link)
    q = model.random_q()
    beauty_print("Random configuration q (rad):")
    print(beauty_print_array(q))

    # 1. Full Jacobian
    J_full = model.jacobian(q, method='analytic')
    beauty_print("Full end-effector Jacobian (6 x n):")
    print(J_full)

    # 2. Intermediate link Jacobian
    if args.elbow_link:
        J_elbow = model.jacobian(q, method='analytic', target_link=args.elbow_link)
        beauty_print(f"Intermediate link Jacobian ({args.elbow_link}) shape={J_elbow.shape}:")
        print(J_elbow)

    # 3. Position-only Jacobian (mask rows 0,1,2)
    J_pos = model.jacobian(q, method='analytic', row_mask=[1,1,1,0,0,0])
    beauty_print("Position-only Jacobian (3 x n):")
    print(J_pos)

    # 4. Subset of joints (e.g. last 3 joints as wrist)
    if model.num_dof() >= 3:
        wrist_idx = list(range(model.num_dof()-3, model.num_dof()))
        J_wrist = model.jacobian(q, method='analytic', joint_indices=wrist_idx)
        beauty_print(f"Wrist subset Jacobian (6 x 3) indices={wrist_idx}:")
        print(J_wrist)

    # 5. Combined example: position-only for elbow over wrist joints
    if args.elbow_link and model.num_dof() >= 3:
        J_combo = model.jacobian(q, method='analytic', target_link=args.elbow_link, row_mask=[1,1,1,0,0,0], joint_indices=wrist_idx)
        beauty_print(f"Combined local+mask+subset Jacobian shape={J_combo.shape}:")
        print(J_combo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', type=str, default=get_robocore_path("assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf"))
    parser.add_argument('--end-link', type=str, default='tool0')
    parser.add_argument('--elbow-link', type=str, default=None, help='Optional intermediate link name (e.g. elbow link)')
    args = parser.parse_args()
    main(args)
