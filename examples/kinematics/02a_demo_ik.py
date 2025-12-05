"""Module

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

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import *


def main(args):
    backend = args.backend
    rc.set_backend(backend)
    
    start_time = time.time()

    robot_model = RobotModel(str(args.model_path), end_link=args.end_link)
    T_fk = np.zeros((4, 4))

    T_fk[:3, 3] = args.end_pose[:3]
    T_fk[3, 3] = 1.0
    T_fk[:3, :3] = quaternion_to_matrix(args.end_pose[3:])

    # Use fixed initial guess to match demo_ik_pk.py
    q_init = np.array([+0.86066, -0.19202, +1.12657, +0.62005, -1.27493, +1.49421])
    beauty_print(f"Initial Guess (radians):")
    print(f"  q_init = {beauty_print_array(q_init)}")

    # Solve IK using DLS method with default (adaptive) parameters
    # Note: RoboCore applies joint limits from URDF, while pytorch_kinematics does not.
    # Use default adaptive parameters for better convergence with joint limits.
    ik_result = inverse_kinematics(
        robot_model,
        T_fk,
        q_init,
        method='dls',
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
        use_analytic_jacobian=True,
    )

    beauty_print(f"IK Solution:")
    print(f"  Success: {ik_result['success']}")
    print(f"  Iterations: {ik_result['iters']}")
    print(f"  Position Error: {ik_result['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
    if 'err_norm' in ik_result:
        print(f"  Total Error: {ik_result['err_norm']:.6e}")
    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
    beauty_print(f"Solved Joint Angles (degrees):")
    print(f"  q_ik = {beauty_print_array(np.rad2deg(ik_result['q']))}")

    # Verify solution with FK
    T_verify = forward_kinematics(robot_model, ik_result['q'], return_end=True)
    beauty_print(f"Verification (FK of IK solution):")
    print(f"  Position: {beauty_print_array(T_verify[:3, 3])}")
    print(f"  Position Error: {np.linalg.norm(T_verify[:3, 3] - T_fk[:3, 3]):.6e} m")

    end_time = time.time()
    beauty_print(f"RoboCore IK Computation Time: {end_time - start_time: .6f} seconds")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6',
                        help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+',
                        default=[0.17006, 0.01704, 0.20533, 0.042114, 0.828366, 0.083037, 0.552396],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--backend', type=str, default='numpy',
                        help='Backend to use for computation (default: numpy)')
    # Target joint angles: [0.1, 0.2, -0.3, 0.0, 0.5, -0.2]
    args = parser.parse_args()
    main(args)
