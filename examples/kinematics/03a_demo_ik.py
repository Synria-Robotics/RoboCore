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
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_ik(robot_model, backend, end_pose, q_init):
    """Compute inverse kinematics for given backend.
    
    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy' or 'torch')
    :param end_pose: Target end-effector pose [px, py, pz, qx, qy, qz, qw]
    :param q_init: Initial joint angles guess
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    T_fk = np.zeros((4, 4))
    T_fk[:3, 3] = end_pose[:3]
    T_fk[3, 3] = 1.0
    T_fk[:3, :3] = quaternion_to_matrix(end_pose[3:])

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

    T_verify = forward_kinematics(robot_model, ik_result['q'], return_end=True)
    elapsed_time = time.time() - start_time

    return {
        'success': ik_result['success'],
        'iters': ik_result['iters'],
        'pos_err': ik_result['pos_err'],
        'ori_err': ik_result['ori_err'],
        'err_norm': ik_result.get('err_norm', None),
        'q': ik_result['q'],
        'verify_pos': T_verify[:3, 3],
        'verify_pos_err': np.linalg.norm(T_verify[:3, 3] - T_fk[:3, 3]),
        'time': elapsed_time
    }


def main(args):
    q_init = np.array([+0.86066, -0.19202, +1.12657, +0.62005, -1.27493, +1.49421])
    beauty_print(f"Initial Guess (radians):")
    print(f"  q_init = {beauty_print_array(q_init)}")

    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    # Compute with both backends
    results_np = compute_ik(robot_model, 'numpy', args.end_pose, q_init)
    results_torch = compute_ik(robot_model, 'torch', args.end_pose, q_init)

    # Convert to numpy for comparison
    q_np = to_numpy(results_np['q'])
    q_torch = to_numpy(results_torch['q'])
    verify_pos_np = to_numpy(results_np['verify_pos'])
    verify_pos_torch = to_numpy(results_torch['verify_pos'])

    beauty_print(f"IK Solution:")
    print(f"  Success:  NumPy={results_np['success']}, Torch={results_torch['success']}")
    print(f"  Iterations:  NumPy={results_np['iters']}, Torch={results_torch['iters']}")
    print(f"  Position Error:  NumPy={results_np['pos_err']:.6e} m, Torch={results_torch['pos_err']:.6e} m")
    print(f"  Orientation Error:  NumPy={results_np['ori_err']:.6e} rad, Torch={results_torch['ori_err']:.6e} rad")
    if results_np['err_norm'] is not None:
        print(f"  Total Error:  NumPy={results_np['err_norm']:.6e}, Torch={results_torch['err_norm']:.6e}")

    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  NumPy:  {beauty_print_array(q_np)}")
    print(f"  Torch:  {beauty_print_array(q_torch)}")
    q_diff = np.linalg.norm(q_np - q_torch)
    print(f"  Diff:   {q_diff:.6e}")

    beauty_print(f"Solved Joint Angles (degrees):")
    print(f"  NumPy:  {beauty_print_array(np.rad2deg(q_np))}")
    print(f"  Torch:  {beauty_print_array(np.rad2deg(q_torch))}")

    beauty_print(f"Verification (FK of IK solution):")
    print(f"  Position:  NumPy={beauty_print_array(verify_pos_np)}, Torch={beauty_print_array(verify_pos_torch)}")
    print(f"  Position Error:  NumPy={results_np['verify_pos_err']:.6e} m, Torch={results_torch['verify_pos_err']:.6e} m")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']* 1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']* 1000:.4f} ms")
    print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+',
                        default=[-0.17006, -0.01704, 0.20533, 0.828399, -0.041455, -0.552330, 0.083476],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy, ignored - both are tested)')
    # Target joint angles: [0.1, 0.2, -0.3, 0.0, 0.5, -0.2]
    args = parser.parse_args()
    main(args)

    """_results_
    [RoboCore:INFO] Initial Guess (radians):
    q_init = [+0.86066, -0.19202, +1.12657, +0.62005, -1.27493, +1.49421]
    [RoboCore:INFO] IK Solution:
    Success: True
    Iterations: 18
    Position Error: 2.956047e-05 m
    Orientation Error: 6.537617e-05 rad
    Total Error: 7.174863e-05
    [RoboCore:INFO] Solved Joint Angles (radians):
    q_ik = [+0.09731, +0.20009, -0.30000, +0.00510, +0.49988, -0.20344]
    [RoboCore:INFO] Solved Joint Angles (degrees):
    q_ik = [+5.57562, +11.46407, -17.18848, +0.29194, +28.64080, -11.65611]
    [RoboCore:INFO] Verification (FK of IK solution):
    Position: [-0.17006, -0.01701, +0.20533]
    Position Error: 2.956047e-05 m
    [RoboCore:INFO] RoboCore IK Computation Time:  0.007680 seconds
    """