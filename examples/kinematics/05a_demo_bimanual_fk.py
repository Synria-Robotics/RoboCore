"""Bimanual Forward Kinematics Demo

This demo demonstrates bimanual forward kinematics computation for dual-arm systems.
It shows how to compute FK for both arms independently, with relative constraints, and mirror mode.

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
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_bimanual_fk(robot_model, backend, q_full, left_end_link, right_end_link, base_link='base_link'):
    """Compute bimanual forward kinematics using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy' or 'torch')
    :param q_full: Full joint configuration [nq] (includes all joints)
    :param left_end_link: Left arm end-effector link name
    :param right_end_link: Right arm end-effector link name
    :param base_link: Base link name
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    # Compute FK for both arms using unified config space
    T_left = robot_model.fk(q_full, base_link=base_link, end_link=left_end_link, return_end=True)
    T_right = robot_model.fk(q_full, base_link=base_link, end_link=right_end_link, return_end=True)

    elapsed_time = time.time() - start_time

    T_left = to_numpy(T_left)
    T_right = to_numpy(T_right)

    pos_left = T_left[:3, 3]
    pos_right = T_right[:3, 3]
    rot_left = T_left[:3, :3]
    rot_right = T_right[:3, :3]
    euler_left = matrix_to_euler(rot_left, seq='XYZ')
    euler_right = matrix_to_euler(rot_right, seq='XYZ')
    quat_left = matrix_to_quaternion(rot_left)
    quat_right = matrix_to_quaternion(rot_right)

    # Compute relative transform if needed
    T_rel = np.linalg.inv(T_left) @ T_right

    ret = {
        'left': {
            'position': pos_left,
            'rotation': rot_left,
            'euler': euler_left,
            'quat': quat_left,
            'transform': T_left,
        },
        'right': {
            'position': pos_right,
            'rotation': rot_right,
            'euler': euler_right,
            'quat': quat_right,
            'transform': T_right,
        },
        'relative': {
            'transform': T_rel,
            'position': T_rel[:3, 3],
            'rotation': T_rel[:3, :3],
            'euler': matrix_to_euler(T_rel[:3, :3], seq='XYZ'),
            'quat': matrix_to_quaternion(T_rel[:3, :3]),
        },
        'time': elapsed_time
    }

    return ret


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.left_base_link)

    if args.verbose:
        beauty_print("Robot Model (Unified Config Space):", type="module")
        robot_model.summary(show_chain=True)
        beauty_print(f"Total DOF: {robot_model.num_dof}")
        beauty_print(f"Available chains: {len(robot_model.available_chains())}")

    # Build unified configuration vector
    # For now, assume q_left and q_right are separate, need to combine them
    # In a real scenario, you'd have the full config including shared joints
    # Here we'll create a simple mapping (this may need adjustment based on actual robot structure)
    q_full = np.zeros(robot_model.num_dof)

    # Get chain joint indices
    left_indices = robot_model._get_joint_indices(args.left_base_link, args.left_end_link)
    right_indices = robot_model._get_joint_indices(args.right_base_link, args.right_end_link)

    # Map q_left and q_right to unified config (assuming no overlap for now)
    if len(args.q_left) <= len(left_indices):
        q_full[left_indices[:len(args.q_left)]] = args.q_left
    if len(args.q_right) <= len(right_indices):
        q_full[right_indices[:len(args.q_right)]] = args.q_right

    # Compute with both backends
    results_np = compute_bimanual_fk(robot_model, 'numpy', q_full,
                                     args.left_end_link, args.right_end_link, args.left_base_link)
    results_torch = compute_bimanual_fk(robot_model, 'torch', q_full,
                                        args.left_end_link, args.right_end_link, args.left_base_link)

    # Display results
    beauty_print(f"Left Arm End-Effector Position (m):")
    pos_left_np = to_numpy(results_np['left']['position'])
    pos_left_torch = to_numpy(results_torch['left']['position'])
    print(f"  NumPy:  {beauty_print_array(pos_left_np)}")
    print(f"  Torch:  {beauty_print_array(pos_left_torch)}")
    pos_diff_left = np.linalg.norm(pos_left_np - pos_left_torch)
    print(f"  Diff:   {pos_diff_left:.6e}")

    beauty_print(f"Left Arm End-Effector Orientation (Euler XYZ, radians):")
    euler_left_np = to_numpy(results_np['left']['euler'])
    euler_left_torch = to_numpy(results_torch['left']['euler'])
    print(f"  NumPy:  {beauty_print_array(euler_left_np)}")
    print(f"  Torch:  {beauty_print_array(euler_left_torch)}")
    euler_diff_left = np.linalg.norm(euler_left_np - euler_left_torch)
    print(f"  Diff:   {euler_diff_left:.6e}")

    beauty_print(f"Left Arm End-Effector Orientation (Quaternion xyzw):")
    quat_left_np = to_numpy(results_np['left']['quat'])
    quat_left_torch = to_numpy(results_torch['left']['quat'])
    print(f"  NumPy:  {beauty_print_array(quat_left_np, precision=6)}")
    print(f"  Torch:  {beauty_print_array(quat_left_torch, precision=6)}")
    quat_diff_left = np.linalg.norm(quat_left_np - quat_left_torch)
    print(f"  Diff:   {quat_diff_left:.6e}")

    beauty_print(f"Right Arm End-Effector Position (m):")
    pos_right_np = to_numpy(results_np['right']['position'])
    pos_right_torch = to_numpy(results_torch['right']['position'])
    print(f"  NumPy:  {beauty_print_array(pos_right_np)}")
    print(f"  Torch:  {beauty_print_array(pos_right_torch)}")
    pos_diff_right = np.linalg.norm(pos_right_np - pos_right_torch)
    print(f"  Diff:   {pos_diff_right:.6e}")

    beauty_print(f"Right Arm End-Effector Orientation (Euler XYZ, radians):")
    euler_right_np = to_numpy(results_np['right']['euler'])
    euler_right_torch = to_numpy(results_torch['right']['euler'])
    print(f"  NumPy:  {beauty_print_array(euler_right_np)}")
    print(f"  Torch:  {beauty_print_array(euler_right_torch)}")
    euler_diff_right = np.linalg.norm(euler_right_np - euler_right_torch)
    print(f"  Diff:   {euler_diff_right:.6e}")

    beauty_print(f"Right Arm End-Effector Orientation (Quaternion xyzw):")
    quat_right_np = to_numpy(results_np['right']['quat'])
    quat_right_torch = to_numpy(results_torch['right']['quat'])
    print(f"  NumPy:  {beauty_print_array(quat_right_np, precision=6)}")
    print(f"  Torch:  {beauty_print_array(quat_right_torch, precision=6)}")
    quat_diff_right = np.linalg.norm(quat_right_np - quat_right_torch)
    print(f"  Diff:   {quat_diff_right:.6e}")

    if 'relative' in results_np:
        beauty_print(f"Relative Transform Position (m):")
        pos_rel_np = to_numpy(results_np['relative']['position'])
        pos_rel_torch = to_numpy(results_torch['relative']['position'])
        print(f"  NumPy:  {beauty_print_array(pos_rel_np)}")
        print(f"  Torch:  {beauty_print_array(pos_rel_torch)}")
        pos_diff_rel = np.linalg.norm(pos_rel_np - pos_rel_torch)
        print(f"  Diff:   {pos_diff_rel:.6e}")

        if 'euler' in results_np['relative']:
            beauty_print(f"Relative Transform Orientation (Euler XYZ, radians):")
            euler_rel_np = to_numpy(results_np['relative']['euler'])
            euler_rel_torch = to_numpy(results_torch['relative']['euler'])
            print(f"  NumPy:  {beauty_print_array(euler_rel_np)}")
            print(f"  Torch:  {beauty_print_array(euler_rel_torch)}")
            euler_diff_rel = np.linalg.norm(euler_rel_np - euler_rel_torch)
            print(f"  Diff:   {euler_diff_rel:.6e}")

        if 'quat' in results_np['relative']:
            beauty_print(f"Relative Transform Orientation (Quaternion xyzw):")
            quat_rel_np = to_numpy(results_np['relative']['quat'])
            quat_rel_torch = to_numpy(results_torch['relative']['quat'])
            print(f"  NumPy:  {beauty_print_array(quat_rel_np, precision=6)}")
            print(f"  Torch:  {beauty_print_array(quat_rel_torch, precision=6)}")
            quat_diff_rel = np.linalg.norm(quat_rel_np - quat_rel_torch)
            print(f"  Diff:   {quat_diff_rel:.6e}")

    if 'mirror' in results_np:
        beauty_print(f"Mirror Transform Position (m):")
        pos_mirror_np = to_numpy(results_np['mirror']['position'])
        pos_mirror_torch = to_numpy(results_torch['mirror']['position'])
        print(f"  NumPy:  {beauty_print_array(pos_mirror_np)}")
        print(f"  Torch:  {beauty_print_array(pos_mirror_torch)}")
        pos_diff_mirror = np.linalg.norm(pos_mirror_np - pos_mirror_torch)
        print(f"  Diff:   {pos_diff_mirror:.6e}")

        if 'euler' in results_np['mirror']:
            beauty_print(f"Mirror Transform Orientation (Euler XYZ, radians):")
            euler_mirror_np = to_numpy(results_np['mirror']['euler'])
            euler_mirror_torch = to_numpy(results_torch['mirror']['euler'])
            print(f"  NumPy:  {beauty_print_array(euler_mirror_np)}")
            print(f"  Torch:  {beauty_print_array(euler_mirror_torch)}")
            euler_diff_mirror = np.linalg.norm(euler_mirror_np - euler_mirror_torch)
            print(f"  Diff:   {euler_diff_mirror:.6e}")

        if 'quat' in results_np['mirror']:
            beauty_print(f"Mirror Transform Orientation (Quaternion xyzw):")
            quat_mirror_np = to_numpy(results_np['mirror']['quat'])
            quat_mirror_torch = to_numpy(results_torch['mirror']['quat'])
            print(f"  NumPy:  {beauty_print_array(quat_mirror_np, precision=6)}")
            print(f"  Torch:  {beauty_print_array(quat_mirror_torch, precision=6)}")
            quat_diff_mirror = np.linalg.norm(quat_mirror_np - quat_mirror_torch)
            print(f"  Diff:   {quat_diff_mirror:.6e}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']*1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']*1000:.4f} ms")
    print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    # Bessica is a dual-arm robot
    model_path = get_model_path("Bessica_D", version="v1_1", variant="skeleton", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    # parser.add_argument('--q-left', type=float, nargs='+', default=[0.32980586939541284, 0.342077715698498, 0.3390097541227267, 0.34054373491061235, 0.3528155812136975, 0.3497476196379262, 0.342077715698498],
    parser.add_argument('--q-left', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.1],
                        help='Left joint angles in radians')
    parser.add_argument('--q-right', type=float, nargs='+', default=[-0.1, -0.2, 0.3, 0.0, -0.5, 0.2, 0.1],
                        help='Right joint angles in radians')
    # Note: mode parameter removed - unified interface always computes both arms independently
    # Relative and mirror transforms can be computed from the results if needed
    parser.add_argument('--verbose', action='store_true', help='Show detailed model information')
    args = parser.parse_args()
    main(args)

    """_result_
    [RoboCore:INFO] Left Arm End-Effector Position (m):
      NumPy:  [+0.22307, +0.22289, +0.28379]
      Torch:  [+0.22307, +0.22289, +0.28379]
      Diff:   2.264633e-12
    [RoboCore:INFO] Left Arm End-Effector Orientation (Euler XYZ, radians):
      NumPy:  [-1.87391, +0.49223, -0.04775]
      Torch:  [-1.87391, +0.49223, -0.04775]
      Diff:   6.257567e-11
    [RoboCore:INFO] Left Arm End-Effector Orientation (Quaternion xyzw):
      NumPy:  [-0.784701, +0.125594, -0.209971, +0.569546]
      Torch:  [-0.784701, +0.125594, -0.209971, +0.569546]
      Diff:   2.965773e-11
    [RoboCore:INFO] Right Arm End-Effector Position (m):
      NumPy:  [+0.17295, -0.26893, +0.24323]
      Torch:  [+0.17295, -0.26893, +0.24323]
      Diff:   6.157462e-12
    [RoboCore:INFO] Right Arm End-Effector Orientation (Euler XYZ, radians):
      NumPy:  [+1.17475, -0.42346, -0.16176]
      Torch:  [+1.17475, -0.42346, -0.16176]
      Diff:   7.290837e-11
    [RoboCore:INFO] Right Arm End-Effector Orientation (Quaternion xyzw):
      NumPy:  [+0.554162, -0.130586, -0.181829, +0.801742]
      Torch:  [+0.554162, -0.130586, -0.181829, +0.801742]
      Diff:   2.993312e-11
    [RoboCore:INFO] Relative Transform Position (m):
      NumPy:  [+0.16289, +0.19352, -0.42669]
      Torch:  [+0.16289, +0.19352, -0.42669]
      Diff:   2.695040e-11
    [RoboCore:INFO] Relative Transform Orientation (Euler XYZ, radians):
      NumPy:  [+3.05998, +0.07088, -0.16549]
      Torch:  [+3.05998, +0.07088, -0.16549]
      Diff:   2.284035e-11
    [RoboCore:INFO] Relative Transform Orientation (Quaternion xyzw):
      NumPy:  [+0.995004, +0.083970, +0.031912, +0.043555]
      Torch:  [+0.995004, +0.083970, +0.031912, +0.043555]
      Diff:   1.158060e-11
    [RoboCore:INFO] Computation Time:
      NumPy:  0.6530 ms
      Torch:  21.3387 ms
      Ratio:  32.68x
    """
