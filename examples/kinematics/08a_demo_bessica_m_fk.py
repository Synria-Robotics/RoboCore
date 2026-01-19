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
    :param q_full: Full joint configuration [nq] (includes all joints including waist)
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

    # Compute relative transform
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
    # For Bessica_M_v1_0: q = [waist_joints, left_arm_joints, right_arm_joints]
    # The q_left and q_right already include waist joints
    q_full = np.zeros(robot_model.num_dof)
    
    # Get chain joint indices
    left_indices = robot_model._get_joint_indices(args.left_base_link, args.left_end_link)
    right_indices = robot_model._get_joint_indices(args.right_base_link, args.right_end_link)
    
    # Map q_left and q_right to unified config
    # Note: For Bessica_M, both chains share waist joints, so we need to handle this carefully
    if len(args.q_left) <= len(left_indices):
        q_full[left_indices[:len(args.q_left)]] = args.q_left
    if len(args.q_right) <= len(right_indices):
        # For right arm, we need to avoid overwriting shared joints
        # Only update non-shared joints
        right_only_indices = [idx for idx in right_indices if idx not in left_indices]
        if len(right_only_indices) > 0:
            # Extract right-arm-only joints from q_right (skip waist joints if they're already set)
            q_right_arm_only = args.q_right[len(left_indices):] if len(args.q_right) > len(left_indices) else args.q_right
            q_full[right_only_indices[:len(q_right_arm_only)]] = q_right_arm_only[:len(right_only_indices)]

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

    # Bessica_M is a dual-arm robot; use the canonical URDF from synriard
    model_path = get_model_path("Bessica_M", version="v1_0", variant=None, model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica_M v1_0 URDF)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    # Match link names exactly as defined in Bessica_M_v1_0.urdf
    parser.add_argument('--left-end-link', type=str, default='Left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='Right_arm_link7', help='Right arm end-effector link name')
    # Bessica_M_v1_0 kinematic chain (with base_link as base for both arms) includes:
    #  - Left arm:  Torso_joint1, Torso_joint2, Torso_joint3, Left_joint1..Left_joint7  → 10 DoF
    #  - Right arm: Torso_joint1, Torso_joint2, Torso_joint3, Right_joint1..Right_joint7 → 10 DoF
    parser.add_argument(
        '--q-left',
        type=float,
        nargs='+',
        # [Torso_joint1, Torso_joint2, Torso_joint3, Left_joint1..Left_joint7]
        default=[0.0, 0.0, 0.0,
                 0.32980586939541284, 0.342077715698498, 0.3390097541227267,
                 0.34054373491061235, 0.3528155812136975, 0.3497476196379262,
                 0.342077715698498],
        help='Left joint angles in radians (torso1-3 + 7 arm joints)',
    )
    parser.add_argument(
        '--q-right',
        type=float,
        nargs='+',
        # [Torso_joint1, Torso_joint2, Torso_joint3, Right_joint1..Right_joint7]
        default=[0.0, 0.0, 0.0, -0.1, -0.2, 0.3, 0.0, -0.5, 0.2, 0.1],
        help='Right joint angles in radians (torso1-3 + 7 arm joints)',
    )
    # Note: mode parameter removed - unified interface always computes both arms independently
    # Relative transform is automatically computed from the results
    parser.add_argument('--verbose', action='store_true', help='Show detailed model information')
    args = parser.parse_args()
    main(args)
