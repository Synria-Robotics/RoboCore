"""Humanoid Forward Kinematics Demo

This demo demonstrates humanoid forward kinematics computation for humanoid systems.
It shows how to compute FK for four end-effectors (left/right thumbs and left/right toes) simultaneously.

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


def compute_humanoid_fk(robot_model, backend, q_full, end_links, base_link='pelvis'):
    """Compute humanoid forward kinematics using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy' or 'torch')
    :param q_full: Full joint configuration [nq] (includes all joints)
    :param end_links: List of end-effector link names [left_thumb, right_thumb, left_toe, right_toe]
    :param base_link: Base link name
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    # Compute FK for all four end-effectors using unified config space
    T_left_thumb = robot_model.fk(q_full, base_link=base_link, end_link=end_links[0], return_end=True)
    T_right_thumb = robot_model.fk(q_full, base_link=base_link, end_link=end_links[1], return_end=True)
    T_left_toe = robot_model.fk(q_full, base_link=base_link, end_link=end_links[2], return_end=True)
    T_right_toe = robot_model.fk(q_full, base_link=base_link, end_link=end_links[3], return_end=True)

    elapsed_time = time.time() - start_time

    T_left_thumb = to_numpy(T_left_thumb)
    T_right_thumb = to_numpy(T_right_thumb)
    T_left_toe = to_numpy(T_left_toe)
    T_right_toe = to_numpy(T_right_toe)

    def extract_pose(T):
        pos = T[:3, 3]
        rot = T[:3, :3]
        euler = matrix_to_euler(rot, seq='XYZ')
        quat = matrix_to_quaternion(rot)
        return {
            'position': pos,
            'rotation': rot,
            'euler': euler,
            'quat': quat,
            'transform': T,
        }

    ret = {
        'left_thumb': extract_pose(T_left_thumb),
        'right_thumb': extract_pose(T_right_thumb),
        'left_toe': extract_pose(T_left_toe),
        'right_toe': extract_pose(T_right_toe),
        'time': elapsed_time
    }

    return ret


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link)

    if args.verbose:
        beauty_print("Robot Model (Unified Config Space):", type="module")
        robot_model.summary(show_chain=True)
        beauty_print(f"Total DOF: {robot_model.num_dof}")

    # Build unified configuration vector
    # q_full = robot_model.random_q_full()
    q_full = np.zeros(robot_model.num_dof)
    
    # Get joint indices for each end-effector chain
    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]
    all_indices = set()
    for end_link in end_links:
        indices = robot_model._get_joint_indices(args.base_link, end_link)
        all_indices.update(indices)
    
    # Set joint angles (using provided joint angles or zeros)
    if args.joint_angles:
        if len(args.joint_angles) <= len(all_indices):
            sorted_indices = sorted(all_indices)
            for i, idx in enumerate(sorted_indices[:len(args.joint_angles)]):
                q_full[idx] = args.joint_angles[i]

    # Compute with both backends
    results_np = compute_humanoid_fk(robot_model, 'numpy', q_full, end_links, args.base_link)
    results_torch = compute_humanoid_fk(robot_model, 'torch', q_full, end_links, args.base_link)

    # Display results for each end-effector
    end_effector_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']
    
    for name, display_name in zip(end_effector_names, display_names):
        beauty_print(f"{display_name} End-Effector Position (m):")
        pos_np = to_numpy(results_np[name]['position'])
        pos_torch = to_numpy(results_torch[name]['position'])
        print(f"  NumPy:  {beauty_print_array(pos_np)}")
        print(f"  Torch:  {beauty_print_array(pos_torch)}")
        pos_diff = np.linalg.norm(pos_np - pos_torch)
        print(f"  Diff:   {pos_diff:.6e}")

        beauty_print(f"{display_name} End-Effector Orientation (Euler XYZ, radians):")
        euler_np = to_numpy(results_np[name]['euler'])
        euler_torch = to_numpy(results_torch[name]['euler'])
        print(f"  NumPy:  {beauty_print_array(euler_np)}")
        print(f"  Torch:  {beauty_print_array(euler_torch)}")
        euler_diff = np.linalg.norm(euler_np - euler_torch)
        print(f"  Diff:   {euler_diff:.6e}")

        beauty_print(f"{display_name} End-Effector Orientation (Quaternion xyzw):")
        quat_np = to_numpy(results_np[name]['quat'])
        quat_torch = to_numpy(results_torch[name]['quat'])
        print(f"  NumPy:  {beauty_print_array(quat_np, precision=6)}")
        print(f"  Torch:  {beauty_print_array(quat_torch, precision=6)}")
        quat_diff = np.linalg.norm(quat_np - quat_torch)
        print(f"  Diff:   {quat_diff:.6e}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']*1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']*1000:.4f} ms")
    if results_np['time'] > 0:
        print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from openrd import get_model_path

    # Unitree G1 is a humanoid robot
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")

    parser = argparse.ArgumentParser(description="Humanoid Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=None,
                        help='Joint angles in radians (optional, uses zeros if not provided)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed model information')
    args = parser.parse_args()
    main(args)
