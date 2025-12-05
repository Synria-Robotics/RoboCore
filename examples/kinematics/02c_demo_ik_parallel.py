"""Inverse Kinematics Parallel Demo

This demo demonstrates parallel/batch inverse kinematics computation.
It shows how to use batch processing for multiple target poses.

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

    # Load robot model
    robot_model = RobotModel(str(args.model_path), end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    # Generate target poses from joint configurations
    joint_configs = args.joint_angles
    if isinstance(joint_configs[0], list):
        pass
    else:
        joint_configs = [joint_configs]

    beauty_print("Generating Target Poses", type="module", centered=True)
    target_poses = []
    for q in joint_configs:
        T = forward_kinematics(robot_model, q, return_end=True)
        target_poses.append(T)
    target_poses = np.array(target_poses)
    beauty_print(f"Generated {len(target_poses)} target pose(s)")

    # Single IK example
    beauty_print("Single IK Example", type="module", centered=True)
    target_single = target_poses[0]
    q0_single = np.zeros(robot_model.num_chain_dof)
    
    start_time = time.time()
    result_single = inverse_kinematics(robot_model, target_single, q0_single, method=args.method)
    single_time = time.time() - start_time
    
    beauty_print(f"Single IK time: {single_time:.6f} seconds")
    beauty_print(f"Success: {result_single['success']}")
    if result_single['success']:
        beauty_print(f"Position error: {result_single.get('pos_err', 0.0):.6e} m")
        beauty_print(f"Orientation error: {result_single.get('ori_err', 0.0):.6e} rad")
        beauty_print(f"Iterations: {result_single.get('iters', 0)}")

    # Batch IK example
    beauty_print("Batch IK Example", type="module", centered=True)
    q0_batch = np.zeros((len(target_poses), robot_model.num_chain_dof))
    
    start_time = time.time()
    results_batch = inverse_kinematics(robot_model, target_poses, q0_batch, method=args.method)
    batch_time = time.time() - start_time
    
    beauty_print(f"Batch IK time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/len(target_poses):.6f} seconds")
    
    if len(target_poses) > 1:
        speedup = (single_time * len(target_poses)) / batch_time
        beauty_print(f"Effective speedup: {speedup:.2f}x")
    
    # Display results
    beauty_print("Batch Results Summary", type="module", centered=True)
    successes = sum(1 for r in results_batch if r['success'])
    beauty_print(f"Successful: {successes}/{len(results_batch)}")
    
    if args.show_details:
        for i, result in enumerate(results_batch):
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Success: {result['success']}")
            if result['success']:
                beauty_print(f"  Joint angles: {beauty_print_array(np.array(result['q']))}")
                beauty_print(f"  Position error: {result.get('pos_err', 0.0):.6e} m")
                beauty_print(f"  Orientation error: {result.get('ori_err', 0.0):.6e} rad")
                beauty_print(f"  Iterations: {result.get('iters', 0)}")


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Inverse Kinematics Parallel Demo - Batch IK processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configurations
  python 01d_demo_ik_parallel.py

  # Use torch backend for better batch performance
  python 01d_demo_ik_parallel.py --backend torch

  # Show detailed results
  python 01d_demo_ik_parallel.py --show-details
        """
    )
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', 
                        default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.2, 0.3, -0.4, 0.1, 0.6, -0.3,
                                 0.0, 0.1, -0.2, 0.0, 0.4, -0.1],
                        help='Joint angles in radians (flattened list, will be reshaped)')
    parser.add_argument('--num-joints', type=int, default=6,
                        help='Number of joints per configuration (default: 6)')
    parser.add_argument('--backend', type=str, default='torch',
                        choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: torch)')
    parser.add_argument('--method', type=str, default='dls',
                        choices=['dls', 'pinv', 'transpose'],
                        help='IK method (default: dls)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed results for each configuration')
    args = parser.parse_args()
    
    # Reshape joint angles into list of configurations
    num_joints = args.num_joints
    joint_angles_flat = args.joint_angles
    if len(joint_angles_flat) % num_joints != 0:
        raise ValueError(f"Total number of joint angles ({len(joint_angles_flat)}) must be divisible by num-joints ({num_joints})")
    
    num_batch = len(joint_angles_flat) // num_joints
    args.joint_angles = [joint_angles_flat[i*num_joints:(i+1)*num_joints] 
                        for i in range(num_batch)]
    
    main(args)

