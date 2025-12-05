"""Jacobian Parallel Demo

This demo demonstrates parallel/batch Jacobian computation.
It shows how to use batch processing for multiple joint configurations.

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
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print_array, beauty_print


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Parse joint configurations
    joint_configs = args.joint_angles
    if isinstance(joint_configs[0], list):
        pass
    else:
        joint_configs = [joint_configs]

    num_batch = len(joint_configs)
    beauty_print(f"Processing {num_batch} joint configuration(s) using {backend} backend")

    # Load robot model
    robot_model = RobotModel(str(args.model_path), end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    # Single Jacobian example
    beauty_print("Single Jacobian Example", type="module", centered=True)
    q_single = joint_configs[0]
    
    start_time = time.time()
    J_single = jacobian(robot_model, q_single, method=args.method)
    single_time = time.time() - start_time
    
    beauty_print(f"Single Jacobian time: {single_time:.6f} seconds")
    beauty_print(f"Jacobian shape: {J_single.shape}")
    beauty_print(f"Condition number: {np.linalg.cond(J_single):.2e}")

    # Batch Jacobian example
    beauty_print("Batch Jacobian Example", type="module", centered=True)
    q_batch = np.array(joint_configs)
    
    start_time = time.time()
    J_batch = jacobian(robot_model, q_batch, method=args.method)
    batch_time = time.time() - start_time
    
    beauty_print(f"Batch Jacobian time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/num_batch:.6f} seconds")
    beauty_print(f"Jacobian batch shape: {J_batch.shape}")
    
    if num_batch > 1:
        speedup = (single_time * num_batch) / batch_time
        beauty_print(f"Effective speedup: {speedup:.2f}x")

    # Display results
    if args.show_details:
        beauty_print("Results for Each Configuration", type="module", centered=True)
        for i in range(num_batch):
            J = J_batch[i] if J_batch.ndim == 3 else J_single
            q_config = joint_configs[i]
            
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Joint angles: {beauty_print_array(np.array(q_config))}")
            beauty_print(f"  Condition number: {np.linalg.cond(J):.2e}")
            
            if args.show_matrix:
                beauty_print(f"  Jacobian Matrix:")
                print(beauty_print_array(J, precision=6))


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Jacobian Parallel Demo - Batch Jacobian computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configurations
  python 01e_demo_jacobian_parallel.py

  # Use torch backend for better batch performance
  python 01e_demo_jacobian_parallel.py --backend torch

  # Show detailed results
  python 01e_demo_jacobian_parallel.py --show-details --show-matrix
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
    parser.add_argument('--method', type=str, default='analytic',
                        choices=['analytic', 'numeric', 'autograd'],
                        help='Jacobian method (default: analytic)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed results for each configuration')
    parser.add_argument('--show-matrix', action='store_true',
                        help='Show full Jacobian matrices')
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

