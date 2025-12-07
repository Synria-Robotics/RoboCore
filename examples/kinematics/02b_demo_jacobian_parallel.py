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

    # Load robot model
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    # Generate random joint configurations
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    joint_configs = robot_model.random_q_batch(args.num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {args.num_configs} random joint configuration(s)")
    num_batch = args.num_configs

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

    parser = argparse.ArgumentParser(description="Jacobian Parallel Demo - Batch Jacobian computation")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--num-configs', type=int, default=1000,
                        help='Number of random joint configurations to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Scaling factor for joint range sampling (0.0 to 1.0, default: 0.8)')
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

    main(args)

