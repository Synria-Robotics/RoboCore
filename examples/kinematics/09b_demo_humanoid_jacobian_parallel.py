"""Humanoid Jacobian Parallel Demo

This demo demonstrates parallel/batch humanoid Jacobian computation.
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
from robocore.modeling import RobotModel
from robocore.utils.beauty_logger import beauty_print_array, beauty_print


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link)
    
    if args.verbose:
        beauty_print("Robot Model:", type="module")
        robot_model.summary(show_chain=True)

    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]

    # Generate random joint configurations (full DOF)
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    joint_configs = robot_model.random_q_full_batch(args.num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {args.num_configs} random joint configuration(s)")
    num_batch = args.num_configs

    # Single Jacobian example
    beauty_print("Single Humanoid Jacobian Example", type="module", centered=True)
    q_single = joint_configs[0]
    
    start_time = time.time()
    J_list = []
    for end_link in end_links:
        J = robot_model.jacobian(q_single, base_link=args.base_link, end_link=end_link)
        J_list.append(J)
    J_single = np.vstack([np.array(j) if not isinstance(j, np.ndarray) else j for j in J_list])
    single_time = time.time() - start_time
    
    beauty_print(f"Single Jacobian time: {single_time:.6f} seconds")
    beauty_print(f"Jacobian shape: {J_single.shape}")
    beauty_print(f"Condition number: {np.linalg.cond(J_single):.2e}")

    # Batch Jacobian example
    beauty_print("Batch Humanoid Jacobian Example", type="module", centered=True)
    
    start_time = time.time()
    J_batch_list = []
    for i in range(num_batch):
        J_list = []
        for end_link in end_links:
            J = robot_model.jacobian(joint_configs[i], base_link=args.base_link, end_link=end_link)
            J_list.append(J)
        J_combined = np.vstack([np.array(j) if not isinstance(j, np.ndarray) else j for j in J_list])
        J_batch_list.append(J_combined)
    batch_time = time.time() - start_time
    
    beauty_print(f"Batch Jacobian time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/num_batch:.6f} seconds")
    
    if num_batch > 1:
        speedup = (single_time * num_batch) / batch_time
        beauty_print(f"Effective speedup: {speedup:.2f}x")

    # Display results
    if args.show_details:
        beauty_print("Results for Each Configuration", type="module", centered=True)
        for i in range(min(num_batch, args.max_display)):
            J = J_batch_list[i]
            q_config = joint_configs[i]
            
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Joint angles (first 10): {beauty_print_array(np.array(q_config)[:10])}...")
            beauty_print(f"  Condition number: {np.linalg.cond(J):.2e}")
            
            if args.show_matrix:
                beauty_print(f"  Jacobian Matrix (first 6 rows):")
                print(beauty_print_array(J[:6, :], precision=6))


if __name__ == "__main__":
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")

    parser = argparse.ArgumentParser(description="Humanoid Jacobian Parallel Demo - Batch Jacobian computation")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--num-configs', type=int, default=1000,
                        help='Number of random joint configurations to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Scaling factor for joint range sampling (0.0 to 1.0, default: 0.8)')
    parser.add_argument('--backend', type=str, default='torch',
                        choices=['numpy', 'torch', 'cpp'],
                        help='Backend to use for computation (default: torch)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed results for each configuration')
    parser.add_argument('--show-matrix', action='store_true',
                        help='Show full Jacobian matrices')
    parser.add_argument('--max-display', type=int, default=5,
                        help='Maximum number of configurations to display (default: 5)')
    args = parser.parse_args()

    main(args)
