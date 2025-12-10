"""Bimanual Jacobian Parallel Demo

This demo demonstrates parallel/batch bimanual Jacobian computation.
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
from robocore.kinematics.bimanual import bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print_array, beauty_print


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot models
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)
    
    if args.verbose:
        beauty_print("Left Arm Model:", type="module")
        left_model.summary(show_chain=True)
        beauty_print("Right Arm Model:", type="module")
        right_model.summary(show_chain=True)

    # Generate random joint configurations
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    joint_configs_left = left_model.random_q_batch(args.num_configs, seed=args.seed, scale=args.scale)
    joint_configs_right = right_model.random_q_batch(args.num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {args.num_configs} random joint configuration(s) for each arm")
    num_batch = args.num_configs

    # Single Jacobian example
    beauty_print("Single Bimanual Jacobian Example", type="module", centered=True)
    q_left_single = joint_configs_left[0]
    q_right_single = joint_configs_right[0]
    
    start_time = time.time()
    J_single = bimanual_jacobian(left_model, right_model, q_left_single, q_right_single, mode=args.mode)
    single_time = time.time() - start_time
    
    beauty_print(f"Single Jacobian time: {single_time:.6f} seconds")
    beauty_print(f"Jacobian shape: {J_single.shape}")
    J_single_np = np.array(J_single) if not isinstance(J_single, np.ndarray) else J_single
    beauty_print(f"Condition number: {np.linalg.cond(J_single_np):.2e}")

    # Batch Jacobian example
    beauty_print("Batch Bimanual Jacobian Example", type="module", centered=True)
    q_left_batch = np.array(joint_configs_left)
    q_right_batch = np.array(joint_configs_right)
    
    start_time = time.time()
    J_batch = bimanual_jacobian(left_model, right_model, q_left_batch, q_right_batch, mode=args.mode)
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
        J_batch_np = np.array(J_batch) if not isinstance(J_batch, np.ndarray) else J_batch
        for i in range(min(num_batch, args.max_display)):
            J = J_batch_np[i] if J_batch_np.ndim == 3 else J_single_np
            q_left_config = joint_configs_left[i]
            q_right_config = joint_configs_right[i]
            
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Left joint angles: {beauty_print_array(np.array(q_left_config))}")
            beauty_print(f"  Right joint angles: {beauty_print_array(np.array(q_right_config))}")
            beauty_print(f"  Condition number: {np.linalg.cond(J):.2e}")
            
            if args.show_matrix:
                beauty_print(f"  Jacobian Matrix (first 6 rows):")
                print(beauty_print_array(J[:6, :], precision=6))
                if J.shape[0] > 6:
                    beauty_print(f"  Jacobian Matrix (last 6 rows):")
                    print(beauty_print_array(J[6:, :], precision=6))


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Jacobian Parallel Demo - Batch Jacobian computation")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--num-configs', type=int, default=1000,
                        help='Number of random joint configurations to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Scaling factor for joint range sampling (0.0 to 1.0, default: 0.8)')
    parser.add_argument('--backend', type=str, default='torch',
                        choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: torch)')
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative', 'mirror'],
                        help='Jacobian mode: indep (independent), relative (relative transform), mirror (mirror mode)')
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

