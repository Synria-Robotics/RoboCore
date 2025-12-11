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
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    # Generate random joint configurations
    num_configs = args.num_configs
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    target_poses = robot_model.random_pose_batch(num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {target_poses.shape[0]} random target pose(s)")

    # Single IK example
    beauty_print("Single IK Example", type="module", centered=True)
    target_single = target_poses[0]
    
    start_time = time.time()
    result_single = inverse_kinematics(
        robot_model, target_single,
        method=args.method,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
    )
    single_time = time.time() - start_time
    
    beauty_print(f"Single IK time: {single_time:.6f} seconds")
    beauty_print(f"Success: {result_single['success']}")
    if result_single['success']:
        beauty_print(f"Position error: {result_single.get('pos_err', 0.0):.6e} m")
        beauty_print(f"Orientation error: {result_single.get('ori_err', 0.0):.6e} rad")
        beauty_print(f"Iterations: {result_single.get('iters', 0)}")

    # Batch IK example
    beauty_print("Batch IK Example", type="module", centered=True)
    
    start_time = time.time()
    results_batch = inverse_kinematics(
        robot_model, target_poses,
        method=args.method,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
    )
    batch_time = time.time() - start_time
    
    beauty_print(f"Batch IK time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/num_configs:.6f} seconds")
    
    if num_configs > 1:
        speedup = (single_time * num_configs) / batch_time
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

    parser = argparse.ArgumentParser(description="Inverse Kinematics Parallel Demo - Batch IK processing with random joint configurations")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--num-configs', type=int, default=1000,
                        help='Number of random joint configurations to generate (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Scaling factor for joint range sampling (0.0 to 1.0, default: 0.8)')
    parser.add_argument('--num-inits', type=int, default=1,
                        help='Number of initial guesses to try per target (default: 1)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Strategy for generating initial guesses (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale factor for joint limits when generating guesses (0.0 to 1.0, default: 1.0)')
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

    main(args)

