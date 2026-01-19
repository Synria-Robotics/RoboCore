"""Humanoid Inverse Kinematics Parallel Demo

This demo demonstrates parallel/batch humanoid inverse kinematics computation.
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
from robocore.modeling import RobotModel
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import *


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link)
    
    if args.verbose:
        beauty_print("Robot Model:", type="module")
        robot_model.summary(show_chain=True)

    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]

    # Generate random joint configurations and compute target poses using FK
    num_configs = args.num_configs
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    joint_configs = robot_model.random_q_full_batch(num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {num_configs} random joint configuration(s)")

    # Generate target poses from joint configurations using FK
    # These poses are guaranteed to be reachable since they come from FK
    beauty_print("Computing Forward Kinematics to generate reachable target poses", type="module", centered=True)
    target_poses_list = []
    for i in range(num_configs):
        targets = {}
        for end_link in end_links:
            T = robot_model.fk(joint_configs[i], base_link=args.base_link, end_link=end_link, return_end=True)
            targets[end_link] = np.array(T)
        target_poses_list.append(targets)
    beauty_print(f"Computed {num_configs} reachable target pose(s) from FK")

    # Single IK example
    beauty_print("Single Humanoid IK Example", type="module", centered=True)
    targets_single = target_poses_list[0]
    
    start_time = time.time()
    result_single = robot_model.ik(
        targets=targets_single,
        end_links=end_links,
        method=args.method,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
        base_link=args.base_link,
    )
    single_time = time.time() - start_time
    
    beauty_print(f"Single IK time: {single_time:.6f} seconds")
    beauty_print(f"Success: {result_single.get('success', False)}")
    if result_single.get('success', False):
        beauty_print(f"Position error: {result_single.get('pos_err', 0.0):.6e} m")
        beauty_print(f"Orientation error: {result_single.get('ori_err', 0.0):.6e} rad")
        beauty_print(f"Iterations: {result_single.get('iters', 0)}")

    # Batch IK example
    beauty_print("Batch Humanoid IK Example", type="module", centered=True)
    
    start_time = time.time()
    results_batch = []
    for i in range(num_configs):
        result = robot_model.ik(
            targets=target_poses_list[i],
            end_links=end_links,
            method=args.method,
            num_initial_guesses=args.num_inits,
            initial_guess_strategy=args.init_strategy,
            initial_guess_scale=args.init_scale,
            random_seed=args.seed,
            base_link=args.base_link,
        )
        results_batch.append(result)
    batch_time = time.time() - start_time
    
    beauty_print(f"Batch IK time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/num_configs:.6f} seconds")
    
    if num_configs > 1:
        speedup = (single_time * num_configs) / batch_time
        beauty_print(f"Effective speedup: {speedup:.2f}x")
    
    # Display results
    beauty_print("Batch Results Summary", type="module", centered=True)
    successes = sum(1 for r in results_batch if r.get('success', False))
    beauty_print(f"Successful: {successes}/{len(results_batch)}")
    
    if args.show_details:
        for i, result in enumerate(results_batch[:args.max_display]):
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Success: {result.get('success', False)}")
            if result.get('success', False):
                q = result.get('q', [])
                beauty_print(f"  Joint angles (first 10): {beauty_print_array(np.array(q)[:10])}...")
                beauty_print(f"  Position error: {result.get('pos_err', 0.0):.6e} m")
                beauty_print(f"  Orientation error: {result.get('ori_err', 0.0):.6e} rad")
                beauty_print(f"  Iterations: {result.get('iters', 0)}")


if __name__ == "__main__":
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")

    parser = argparse.ArgumentParser(description="Humanoid Inverse Kinematics Parallel Demo - Batch IK processing with random joint configurations")
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
    parser.add_argument('--num-configs', type=int, default=100,
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
    parser.add_argument('--backend', type=str, default='numpy',
                        choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy)')
    parser.add_argument('--method', type=str, default='dls',
                        choices=['dls', 'pinv', 'transpose'],
                        help='IK method (default: dls)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed results for each configuration')
    parser.add_argument('--max-display', type=int, default=5,
                        help='Maximum number of configurations to display (default: 5)')
    args = parser.parse_args()

    main(args)
