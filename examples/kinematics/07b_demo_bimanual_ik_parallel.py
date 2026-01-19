"""Bimanual Inverse Kinematics Parallel Demo

This demo demonstrates parallel/batch bimanual inverse kinematics computation.
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
from robocore.kinematics.bimanual import bimanual_inverse_kinematics, bimanual_forward_kinematics
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
    num_configs = args.num_configs
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    joint_configs_left = left_model.random_q_batch(num_configs, seed=args.seed, scale=args.scale)
    joint_configs_right = right_model.random_q_batch(num_configs, seed=args.seed, scale=args.scale)
    beauty_print(f"Generated {num_configs} random joint configuration(s) for each arm")

    # Generate target poses from joint configurations using FK
    beauty_print("Computing Forward Kinematics", type="module", centered=True)
    fk_results = []
    for i in range(num_configs):
        fk_result = bimanual_forward_kinematics(
            left_model, right_model,
            joint_configs_left[i], joint_configs_right[i],
            return_end=True, mode='indep'
        )
        fk_results.append({
            'left': np.array(fk_result['left']),
            'right': np.array(fk_result['right'])
        })
    beauty_print(f"Computed {num_configs} target pose(s) from FK")

    # Single IK example
    beauty_print("Single Bimanual IK Example", type="module", centered=True)
    target_left_single = fk_results[0]['left']
    target_right_single = fk_results[0]['right']
    
    start_time = time.time()
    result_single = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_single,
        target_right=target_right_single,
        method=args.method,
        coordination=args.coordination,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
    )
    single_time = time.time() - start_time
    
    beauty_print(f"Single IK time: {single_time:.6f} seconds")
    beauty_print(f"Success Left: {result_single['success_left']}")
    beauty_print(f"Success Right: {result_single['success_right']}")
    if result_single['success_left'] and result_single['success_right']:
        # Extract error information from res_left and res_right
        res_left = result_single['res_left']
        res_right = result_single['res_right']
        
        # Handle case where res_left/res_right might be dict or list
        if isinstance(res_left, list) and len(res_left) > 0:
            res_left = res_left[0]
        if isinstance(res_right, list) and len(res_right) > 0:
            res_right = res_right[0]
        
        # Get iters (use max of both arms if available)
        iters_left = res_left['iters']
        iters_right = res_right['iters']
        iters = max(iters_left, iters_right)
        
        pos_err_left = res_left['pos_err']
        pos_err_right = res_right['pos_err']
        ori_err_left = res_left['ori_err']
        ori_err_right = res_right['ori_err']
        
        print(f"Position error left: {pos_err_left:.6e} m")
        print(f"Position error right: {pos_err_right:.6e} m")
        print(f"Orientation error left: {ori_err_left:.6e} rad")
        print(f"Orientation error right: {ori_err_right:.6e} rad")
        print(f"Iterations: {iters}")

    # Batch IK example
    beauty_print("Batch Bimanual IK Example", type="module", centered=True)
    
    target_lefts = [fk['left'] for fk in fk_results]
    target_rights = [fk['right'] for fk in fk_results]
    
    start_time = time.time()
    results_batch = []
    for i in range(num_configs):
        result = bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=target_lefts[i],
            target_right=target_rights[i],
            method=args.method,
            coordination=args.coordination,
            num_initial_guesses=args.num_inits,
            initial_guess_strategy=args.init_strategy,
            initial_guess_scale=args.init_scale,
            random_seed=args.seed
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
    successes_left = sum(1 for r in results_batch if r['success_left'])
    successes_right = sum(1 for r in results_batch if r['success_right'])
    successes_both = sum(1 for r in results_batch if r['success_left'] and r['success_right'])
    beauty_print(f"Successful Left: {successes_left}/{len(results_batch)}")
    beauty_print(f"Successful Right: {successes_right}/{len(results_batch)}")
    beauty_print(f"Successful Both: {successes_both}/{len(results_batch)}")
    
    if args.show_details:
        for i, result in enumerate(results_batch[:args.max_display]):
            beauty_print(f"\nConfiguration {i+1}:")
            beauty_print(f"  Success Left: {result['success_left']}")
            beauty_print(f"  Success Right: {result['success_right']}")
            if result['success_left'] and result['success_right']:
                # Extract error information from res_left and res_right
                res_left = result['res_left']
                res_right = result['res_right']
                
                # Handle case where res_left/res_right might be dict or list
                if isinstance(res_left, list) and len(res_left) > 0:
                    res_left = res_left[0]
                if isinstance(res_right, list) and len(res_right) > 0:
                    res_right = res_right[0]
                
                # Get iters (use max of both arms if available)
                iters_left = res_left['iters'] if isinstance(res_left, dict) else 0
                iters_right = res_right['iters'] if isinstance(res_right, dict) else 0
                iters = max(iters_left, iters_right)
                
                pos_err_left = res_left['pos_err'] if isinstance(res_left, dict) else 0.0
                pos_err_right = res_right['pos_err'] if isinstance(res_right, dict) else 0.0
                ori_err_left = res_left['ori_err'] if isinstance(res_left, dict) else 0.0
                ori_err_right = res_right['ori_err'] if isinstance(res_right, dict) else 0.0
                
                print(f"  Left joint angles: {beauty_print_array(np.array(result['q_left']))}")
                print(f"  Right joint angles: {beauty_print_array(np.array(result['q_right']))}")
                print(f"  Position error left: {pos_err_left:.6e} m")
                print(f"  Position error right: {pos_err_right:.6e} m")
                print(f"  Orientation error left: {ori_err_left:.6e} rad")
                print(f"  Orientation error right: {ori_err_right:.6e} rad")
                print(f"  Iterations: {iters}")


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics Parallel Demo - Batch IK processing with random joint configurations")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
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
                        help='Backend to use for computation (default: torch)')
    parser.add_argument('--method', type=str, default='dls',
                        choices=['dls', 'pinv', 'transpose'],
                        help='IK method (default: dls)')
    parser.add_argument('--coordination', type=str, default='indep',
                        choices=['indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror'],
                        help='Coordination mode (default: indep)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_false',
                        help='Show detailed results for each configuration')
    parser.add_argument('--max-display', type=int, default=5,
                        help='Maximum number of configurations to display (default: 5)')
    args = parser.parse_args()

    main(args)

