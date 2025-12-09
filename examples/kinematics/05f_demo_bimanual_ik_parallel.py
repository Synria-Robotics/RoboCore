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
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import bimanual_inverse_kinematics, bimanual_forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model (Bessica is a dual-arm robot, use same model with different base/end links)
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)

    # Generate random joint configurations and compute target poses
    num_configs = args.num_configs
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    q_left_batch = left_model.random_q_batch(num_configs, seed=args.seed, scale=args.scale)
    q_right_batch = right_model.random_q_batch(num_configs, seed=args.seed+1, scale=args.scale)

    # Compute target poses using FK
    beauty_print("Computing Target Poses via FK", type="module", centered=True)
    fk_results = bimanual_forward_kinematics(
        left_model, right_model, q_left_batch, q_right_batch,
        return_end=True, mode='indep'
    )
    target_left_batch = to_numpy(fk_results['left'])
    target_right_batch = to_numpy(fk_results['right'])

    beauty_print(f"Processing {num_configs} target pose(s) using {backend} backend", type="module", centered=True)

    # Prepare IK kwargs
    ik_kwargs = {
        'method': args.method,
        'num_initial_guesses': args.num_initial_guesses,
        'initial_guess_strategy': args.initial_guess_strategy,
        'initial_guess_scale': args.initial_guess_scale,
        'random_seed': args.random_seed,
        'max_iters': args.max_iters,
        'pos_tol': args.pos_tol,
        'ori_tol': args.ori_tol,
    }

    # Batch processing
    # Note: We don't pass q0_left/q0_right to test IK solver's ability to find solutions
    # from scratch, even though targets are guaranteed to be in workspace (from FK)
    start_time = time.time()
    results_batch = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_batch, target_right=target_right_batch,
        coordination=args.coordination,
        **ik_kwargs
    )
    batch_time = time.time() - start_time

    # Serial processing (for comparison)
    start_time = time.time()
    results_serial = []
    for i in range(num_configs):
        result = bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=target_left_batch[i], target_right=target_right_batch[i],
            coordination=args.coordination,
            **ik_kwargs
        )
        results_serial.append(result)
    serial_time = time.time() - start_time

    # Analyze results
    if isinstance(results_batch, dict):
        # Single result (shouldn't happen in batch mode, but handle it)
        success_left = results_batch.get('success_left', False)
        success_right = results_batch.get('success_right', False)
        success_count = 1 if (success_left and success_right) else 0
    else:
        # List of results
        success_count = sum(1 for r in results_batch if r.get('success_left', False) and r.get('success_right', False))

    serial_success_count = sum(1 for r in results_serial if r.get('success_left', False) and r.get('success_right', False))

    # Display results
    beauty_print(f"Batch Processing Results:", type="module")
    print(f"  Success Rate: {success_count}/{num_configs} ({100*success_count/num_configs:.1f}%)")
    print(f"  Serial Success Rate: {serial_success_count}/{num_configs} ({100*serial_success_count/num_configs:.1f}%)")

    # Analyze failures
    if isinstance(results_batch, list):
        failed_samples = []
        for i, result in enumerate(results_batch):
            if not (result.get('success_left', False) and result.get('success_right', False)):
                failed_samples.append(i)
                if len(failed_samples) <= 3:  # Show first 3 failures
                    res_left = result.get('res_left', {})
                    res_right = result.get('res_right', {})
                    pos_err_left = res_left.get('pos_err', float('inf')) if isinstance(res_left, dict) else float('inf')
                    pos_err_right = res_right.get('pos_err', float('inf')) if isinstance(res_right, dict) else float('inf')
                    ori_err_left = res_left.get('ori_err', float('inf')) if isinstance(res_left, dict) else float('inf')
                    ori_err_right = res_right.get('ori_err', float('inf')) if isinstance(res_right, dict) else float('inf')
                    print(f"  Failed Sample {i+1}:")
                    print(f"    Left:  pos_err={pos_err_left:.6e} m, ori_err={ori_err_left:.6e} rad, success={result.get('success_left', False)}")
                    print(f"    Right: pos_err={pos_err_right:.6e} m, ori_err={ori_err_right:.6e} rad, success={result.get('success_right', False)}")

        if len(failed_samples) > 3:
            print(f"  ... and {len(failed_samples) - 3} more failed samples")

    beauty_print(f"Performance Comparison:", type="module")
    print(f"  Serial Time:   {serial_time*1000:.4f} ms ({serial_time/num_configs*1000:.4f} ms/sample)")
    print(f"  Batch Time:    {batch_time*1000:.4f} ms ({batch_time/num_configs*1000:.4f} ms/sample)")
    if batch_time > 0:
        print(f"  Speedup:       {serial_time / batch_time:.2f}x")

    # Show first few successful results
    if isinstance(results_batch, list):
        beauty_print(f"First 3 Successful Solutions:", type="module")
        count = 0
        for i, result in enumerate(results_batch):
            if result.get('success_left', False) and result.get('success_right', False) and count < 3:
                q_left = result.get('q_left', [])
                q_right = result.get('q_right', [])
                print(f"  Sample {i+1}:")
                print(f"    Left:  {beauty_print_array(np.array(q_left)[:3])} ...")
                print(f"    Right: {beauty_print_array(np.array(q_right)[:3])} ...")
                count += 1


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics Parallel Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--num-configs', type=int, default=50, help='Number of target poses to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generating configurations')
    parser.add_argument('--scale', type=float, default=0.8, help='Scale factor for joint limits (0.0 to 1.0, default: 0.8 for better reachability)')
    parser.add_argument('--coordination', type=str, default='indep',
                        choices=['indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror'],
                        help='Coordination mode')
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'],
                        help='IK solving method')
    parser.add_argument('--num-initial-guesses', type=int, default=10,
                        help='Number of initial guesses to try (default: 10 for better success rate when solving from scratch)')
    parser.add_argument('--initial-guess-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Initial guess strategy')
    parser.add_argument('--initial-guess-scale', type=float, default=1.0,
                        help='Scale factor for joint limits (0.0 to 1.0)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-iters', type=int, default=200, help='Maximum IK iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance (meters)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (radians)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy)')
    args = parser.parse_args()
    main(args)

