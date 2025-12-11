"""Test IK: High max_iters vs Multiple initial guesses.

This demo compares the performance of:
1. Single guess with high max_iters
2. Multiple guesses with standard max_iters
"""

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import bimanual_forward_kinematics, bimanual_inverse_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model
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

    beauty_print(f"Testing with {num_configs} target poses", type="module", centered=True)

    # Test 1: Single guess with high max_iters
    beauty_print(f"Test 1: Single Guess + High Max Iters ({args.high_max_iters})", type="module")
    ik_kwargs_high_iters = {
        'method': args.method,
        'num_initial_guesses': 1,
        'initial_guess_strategy': args.initial_guess_strategy,
        'initial_guess_scale': args.initial_guess_scale,
        'random_seed': args.random_seed,
        'max_iters': args.high_max_iters,
        'pos_tol': args.pos_tol,
        'ori_tol': args.ori_tol,
    }
    
    start_time = time.time()
    results_high_iters = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_batch, target_right=target_right_batch,
        **ik_kwargs_high_iters
    )
    time_high_iters = time.time() - start_time
    
    success_high_iters = sum(1 for r in results_high_iters if r.get('success_left', False) and r.get('success_right', False)) if isinstance(results_high_iters, list) else 0
    print(f"  Time: {time_high_iters*1000:.2f} ms ({time_high_iters/num_configs*1000:.2f} ms/sample)")
    print(f"  Success Rate: {success_high_iters}/{num_configs} ({100*success_high_iters/num_configs:.1f}%)")
    
    # Calculate average iterations
    avg_iters_high = 0
    if isinstance(results_high_iters, list):
        total_iters = 0
        count = 0
        for r in results_high_iters:
            res_left = r.get('res_left', {})
            res_right = r.get('res_right', {})
            if isinstance(res_left, dict):
                total_iters += res_left.get('iters', 0)
                count += 1
            if isinstance(res_right, dict):
                total_iters += res_right.get('iters', 0)
                count += 1
        if count > 0:
            avg_iters_high = total_iters / count
    print(f"  Avg Iterations: {avg_iters_high:.1f}")

    # Test 2: Multiple guesses with standard max_iters
    beauty_print(f"Test 2: Multiple Guesses ({args.num_guesses}) + Standard Max Iters ({args.standard_max_iters})", type="module")
    ik_kwargs_multi = {
        'method': args.method,
        'num_initial_guesses': args.num_guesses,
        'initial_guess_strategy': args.initial_guess_strategy,
        'initial_guess_scale': args.initial_guess_scale,
        'random_seed': args.random_seed,
        'max_iters': args.standard_max_iters,
        'pos_tol': args.pos_tol,
        'ori_tol': args.ori_tol,
    }
    
    start_time = time.time()
    results_multi = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_batch, target_right=target_right_batch,
        **ik_kwargs_multi
    )
    time_multi = time.time() - start_time
    
    success_multi = sum(1 for r in results_multi if r.get('success_left', False) and r.get('success_right', False)) if isinstance(results_multi, list) else 0
    print(f"  Time: {time_multi*1000:.2f} ms ({time_multi/num_configs*1000:.2f} ms/sample)")
    print(f"  Success Rate: {success_multi}/{num_configs} ({100*success_multi/num_configs:.1f}%)")
    
    # Calculate average iterations (across all guesses)
    avg_iters_multi = 0
    if isinstance(results_multi, list):
        total_iters = 0
        count = 0
        for r in results_multi:
            res_left = r.get('res_left', {})
            res_right = r.get('res_right', {})
            if isinstance(res_left, dict):
                total_iters += res_left.get('iters', 0)
                count += 1
            if isinstance(res_right, dict):
                total_iters += res_right.get('iters', 0)
                count += 1
        if count > 0:
            avg_iters_multi = total_iters / count
    print(f"  Avg Iterations: {avg_iters_multi:.1f}")

    # Comparison
    beauty_print("Performance Comparison", type="module")
    print(f"  High Iters Time:    {time_high_iters*1000:.2f} ms")
    print(f"  Multi Guesses Time: {time_multi*1000:.2f} ms")
    if time_high_iters > 0:
        print(f"  Speed Ratio:        {time_high_iters / time_multi:.2f}x")
    print(f"  High Iters Success: {success_high_iters}/{num_configs} ({100*success_high_iters/num_configs:.1f}%)")
    print(f"  Multi Guesses Success: {success_multi}/{num_configs} ({100*success_multi/num_configs:.1f}%)")
    print(f"  Success Difference: {success_multi - success_high_iters} ({100*(success_multi-success_high_iters)/num_configs:.1f}%)")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Test IK: High max_iters vs Multiple guesses")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to robot model file')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--num-configs', type=int, default=20, help='Number of target poses to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scale', type=float, default=0.8, help='Scale factor for joint limits')
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'], help='IK method')
    parser.add_argument('--initial-guess-strategy', type=str, default='random', help='Initial guess strategy')
    parser.add_argument('--initial-guess-scale', type=float, default=1.0, help='Initial guess scale')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for IK')
    parser.add_argument('--standard-max-iters', type=int, default=200, help='Standard max iterations')
    parser.add_argument('--high-max-iters', type=int, default=1000, help='High max iterations for single guess test')
    parser.add_argument('--num-guesses', type=int, default=10, help='Number of initial guesses for multi-guess test')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'], help='Backend')
    args = parser.parse_args()
    main(args)

