"""Test IK timing with different number of initial guesses.

This demo compares the performance of IK solving with single vs multiple initial guesses.
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

    # Prepare IK kwargs
    ik_kwargs = {
        'method': args.method,
        'initial_guess_strategy': args.initial_guess_strategy,
        'initial_guess_scale': args.initial_guess_scale,
        'random_seed': args.random_seed,
        'max_iters': args.max_iters,
        'pos_tol': args.pos_tol,
        'ori_tol': args.ori_tol,
    }

    beauty_print(f"Testing with {num_configs} target poses", type="module", centered=True)

    # Test with single initial guess
    beauty_print("Single Initial Guess (num_initial_guesses=1)", type="module")
    start_time = time.time()
    results_single = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_batch, target_right=target_right_batch,
        num_initial_guesses=1,
        **ik_kwargs
    )
    time_single = time.time() - start_time
    
    success_single = sum(1 for r in results_single if r.get('success_left', False) and r.get('success_right', False)) if isinstance(results_single, list) else 0
    print(f"  Time: {time_single*1000:.2f} ms ({time_single/num_configs*1000:.2f} ms/sample)")
    print(f"  Success Rate: {success_single}/{num_configs} ({100*success_single/num_configs:.1f}%)")

    # Test with multiple initial guesses
    beauty_print(f"Multiple Initial Guesses (num_initial_guesses={args.num_initial_guesses})", type="module")
    start_time = time.time()
    results_multi = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left_batch, target_right=target_right_batch,
        num_initial_guesses=args.num_initial_guesses,
        **ik_kwargs
    )
    time_multi = time.time() - start_time
    
    success_multi = sum(1 for r in results_multi if r.get('success_left', False) and r.get('success_right', False)) if isinstance(results_multi, list) else 0
    print(f"  Time: {time_multi*1000:.2f} ms ({time_multi/num_configs*1000:.2f} ms/sample)")
    print(f"  Success Rate: {success_multi}/{num_configs} ({100*success_multi/num_configs:.1f}%)")

    # Comparison
    beauty_print("Performance Comparison", type="module")
    print(f"  Single guess time:   {time_single*1000:.2f} ms")
    print(f"  Multi guess time:    {time_multi*1000:.2f} ms")
    if time_single > 0:
        print(f"  Slowdown factor:    {time_multi / time_single:.2f}x")
        print(f"  Expected slowdown: {args.num_initial_guesses:.2f}x (if fully serial)")
    print(f"  Success improvement: {success_multi - success_single} ({100*(success_multi-success_single)/num_configs:.1f}%)")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Test IK timing with different initial guesses")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--num-configs', type=int, default=20, help='Number of target poses to process')
    parser.add_argument('--num-initial-guesses', type=int, default=10, help='Number of initial guesses for multi-guess test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scale', type=float, default=0.8, help='Scale factor for joint limits')
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'], help='IK method')
    parser.add_argument('--initial-guess-strategy', type=str, default='random', help='Initial guess strategy')
    parser.add_argument('--initial-guess-scale', type=float, default=1.0, help='Initial guess scale')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for IK')
    parser.add_argument('--max-iters', type=int, default=200, help='Maximum IK iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'], help='Backend')
    args = parser.parse_args()
    main(args)

