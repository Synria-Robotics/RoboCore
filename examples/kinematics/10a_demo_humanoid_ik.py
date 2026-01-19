"""Humanoid Inverse Kinematics Demo

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
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_ik(robot_model, backend, targets, end_links, base_link='pelvis', 
               q0=None, num_initial_guesses=1, initial_guess_strategy='random', 
               initial_guess_scale=1.0, random_seed=None, max_iters=200, pos_tol=1e-3, ori_tol=1e-3):
    """Compute humanoid inverse kinematics using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy' or 'torch')
    :param targets: Dictionary mapping end_link to target pose (4x4 matrix or [px, py, pz, qx, qy, qz, qw])
    :param end_links: List of end-effector link names
    :param base_link: Base link name
    :param q0: Initial full configuration [nq] (optional)
    :param num_initial_guesses: Number of initial guesses to try
    :param initial_guess_strategy: Strategy for generating initial guesses
    :param initial_guess_scale: Scale factor for joint limits
    :param random_seed: Random seed for reproducibility
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    # Convert target poses to 4x4 matrices if needed
    targets_4x4 = {}
    for end_link in end_links:
        target = targets[end_link]
        if isinstance(target, (list, np.ndarray)) and len(target) == 7:
            T = np.zeros((4, 4))
            T[:3, 3] = target[:3]
            T[3, 3] = 1.0
            T[:3, :3] = quaternion_to_matrix(target[3:])
            targets_4x4[end_link] = T
        else:
            targets_4x4[end_link] = np.array(target)

    # Use unified multi-chain IK
    ik_result = robot_model.ik(
        targets=targets_4x4,
        end_links=end_links,
        q_initial=q0,
        method='dls',
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        num_initial_guesses=num_initial_guesses,
        initial_guess_strategy=initial_guess_strategy,
        initial_guess_scale=initial_guess_scale,
        random_seed=random_seed,
        base_link=base_link,
    )

    elapsed_time = time.time() - start_time

    # Extract results
    q_full = np.array(ik_result['q'])
    iters = ik_result.get('iters', 0)
    success = ik_result.get('success', False)
    pos_err = ik_result.get('pos_err', 0.0)
    ori_err = ik_result.get('ori_err', 0.0)

    return {
        'success': success,
        'iters': iters,
        'pos_err': pos_err,
        'ori_err': ori_err,
        'q_full': q_full,
        'time': elapsed_time
    }


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link)
    
    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]
    end_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']
    
    # Generate reachable target poses using FK from random configurations
    # This ensures targets are within workspace
    targets = {}
    if args.use_random_targets:
        beauty_print("Generating reachable target poses from random configurations", type="info")
        # Generate random full configuration
        q_random = robot_model.random_q_full(seed=args.seed, scale=0.8)
        
        # Compute FK for each end-effector to get reachable poses
        for end_link in end_links:
            T = robot_model.fk(q_random, base_link=args.base_link, end_link=end_link, return_end=True)
            T = to_numpy(T)
            # Convert to [px, py, pz, qx, qy, qz, qw] format
            pos = T[:3, 3]
            quat = matrix_to_quaternion(T[:3, :3])
            targets[end_link] = np.concatenate([pos, quat])
        
        beauty_print("Generated target poses:")
        for end_link, display_name in zip(end_links, display_names):
            pos = targets[end_link][:3]
            print(f"  {display_name}: position = {beauty_print_array(pos)}")
    else:
        # Parse target poses from arguments
        target_list = [args.target_left_thumb, args.target_right_thumb, args.target_left_toe, args.target_right_toe]
        for end_link, target in zip(end_links, target_list):
            targets[end_link] = target
    
    # Compute with both backends
    results_np = compute_ik(
        robot_model, 'numpy', targets, end_links, args.base_link,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
    )
    results_torch = compute_ik(
        robot_model, 'torch', targets, end_links, args.base_link,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
    )

    # Convert to numpy for comparison
    q_np = to_numpy(results_np['q_full'])
    q_torch = to_numpy(results_torch['q_full'])

    beauty_print(f"IK Solution:")
    print(f"  Success:  NumPy={results_np['success']}, Torch={results_torch['success']}")
    print(f"  Iterations:  NumPy={results_np['iters']}, Torch={results_torch['iters']}")
    print(f"  Position Error:  NumPy={results_np['pos_err']:.6e} m, Torch={results_torch['pos_err']:.6e} m")
    print(f"  Orientation Error:  NumPy={results_np['ori_err']:.6e} rad, Torch={results_torch['ori_err']:.6e} rad")

    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  NumPy:  {beauty_print_array(q_np)}")
    print(f"  Torch:  {beauty_print_array(q_torch)}")
    q_diff = np.linalg.norm(q_np - q_torch)
    print(f"  Diff:   {q_diff:.6e}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']* 1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']* 1000:.4f} ms")
    if results_np['time'] > 0:
        ratio = results_torch['time'] / results_np['time']
        print(f"  Ratio:  {ratio:.2f}x")


if __name__ == "__main__":
    from openrd import get_model_path

    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Humanoid Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--target-left-thumb', type=float, nargs='+',
                        default=[+0.26426, +0.1655, +0.6523, 0.0, 0.0, 0.0, 1.0],
                        help='Target left thumb pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right-thumb', type=float, nargs='+',
                        default=[+0.26426, -0.1654, +0.1523, 0.0, 0.0, 0.0, 1.0],
                        help='Target right thumb pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-left-toe', type=float, nargs='+',
                        default=[-0.00000, +0.11851, -0.75686, 0.0, 0.0, 0.0, 1.0],
                        help='Target left toe pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right-toe', type=float, nargs='+',
                        default=[-0.00000, -0.11851, -0.75686, 0.0, 0.0, 0.0, 1.0],
                        help='Target right toe pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--use-random-targets', action='store_true',
                        help='Generate reachable target poses from random configurations (ensures IK success)')
    parser.add_argument('--max-iters', type=int, default=200,
                        help='Maximum IK iterations (default: 200)')
    parser.add_argument('--pos-tol', type=float, default=1e-3,
                        help='Position tolerance in meters (default: 1e-3)')
    parser.add_argument('--ori-tol', type=float, default=1e-3,
                        help='Orientation tolerance in radians (default: 1e-3)')
    parser.add_argument('--num-inits', type=int, default=1,
                        help='Number of initial guesses to try per target (default: 1)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Strategy for generating initial guesses (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale factor for joint limits when generating guesses (0.0 to 1.0, default: 1.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy, ignored - both are tested)')
    args = parser.parse_args()

    main(args)
