"""Bimanual Inverse Kinematics Demo

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


def compute_ik(robot_model, backend, target_left, target_right, left_end_link, right_end_link,
               base_link='base_link', q0=None, num_initial_guesses=1,
               initial_guess_strategy='random', initial_guess_scale=1.0, random_seed=None):
    """Compute bimanual inverse kinematics using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy' or 'torch')
    :param target_left: Target left end-effector pose [px, py, pz, qx, qy, qz, qw] or 4x4 matrix
    :param target_right: Target right end-effector pose [px, py, pz, qx, qy, qz, qw] or 4x4 matrix
    :param left_end_link: Left arm end-effector link name
    :param right_end_link: Right arm end-effector link name
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
    if isinstance(target_left, (list, np.ndarray)) and len(target_left) == 7:
        T_left = np.zeros((4, 4))
        T_left[:3, 3] = target_left[:3]
        T_left[3, 3] = 1.0
        T_left[:3, :3] = quaternion_to_matrix(target_left[3:])
    else:
        T_left = np.array(target_left)
    
    if isinstance(target_right, (list, np.ndarray)) and len(target_right) == 7:
        T_right = np.zeros((4, 4))
        T_right[:3, 3] = target_right[:3]
        T_right[3, 3] = 1.0
        T_right[:3, :3] = quaternion_to_matrix(target_right[3:])
    else:
        T_right = np.array(target_right)

    # Use unified multi-chain IK
    ik_result = robot_model.ik(
        targets={
            left_end_link: T_left,
            right_end_link: T_right,
        },
        end_links=[left_end_link, right_end_link],
        q_initial=q0,
        method='dls',
        max_iters=200,
        pos_tol=1e-3,
        ori_tol=1e-3,
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
    
    # Extract left and right arm joint values from unified config
    left_indices = robot_model._get_joint_indices(base_link, left_end_link)
    right_indices = robot_model._get_joint_indices(base_link, right_end_link)
    
    q_left = q_full[left_indices] if len(left_indices) > 0 else np.array([])
    q_right = q_full[right_indices] if len(right_indices) > 0 else np.array([])

    return {
        'success': success,
        'success_left': success,  # For compatibility
        'success_right': success,  # For compatibility
        'iters': iters,
        'pos_err': pos_err,
        'pos_err_left': pos_err,  # For compatibility
        'pos_err_right': pos_err,  # For compatibility
        'ori_err': ori_err,
        'ori_err_left': ori_err,  # For compatibility
        'ori_err_right': ori_err,  # For compatibility
        'q_full': q_full,
        'q_left': q_left,
        'q_right': q_right,
        'time': elapsed_time
    }


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.left_base_link)
    
    # Compute with both backends
    results_np = compute_ik(
        robot_model, 'numpy', args.target_left, args.target_right,
        args.left_end_link, args.right_end_link, args.left_base_link,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )
    results_torch = compute_ik(
        robot_model, 'torch', args.target_left, args.target_right,
        args.left_end_link, args.right_end_link, args.left_base_link,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )

    # Convert to numpy for comparison
    q_left_np = to_numpy(results_np['q_left'])
    q_right_np = to_numpy(results_np['q_right'])
    q_left_torch = to_numpy(results_torch['q_left'])
    q_right_torch = to_numpy(results_torch['q_right'])

    beauty_print(f"IK Solution:")
    print(f"  Success Left:  NumPy={results_np['success_left']}, Torch={results_torch['success_left']}")
    print(f"  Success Right:  NumPy={results_np['success_right']}, Torch={results_torch['success_right']}")
    print(f"  Iterations:  NumPy={results_np['iters']}, Torch={results_torch['iters']}")
    print(f"  Position Error Left:  NumPy={results_np['pos_err_left']:.6e} m, Torch={results_torch['pos_err_left']:.6e} m")
    print(f"  Position Error Right:  NumPy={results_np['pos_err_right']:.6e} m, Torch={results_torch['pos_err_right']:.6e} m")
    print(f"  Orientation Error Left:  NumPy={results_np['ori_err_left']:.6e} rad, Torch={results_torch['ori_err_left']:.6e} rad")
    print(f"  Orientation Error Right:  NumPy={results_np['ori_err_right']:.6e} rad, Torch={results_torch['ori_err_right']:.6e} rad")

    beauty_print(f"Solved Joint Angles - Left Arm (radians):")
    print(f"  NumPy:  {beauty_print_array(q_left_np)}")
    print(f"  Torch:  {beauty_print_array(q_left_torch)}")

    beauty_print(f"Solved Joint Angles - Right Arm (radians):")
    print(f"  NumPy:  {beauty_print_array(q_right_np)}")
    print(f"  Torch:  {beauty_print_array(q_right_torch)}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']* 1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']* 1000:.4f} ms")
    if results_np['time'] > 0:
        ratio = results_torch['time'] / results_np['time']
        print(f"  Ratio:  {ratio:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--target-left', type=float, nargs='+',
                        default=[0.05717, -0.35161, 0.45995, -0.504640, 0.483414, 0.385570, 0.602483],
                        help='Target left end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right', type=float, nargs='+',
                        default=[0.05715, 0.12706, 0.46730, 0.385071, 0.387550, -0.485394, 0.682582],
                        help='Target right end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    # Note: coordination parameter removed - unified interface uses independent multi-chain IK
    # Relative and mirror modes can be implemented by constraining targets if needed
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
