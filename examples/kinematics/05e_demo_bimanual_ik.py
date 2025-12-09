"""Bimanual Inverse Kinematics Demo

This demo demonstrates bimanual inverse kinematics computation for dual-arm systems.
It shows how to solve IK for both arms independently, with relative constraints, and mirror mode.

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


def compute_bimanual_ik(left_model, right_model, backend, target_left, target_right, 
                        q0_left=None, q0_right=None, coordination='indep', **ik_kwargs):
    """Compute bimanual inverse kinematics for given backend.
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param backend: Backend name ('numpy' or 'torch')
    :param target_left: Left target pose (4x4 matrix)
    :param target_right: Right target pose (4x4 matrix)
    :param q0_left: Initial left configuration (optional)
    :param q0_right: Initial right configuration (optional)
    :param coordination: Coordination mode ('indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror')
    :param ik_kwargs: Additional IK parameters
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    result = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left, target_right=target_right,
        q0_left=q0_left, q0_right=q0_right,
        coordination=coordination,
        **ik_kwargs
    )

    elapsed_time = time.time() - start_time

    # Verify solution
    q_left = result.get('q_left', q0_left if q0_left is not None else [0.0] * left_model.num_chain_dof)
    q_right = result.get('q_right', q0_right if q0_right is not None else [0.0] * right_model.num_chain_dof)

    # Compute FK to verify
    verify_result = bimanual_forward_kinematics(left_model, right_model, q_left, q_right, return_end=True)
    T_left_verify = to_numpy(verify_result['left'])
    T_right_verify = to_numpy(verify_result['right'])

    pos_left_verify = T_left_verify[:3, 3]
    pos_right_verify = T_right_verify[:3, 3]
    pos_left_target = target_left[:3, 3] if target_left is not None else None
    pos_right_target = target_right[:3, 3] if target_right is not None else None

    verify_pos_err_left = np.linalg.norm(pos_left_verify - pos_left_target) if pos_left_target is not None else 0.0
    verify_pos_err_right = np.linalg.norm(pos_right_verify - pos_right_target) if pos_right_target is not None else 0.0

    return {
        'q_left': q_left,
        'q_right': q_right,
        'success_left': result.get('success_left', False),
        'success_right': result.get('success_right', False),
        'verify_pos_left': pos_left_verify,
        'verify_pos_right': pos_right_verify,
        'verify_pos_err_left': verify_pos_err_left,
        'verify_pos_err_right': verify_pos_err_right,
        'res_left': result.get('res_left', {}),
        'res_right': result.get('res_right', {}),
        'time': elapsed_time
    }


def main(args):
    # Load robot model (Bessica is a dual-arm robot, use same model with different base/end links)
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)

    if args.verbose:
        beauty_print("Left Arm Model:", type="module")
        left_model.summary(show_chain=True)
        beauty_print("Right Arm Model:", type="module")
        right_model.summary(show_chain=True)

    # Prepare target poses
    if args.target_left is not None:
        target_left = pose_7d_to_matrix(args.target_left)
    else:
        # Use default target from zero-config FK to ensure it's in workspace
        q_zero_left = np.zeros(left_model.num_chain_dof)
        fk_result_left = left_model.fk(q_zero_left)
        target_left = fk_result_left['end']
        # Slightly modify position to create a reachable target
        target_left[0, 3] -= 0.1  # Move left arm slightly left
        target_left[2, 3] += 0.05  # Move slightly up

    if args.target_right is not None:
        target_right = pose_7d_to_matrix(args.target_right)
    else:
        # Use default target from zero-config FK to ensure it's in workspace
        q_zero_right = np.zeros(right_model.num_chain_dof)
        fk_result_right = right_model.fk(q_zero_right)
        target_right = fk_result_right['end']
        # Slightly modify position to create a reachable target
        target_right[0, 3] += 0.1  # Move right arm slightly right
        target_right[2, 3] += 0.05  # Move slightly up

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

    # Compute with both backends
    results_np = compute_bimanual_ik(left_model, right_model, 'numpy', 
                                    target_left, target_right,
                                    coordination=args.coordination,
                                    **ik_kwargs)
    results_torch = compute_bimanual_ik(left_model, right_model, 'torch', 
                                       target_left, target_right,
                                       coordination=args.coordination,
                                       **ik_kwargs)

    # Display results
    beauty_print(f"IK Solution Status:")
    print(f"  Left Arm - NumPy:  {'Success' if results_np['success_left'] else 'Failed'}")
    print(f"  Left Arm - Torch:  {'Success' if results_torch['success_left'] else 'Failed'}")
    print(f"  Right Arm - NumPy:  {'Success' if results_np['success_right'] else 'Failed'}")
    print(f"  Right Arm - Torch:  {'Success' if results_torch['success_right'] else 'Failed'}")

    beauty_print(f"Left Arm Joint Angles (radians):")
    q_left_np = results_np['q_left']
    q_left_torch = results_torch['q_left']
    print(f"  NumPy:  {beauty_print_array(q_left_np)}")
    print(f"  Torch:  {beauty_print_array(q_left_torch)}")
    q_diff = np.linalg.norm(np.array(q_left_np) - np.array(q_left_torch))
    print(f"  Diff:   {q_diff:.6e}")

    beauty_print(f"Right Arm Joint Angles (radians):")
    q_right_np = results_np['q_right']
    q_right_torch = results_torch['q_right']
    print(f"  NumPy:  {beauty_print_array(q_right_np)}")
    print(f"  Torch:  {beauty_print_array(q_right_torch)}")
    q_diff = np.linalg.norm(np.array(q_right_np) - np.array(q_right_torch))
    print(f"  Diff:   {q_diff:.6e}")

    beauty_print(f"Verification (FK of IK solution):")
    print(f"  Left Position Error:  NumPy={results_np['verify_pos_err_left']:.6e} m, Torch={results_torch['verify_pos_err_left']:.6e} m")
    print(f"  Right Position Error:  NumPy={results_np['verify_pos_err_right']:.6e} m, Torch={results_torch['verify_pos_err_right']:.6e} m")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']*1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']*1000:.4f} ms")
    print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--target-left', type=float, nargs=7, default=None,
                        help='Left target pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right', type=float, nargs=7, default=None,
                        help='Right target pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--coordination', type=str, default='indep',
                        choices=['indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror'],
                        help='Coordination mode')
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'],
                        help='IK solving method')
    parser.add_argument('--num-initial-guesses', type=int, default=1,
                        help='Number of initial guesses to try')
    parser.add_argument('--initial-guess-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Initial guess strategy')
    parser.add_argument('--initial-guess-scale', type=float, default=1.0,
                        help='Scale factor for joint limits (0.0 to 1.0)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-iters', type=int, default=200, help='Maximum IK iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance (meters)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (radians)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed model information')
    args = parser.parse_args()
    main(args)

