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
import sys
from pathlib import Path

# Prefer the in-repo RoboCore package when running this demo directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.bimanual import bimanual_inverse_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def pose7_to_matrix(target_pose):
    """Convert [px, py, pz, qx, qy, qz, qw] pose to 4x4 matrix."""
    target_pose = np.asarray(target_pose, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = target_pose[:3]
    T[:3, :3] = np.asarray(quaternion_to_matrix(target_pose[3:]), dtype=np.float64)
    return T


def compute_ik(left_model, right_model, backend, target_left, target_right, q0_left=None, q0_right=None,
               coordination='indep', num_initial_guesses=1, initial_guess_strategy='random',
               initial_guess_scale=1.0, random_seed=None):
    """Compute bimanual inverse kinematics for given backend.
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param backend: Backend name ('numpy' or 'torch')
    :param target_left: Target left end-effector pose [px, py, pz, qx, qy, qz, qw] or 4x4 matrix
    :param target_right: Target right end-effector pose [px, py, pz, qx, qy, qz, qw] or 4x4 matrix
    :param coordination: Coordination mode ('indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror')
    :param num_initial_guesses: Number of initial guesses to try
    :param initial_guess_strategy: Strategy for generating initial guesses
    :param initial_guess_scale: Scale factor for joint limits
    :param random_seed: Random seed for reproducibility
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    # Convert target poses to 4x4 matrices if needed.
    if isinstance(target_left, (list, np.ndarray)) and len(target_left) == 7:
        T_left = pose7_to_matrix(target_left)
    else:
        T_left = np.array(target_left, dtype=np.float64)

    if isinstance(target_right, (list, np.ndarray)) and len(target_right) == 7:
        T_right = pose7_to_matrix(target_right)
    else:
        T_right = np.array(target_right, dtype=np.float64)

    ik_result = bimanual_inverse_kinematics(
        left_model,
        right_model,
        target_left=T_left,
        target_right=T_right,
        q0_left=q0_left,
        q0_right=q0_right,
        method='dls',
        coordination=coordination,
        use_analytic_jacobian=True,
        num_initial_guesses=num_initial_guesses,
        initial_guess_strategy=initial_guess_strategy,
        initial_guess_scale=initial_guess_scale,
        random_seed=random_seed,
    )

    elapsed_time = time.time() - start_time

    # Extract error information from res_left and res_right
    res_left = ik_result['res_left']
    res_right = ik_result['res_right']
    
    # Handle case where res_left/res_right might be dict or list
    if isinstance(res_left, list) and len(res_left) > 0:
        res_left = res_left[0]
    if isinstance(res_right, list) and len(res_right) > 0:
        res_right = res_right[0]
    
    # Get iters (use max of both arms if available)
    iters_left = res_left['iters']
    iters_right = res_right['iters']
    iters = max(iters_left, iters_right)

    return {
        'success_left': ik_result['success_left'],
        'success_right': ik_result['success_right'],
        'iters': iters,
        'pos_err_left': res_left['pos_err'],
        'pos_err_right': res_right['pos_err'],
        'ori_err_left': res_left['ori_err'],
        'ori_err_right': res_right['ori_err'],
        'q_left': ik_result['q_left'],
        'q_right': ik_result['q_right'],
        'time': elapsed_time
    }


def main(args):
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)
    target_left = np.asarray(args.target_left, dtype=np.float64)
    target_right = np.asarray(args.target_right, dtype=np.float64)

    # Compute with both backends
    results_np = compute_ik(
        left_model, right_model, 'numpy', target_left, target_right,
        coordination=args.coordination,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )
    results_torch = compute_ik(
        left_model, right_model, 'torch', target_left, target_right,
        coordination=args.coordination,
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

    # model_path = get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")
    model_path = get_model_path("Alicia_M", version="v1_1", variant="bimanual_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='world_base', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='tool0_l', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='world_base', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='tool0_r', help='Right arm end-effector link name')
    parser.add_argument('--target-left', type=float, nargs='+',
                        default=[-0.19449737591558663, -0.20039701467905618, 0.12041091987533925,
                                 0.9811121564712746, -0.022305916485078278, -0.1621691940912966, 0.10306568294939504],
                        help='Target left end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right', type=float, nargs='+',
                        default=[0.09020439652403661, -0.07509374443188993, 0.08822214081723735,
                                 -0.08784407246632346, 0.9749422854306018, -0.02894067453598843, -0.20232003452275402],
                        help='Target right end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--coordination', type=str, default='indep',
                        choices=['indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror'],
                        help='Coordination mode: indep (independent), relative_pose, relative_pos, relative_ori, mirror')
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
