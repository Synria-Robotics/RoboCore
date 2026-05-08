"""Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_ik(robot_model, backend, end_pose, q0=None, num_initial_guesses=1, initial_guess_strategy='random', initial_guess_scale=1.0, random_seed=None):
    """Compute inverse kinematics for given backend.

    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param end_pose: Target end-effector pose [px, py, pz, qx, qy, qz, qw]
    :param q0: Base initial joint angles guess (optional, used as base for strategies)
    :param num_initial_guesses: Number of initial guesses to try
    :param initial_guess_strategy: Strategy for generating initial guesses
    :param initial_guess_scale: Scale factor for joint limits
    :param random_seed: Random seed for reproducibility
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    T_fk = np.zeros((4, 4))
    T_fk[:3, 3] = end_pose[:3]
    T_fk[3, 3] = 1.0
    T_fk[:3, :3] = to_numpy(quaternion_to_matrix(end_pose[3:]))

    ik_result = inverse_kinematics(
        robot_model,
        T_fk,
        q0,
        method='dls',
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
        use_analytic_jacobian=True,
        num_initial_guesses=num_initial_guesses,
        initial_guess_strategy=initial_guess_strategy,
        initial_guess_scale=initial_guess_scale,
        random_seed=random_seed,
    )

    elapsed_time = time.time() - start_time

    return {
        'success': ik_result['success'],
        'iters': ik_result['iters'],
        'pos_err': ik_result['pos_err'],
        'ori_err': ik_result['ori_err'],
        'err_norm': ik_result.get('err_norm', None),
        'q': ik_result['q'],
        'time': elapsed_time
    }


def main(args):
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)


    # Compute with both backends
    results_np = compute_ik(
        robot_model, 'numpy', args.end_pose,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )
    results_torch = compute_ik(
        robot_model, 'torch', args.end_pose,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )
    results_cpp = compute_ik(
        robot_model, 'cpp', args.end_pose,
        q0=None,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed
    )

    # Convert to numpy for comparison
    q_np = to_numpy(results_np['q'])
    q_torch = to_numpy(results_torch['q'])
    q_cpp = to_numpy(results_cpp['q'])

    beauty_print(f"IK Solution:")
    print(f"  Success:  NumPy={results_np['success']}, Torch={results_torch['success']}, C++={results_cpp['success']}")
    print(f"  Iterations:  NumPy={results_np['iters']}, Torch={results_torch['iters']}, C++={results_cpp['iters']}")
    print(f"  Position Error:  NumPy={results_np['pos_err']:.6e} m, Torch={results_torch['pos_err']:.6e} m, C++={results_cpp['pos_err']:.6e} m")
    print(f"  Orientation Error:  NumPy={results_np['ori_err']:.6e} rad, Torch={results_torch['ori_err']:.6e} rad, C++={results_cpp['ori_err']:.6e} rad")
    if results_np['err_norm'] is not None:
        print(f"  Total Error:  NumPy={results_np['err_norm']:.6e}, Torch={results_torch['err_norm']:.6e}, C++={results_cpp['err_norm']:.6e}")

    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  NumPy:  {beauty_print_array(q_np)}")
    print(f"  Torch:  {beauty_print_array(q_torch)}")
    print(f"  C++:    {beauty_print_array(q_cpp)}")
    print(f"  np vs torch: {np.linalg.norm(q_np - q_torch):.6e}   np vs cpp: {np.linalg.norm(q_np - q_cpp):.6e}")

    beauty_print(f"Solved Joint Angles (degrees):")
    print(f"  NumPy:  {beauty_print_array(np.rad2deg(q_np))}")
    print(f"  Torch:  {beauty_print_array(np.rad2deg(q_torch))}")
    print(f"  C++:    {beauty_print_array(np.rad2deg(q_cpp))}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']* 1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']* 1000:.4f} ms")
    print(f"  C++:    {results_cpp['time']* 1000:.4f} ms")
    tnp = max(results_np['time'], 1e-15)
    print(f"  torch/np: {results_torch['time'] / tnp:.2f}x   cpp/np: {results_cpp['time'] / tnp:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Inverse Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to model file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+',
                        default=[0.16993, 0.01740, 0.20530, 0.041461, 0.828399, 0.083471, 0.552331],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--num-inits', type=int, default=1,
                        help='Number of initial guesses to try per target (default: 1)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Strategy for generating initial guesses (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale factor for joint limits when generating guesses (0.0 to 1.0, default: 1.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch', 'cpp'],
                        help='Legacy option (ignored — NumPy, Torch, and C++ are all tested)')
    args = parser.parse_args()
    main(args)

    """_results_
    [RoboCore:INFO] IK Solution:
      Success:  NumPy=True, Torch=True
      Iterations:  NumPy=5, Torch=9
      Position Error:  NumPy=4.704810e-07 m, Torch=7.463299e-05 m
      Orientation Error:  NumPy=2.591603e-06 rad, Torch=1.938068e-05 rad
      Total Error:  NumPy=3.062084e-06, Torch=7.710832e-05
    [RoboCore:INFO] Solved Joint Angles (radians):
      NumPy:  [+0.10000, +0.19996, -0.30000, +0.00000, +0.50004, -0.20000]
      Torch:  [+0.10067, +0.20002, -0.30002, -0.00132, +0.49999, -0.19911]
      Diff:   1.732748e-03
    [RoboCore:INFO] Solved Joint Angles (degrees):
      NumPy:  [+5.72942, +11.45659, -17.18848, +0.00012, +28.65011, -11.45917]
      Torch:  [+5.76814, +11.46026, -17.18997, -0.07550, +28.64727, -11.40804]
    [RoboCore:INFO] Computation Time:
      NumPy:  2.8539 ms
      Torch:  61.7158 ms
      Ratio:  21.63x
    """