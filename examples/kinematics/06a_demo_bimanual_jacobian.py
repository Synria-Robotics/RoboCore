"""Bimanual Jacobian validation and comparison.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations
import argparse
import time
import numpy as np

import robocore as rc
from robocore.modeling import RobotModel
# Using unified interface - no need for bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def compute_jacobian_results(robot_model, backend, q_full, left_end_link, right_end_link, 
                             base_link='base_link', device=None):
    """Compute bimanual Jacobian using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param q_full: Full joint configuration [nq]
    :param left_end_link: Left arm end-effector link name
    :param right_end_link: Right arm end-effector link name
    :param base_link: Base link name
    :param device: Device for torch backend
    :return: Dictionary with Jacobian matrix and computation time
    """
    if backend == 'torch':
        import torch
        if device is None:
            device = torch.device('cpu')
        rc.set_backend('torch', device=str(device))
    else:
        rc.set_backend(backend)

    start_time = time.perf_counter()
    
    # Compute Jacobians for both arms
    J_left = robot_model.jacobian(q_full, base_link=base_link, end_link=left_end_link)
    J_right = robot_model.jacobian(q_full, base_link=base_link, end_link=right_end_link)
    
    # Stack into combined Jacobian [12, nq_full] (6 rows per arm)
    J = np.vstack([J_left, J_right])
    
    elapsed_time = (time.perf_counter() - start_time) * 1000

    return {
        'J': J,
        'time': elapsed_time
    }


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(args.model_path, base_link=args.left_base_link)
    
    beauty_print(f"Bimanual Jacobian Validation: {robot_model.name} (Total DOF: {robot_model.num_dof})", type="module")

    # Build unified configuration vector
    q_full = np.zeros(robot_model.num_dof)
    left_indices = robot_model._get_joint_indices(args.left_base_link, args.left_end_link)
    right_indices = robot_model._get_joint_indices(args.right_base_link, args.right_end_link)
    
    q_left = np.array(args.q_left)
    q_right = np.array(args.q_right)
    
    if len(q_left) <= len(left_indices):
        q_full[left_indices[:len(q_left)]] = q_left
    if len(q_right) <= len(right_indices):
        q_full[right_indices[:len(q_right)]] = q_right

    beauty_print(f"Left arm joint configuration (rad):")
    print(f"  q_left = {beauty_print_array(q_left)}")
    beauty_print(f"Right arm joint configuration (rad):")
    print(f"  q_right = {beauty_print_array(q_right)}")
    beauty_print(f"Unified config space size: {robot_model.num_dof}")

    # Compute with both backends
    results_np = compute_jacobian_results(robot_model, 'numpy', q_full, 
                                          args.left_end_link, args.right_end_link, args.left_base_link)
    import torch
    device = torch.device(args.device)
    q_full_torch = torch.tensor(q_full, dtype=torch.float64, device=device)
    results_torch = compute_jacobian_results(robot_model, 'torch', q_full_torch,
                                            args.left_end_link, args.right_end_link, args.left_base_link, device=device)
    results_cpp = compute_jacobian_results(robot_model, 'cpp', q_full,
                                           args.left_end_link, args.right_end_link, args.left_base_link)

    # Convert to numpy for comparison
    J_np = to_numpy(results_np['J'])
    J_torch = to_numpy(results_torch['J'])
    J_cpp = to_numpy(results_cpp['J'])

    beauty_print("[1] Jacobian Comparison (NumPy vs Torch vs C++)", type="module", centered=False)
    beauty_print(f"Jacobian shape: {J_np.shape}")
    beauty_print(f"Condition number (NumPy): {np.linalg.cond(J_np):.2e}")
    beauty_print(f"Condition number (Torch): {np.linalg.cond(J_torch):.2e}")
    beauty_print(f"Condition number (C++):   {np.linalg.cond(J_cpp):.2e}")

    beauty_print(f"Jacobian Matrix (NumPy, first 6 rows):")
    print(beauty_print_array(J_np[:6, :], precision=6))
    if J_np.shape[0] > 6:
        beauty_print(f"Jacobian Matrix (NumPy, last 6 rows):")
        print(beauty_print_array(J_np[6:, :], precision=6))
    
    beauty_print(f"Jacobian Matrix (Torch, first 6 rows):")
    print(beauty_print_array(J_torch[:6, :], precision=6))
    if J_torch.shape[0] > 6:
        beauty_print(f"Jacobian Matrix (Torch, last 6 rows):")
        print(beauty_print_array(J_torch[6:, :], precision=6))

    beauty_print(f"Jacobian Matrix (C++, first 6 rows):")
    print(beauty_print_array(J_cpp[:6, :], precision=6))
    if J_cpp.shape[0] > 6:
        beauty_print(f"Jacobian Matrix (C++, last 6 rows):")
        print(beauty_print_array(J_cpp[6:, :], precision=6))

    diff_nt = J_np - J_torch
    diff_nc = J_np - J_cpp
    beauty_print("NumPy vs Torch:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_nt)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_nt, 'fro'):.3e}")
    beauty_print("NumPy vs C++:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_nc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_nc, 'fro'):.3e}")

    beauty_print("[2] Performance Comparison", type="module", centered=False)
    beauty_print(f"NumPy:  {results_np['time']:.4f} ms")
    beauty_print(f"Torch:  {results_torch['time']:.4f} ms")
    beauty_print(f"C++:    {results_cpp['time']:.4f} ms")
    if results_np['time'] > 0:
        beauty_print(f"torch/np: {results_torch['time'] / results_np['time']:.2f}x   cpp/np: {results_cpp['time'] / results_np['time']:.2f}x")

    beauty_print("✓ Bimanual Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    # Bessica is a dual-arm robot
    model_path = synriard.get_model_path("Bessica_D", version="v1_1", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--q-left', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.1],
                        help='Left joint angles in radians')
    parser.add_argument('--q-right', type=float, nargs='+', default=[-0.1, -0.2, 0.3, 0.0, -0.5, 0.2, 0.1],
                        help='Right joint angles in radians')
    # Note: mode parameter removed - unified interface computes independent Jacobians for each arm
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
    
    """_result_
    [RoboCore:MODULE] [1] Jacobian Comparison (NumPy vs Torch)
    [RoboCore:INFO] Jacobian shape: (12, 14)
    [RoboCore:INFO] Condition number (NumPy): 1.33e+04
    [RoboCore:INFO] Condition number (Torch): 1.33e+04
    [RoboCore:INFO] Jacobian Matrix (NumPy, first 6 rows):
    [
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.555047  -0.009408  +0.018265  -0.083549  +0.014537  -0.010736  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.000000  -0.557982  -0.001839  +0.282590  -0.002764  -0.054536  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.057174  +0.093768  +0.002207  -0.052441  +0.002022  -0.001395  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.995004  +0.097843  +0.956425  +0.097843  -0.971230  -0.193349]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -1.000000  +0.000000  -0.198669  +0.289629  -0.198669  +0.194709  -0.980853]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.000000  -0.099833  -0.975170  +0.036957  -0.975170  -0.137116  -0.023300]
    ]
    [RoboCore:INFO] Jacobian Matrix (NumPy, last 6 rows):
    [
      [-0.547700  +0.013010  -0.018160  +0.094867  -0.014595  +0.008619  -0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.000000  +0.550669  -0.001752  +0.277383  -0.003052  -0.050108  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.057152  -0.129664  -0.001465  -0.060358  -0.000843  +0.022501  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [+0.000000  +0.995004  +0.097843  +0.944703  +0.097843  -0.979111  +0.155247  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [+1.000000  +0.000000  -0.198669  -0.289629  -0.198669  -0.194709  -0.901914  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.000000  +0.099833  -0.975170  +0.153792  -0.975170  -0.058571  +0.403050  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
    ]
    [RoboCore:INFO] Jacobian Matrix (Torch, first 6 rows):
    [
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.555047  -0.009408  +0.018265  -0.083549  +0.014537  -0.010736  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.000000  -0.557982  -0.001839  +0.282590  -0.002764  -0.054536  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.057174  +0.093768  +0.002207  -0.052441  +0.002022  -0.001395  +0.000000]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.995004  +0.097843  +0.956425  +0.097843  -0.971230  -0.193349]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -1.000000  +0.000000  -0.198669  +0.289629  -0.198669  +0.194709  -0.980853]
      [+0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  -0.000000  -0.099833  -0.975170  +0.036957  -0.975170  -0.137116  -0.023300]
    ]
    [RoboCore:INFO] Jacobian Matrix (Torch, last 6 rows):
    [
      [-0.547700  +0.013010  -0.018160  +0.094867  -0.014595  +0.008619  -0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.000000  +0.550669  -0.001752  +0.277383  -0.003052  -0.050108  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.057152  -0.129664  -0.001465  -0.060358  -0.000843  +0.022501  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [+0.000000  +0.995004  +0.097843  +0.944703  +0.097843  -0.979111  +0.155247  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [+1.000000  +0.000000  -0.198669  -0.289629  -0.198669  -0.194709  -0.901914  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
      [-0.000000  +0.099833  -0.975170  +0.153792  -0.975170  -0.058571  +0.403050  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000  +0.000000]
    ]
    [RoboCore:INFO] NumPy vs Torch:
    [RoboCore:INFO]   Max difference:        2.776e-17
    [RoboCore:INFO]   Frobenius norm:        2.973e-17
    [RoboCore:MODULE] [2] Performance Comparison
    [RoboCore:INFO] NumPy:  0.6100 ms
    [RoboCore:INFO] Torch:  35.5364 ms
    [RoboCore:INFO] Ratio:  58.25x    
    """

