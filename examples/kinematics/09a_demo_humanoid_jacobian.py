"""Humanoid Jacobian validation and comparison.

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

from __future__ import annotations
import argparse
import time
import numpy as np

import robocore as rc
from robocore.modeling import RobotModel
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def compute_jacobian_results(robot_model, backend, q_full, end_links, 
                             base_link='pelvis', device=None):
    """Compute humanoid Jacobian using unified configuration space.
    
    :param robot_model: RobotModel with unified config space
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param q_full: Full joint configuration [nq]
    :param end_links: List of end-effector link names
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
    
    # Compute Jacobians for all four end-effectors
    J_list = []
    for end_link in end_links:
        J = robot_model.jacobian(q_full, base_link=base_link, end_link=end_link)
        J_list.append(J)
    
    # Stack into combined Jacobian [24, nq_full] (6 rows per end-effector)
    J = np.vstack([to_numpy(j) for j in J_list])
    
    elapsed_time = (time.perf_counter() - start_time) * 1000

    return {
        'J': J,
        'time': elapsed_time
    }


def main(args):
    # Load robot model with unified configuration space
    robot_model = RobotModel(args.model_path, base_link=args.base_link)
    
    beauty_print(f"Humanoid Jacobian Validation: {robot_model.name} (Total DOF: {robot_model.num_dof})", type="module")

    # Build unified configuration vector
    q_full = np.zeros(robot_model.num_dof)
    if args.joint_angles and len(args.joint_angles) <= robot_model.num_dof:
        q_full[:len(args.joint_angles)] = args.joint_angles

    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]
    end_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']

    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q_full[:min(10, len(q_full))])}...")
    beauty_print(f"Unified config space size: {robot_model.num_dof}")

    # Compute with both backends
    results_np = compute_jacobian_results(robot_model, 'numpy', q_full, 
                                          end_links, args.base_link)
    import torch
    device = torch.device(args.device)
    q_full_torch = torch.tensor(q_full, dtype=torch.float64, device=device)
    results_torch = compute_jacobian_results(robot_model, 'torch', q_full_torch,
                                            end_links, args.base_link, device=device)
    results_cpp = compute_jacobian_results(robot_model, 'cpp', q_full,
                                           end_links, args.base_link)

    # Convert to numpy for comparison
    J_np = to_numpy(results_np['J'])
    J_torch = to_numpy(results_torch['J'])
    J_cpp = to_numpy(results_cpp['J'])

    beauty_print("[1] Jacobian Comparison (NumPy vs Torch vs C++)", type="module", centered=False)
    beauty_print(f"Jacobian shape: {J_np.shape}")
    beauty_print(f"Condition number (NumPy): {np.linalg.cond(J_np):.2e}")
    beauty_print(f"Condition number (Torch): {np.linalg.cond(J_torch):.2e}")
    beauty_print(f"Condition number (C++):   {np.linalg.cond(J_cpp):.2e}")

    # Display Jacobian for each end-effector
    for i, (name, display_name) in enumerate(zip(end_names, display_names)):
        start_row = i * 6
        end_row = start_row + 6
        beauty_print(f"{display_name} Jacobian (NumPy, rows {start_row}-{end_row-1}):")
        print(beauty_print_array(J_np[start_row:end_row, :], precision=6))
        beauty_print(f"{display_name} Jacobian (Torch, rows {start_row}-{end_row-1}):")
        print(beauty_print_array(J_torch[start_row:end_row, :], precision=6))
        beauty_print(f"{display_name} Jacobian (C++, rows {start_row}-{end_row-1}):")
        print(beauty_print_array(J_cpp[start_row:end_row, :], precision=6))

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

    beauty_print("✓ Humanoid Jacobian validation complete", type="success")


if __name__ == '__main__':
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")
    
    parser = argparse.ArgumentParser(description="Humanoid Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=None,
                        help='Joint angles in radians (optional)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    args = parser.parse_args()
    
    main(args)
