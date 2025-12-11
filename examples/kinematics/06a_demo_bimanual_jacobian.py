"""Bimanual Jacobian validation and comparison.

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
from robocore.kinematics.bimanual import bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def compute_jacobian_results(left_model, right_model, backend, q_left, q_right, mode='indep', device=None):
    """Compute bimanual Jacobian results for given backend.
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param backend: Backend name ('numpy' or 'torch')
    :param q_left: Left joint configuration
    :param q_right: Right joint configuration
    :param mode: 'indep'|'relative'|'mirror'
    :param device: Device for torch backend
    :return: Dictionary with Jacobian matrix and computation time
    """
    rc.set_backend(backend)
    if backend == 'torch' and device is None:
        import torch
        device = torch.device('cpu')
        rc.set_backend('torch', device=str(device))

    start_time = time.perf_counter()
    J = bimanual_jacobian(left_model, right_model, q_left, q_right, mode=mode, device=device)
    elapsed_time = (time.perf_counter() - start_time) * 1000

    return {
        'J': J,
        'time': elapsed_time
    }


def main(args):
    left_model = RobotModel(args.model_path, base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(args.model_path, base_link=args.right_base_link, end_link=args.right_end_link)
    
    beauty_print(f"Bimanual Jacobian Validation: {left_model.name} (Left: {left_model.num_chain_dof} DOF, Right: {right_model.num_chain_dof} DOF)", type="module")
    beauty_print(f"Mode: {args.mode}", type="info")

    q_left = np.array(args.q_left)
    q_right = np.array(args.q_right)

    beauty_print(f"Left arm joint configuration (rad):")
    print(f"  q_left = {beauty_print_array(q_left)}")
    beauty_print(f"Right arm joint configuration (rad):")
    print(f"  q_right = {beauty_print_array(q_right)}")

    # Compute with both backends
    results_np = compute_jacobian_results(left_model, right_model, 'numpy', q_left, q_right, mode=args.mode)
    import torch
    device = torch.device(args.device)
    q_left_torch = torch.tensor(q_left, dtype=torch.float64, device=device)
    q_right_torch = torch.tensor(q_right, dtype=torch.float64, device=device)
    results_torch = compute_jacobian_results(left_model, right_model, 'torch', q_left_torch, q_right_torch, mode=args.mode, device=device)

    # Convert to numpy for comparison
    J_np = to_numpy(results_np['J'])
    J_torch = to_numpy(results_torch['J'])

    beauty_print("[1] Jacobian Comparison (NumPy vs Torch)", type="module", centered=False)
    beauty_print(f"Jacobian shape: {J_np.shape}")
    beauty_print(f"Condition number (NumPy): {np.linalg.cond(J_np):.2e}")
    beauty_print(f"Condition number (Torch): {np.linalg.cond(J_torch):.2e}")

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

    diff = J_np - J_torch
    beauty_print("NumPy vs Torch:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff, 'fro'):.3e}")

    beauty_print("[2] Performance Comparison", type="module", centered=False)
    beauty_print(f"NumPy:  {results_np['time']:.4f} ms")
    beauty_print(f"Torch:  {results_torch['time']:.4f} ms")
    if results_np['time'] > 0:
        ratio = results_torch['time'] / results_np['time']
        beauty_print(f"Ratio:  {ratio:.2f}x")

    beauty_print("✓ Bimanual Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    # Bessica is a dual-arm robot
    model_path = synriard.get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

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
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative', 'mirror'],
                        help='Jacobian mode: indep (independent), relative (relative transform), mirror (mirror mode)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)

