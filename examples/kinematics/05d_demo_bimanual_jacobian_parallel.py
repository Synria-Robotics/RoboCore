"""Bimanual Jacobian Parallel Demo

This demo demonstrates parallel/batch bimanual Jacobian computation.
It shows how to use batch processing for multiple joint configurations.

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
from robocore.kinematics.bimanual import bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model (Bessica is a dual-arm robot, use same model with different base/end links)
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)

    # Generate random joint configurations
    num_configs = args.num_configs
    beauty_print("Generating Random Joint Configurations", type="module", centered=True)
    q_left_batch = left_model.random_q_batch(num_configs, seed=args.seed, scale=args.scale)
    q_right_batch = right_model.random_q_batch(num_configs, seed=args.seed+1, scale=args.scale)

    beauty_print(f"Processing {num_configs} joint configuration(s) using {backend} backend", type="module", centered=True)

    # Batch processing
    start_time = time.time()
    J_batch = bimanual_jacobian(
        left_model, right_model, q_left_batch, q_right_batch,
        mode=args.mode
    )
    batch_time = time.time() - start_time

    # Serial processing (for comparison)
    start_time = time.time()
    J_serial = []
    for i in range(num_configs):
        J = bimanual_jacobian(
            left_model, right_model, q_left_batch[i], q_right_batch[i],
            mode=args.mode
        )
        J_serial.append(J)
    serial_time = time.time() - start_time

    # Convert to numpy for analysis
    J_batch_np = to_numpy(J_batch)
    J_serial_np = [to_numpy(J) for J in J_serial]

    # Display results
    beauty_print(f"Batch Processing Results:", type="module")
    if J_batch_np.ndim == 3:
        print(f"  Batch Jacobian Shape: {J_batch_np.shape}")
        print(f"  Serial Jacobian Shape: {J_serial_np[0].shape} (per config)")
        
        # Show first few condition numbers
        beauty_print(f"Condition Numbers (first 3 samples):")
        for i in range(min(3, num_configs)):
            cond = np.linalg.cond(J_batch_np[i])
            print(f"  Sample {i+1}: {cond:.6e}")
    else:
        print(f"  Jacobian Shape: {J_batch_np.shape}")

    beauty_print(f"Performance Comparison:", type="module")
    print(f"  Serial Time:   {serial_time*1000:.4f} ms ({serial_time/num_configs*1000:.4f} ms/sample)")
    print(f"  Batch Time:    {batch_time*1000:.4f} ms ({batch_time/num_configs*1000:.4f} ms/sample)")
    if batch_time > 0:
        print(f"  Speedup:       {serial_time / batch_time:.2f}x")

    # Show sub-matrices for independent mode (first sample)
    if args.mode == 'indep' and J_batch_np.ndim == 3:
        nL = left_model.num_chain_dof
        nR = right_model.num_chain_dof
        J_first = J_batch_np[0]
        J_L = J_first[:6, :nL]
        J_R = J_first[6:, nL:]

        beauty_print(f"First Sample - Left Arm Jacobian (6 x {nL}):")
        print(beauty_print_array(J_L, precision=4))

        beauty_print(f"First Sample - Right Arm Jacobian (6 x {nR}):")
        print(beauty_print_array(J_R, precision=4))


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Bimanual Jacobian Parallel Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--num-configs', type=int, default=50, help='Number of joint configurations to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generating configurations')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for joint limits (0.0 to 1.0)')
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative'],
                        help='Jacobian mode: indep (independent), relative (relative constraint)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy)')
    args = parser.parse_args()
    main(args)
