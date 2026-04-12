"""Humanoid Forward Kinematics Parallel Demo

This demo demonstrates parallel/batch humanoid forward kinematics computation.
It compares serial vs parallel processing performance for multiple joint configurations.

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

import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Load robot model with unified configuration space
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link)

    # Generate random joint configurations (full DOF)
    num_batch = args.num_configs
    beauty_print(f"Generating {num_batch} Random Joint Configurations", type="info")
    # Generate full DOF configurations using new method
    q_batch = robot_model.random_q_full_batch(num_batch, seed=args.seed, scale=args.scale)

    beauty_print(f"Processing {num_batch} joint configuration(s) using {backend} backend", type="info")

    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]

    # Batch processing
    start_time = time.time()
    results_batch = []
    for i in range(num_batch):
        T_left_thumb = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[0], return_end=True)
        T_right_thumb = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[1], return_end=True)
        T_left_toe = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[2], return_end=True)
        T_right_toe = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[3], return_end=True)
        results_batch.append({
            'left_thumb': to_numpy(T_left_thumb),
            'right_thumb': to_numpy(T_right_thumb),
            'left_toe': to_numpy(T_left_toe),
            'right_toe': to_numpy(T_right_toe),
        })
    batch_time = time.time() - start_time

    # Serial processing (for comparison)
    start_time = time.time()
    results_serial = []
    for i in range(num_batch):
        T_left_thumb = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[0], return_end=True)
        T_right_thumb = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[1], return_end=True)
        T_left_toe = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[2], return_end=True)
        T_right_toe = robot_model.fk(q_batch[i], base_link=args.base_link, end_link=end_links[3], return_end=True)
        results_serial.append({
            'left_thumb': to_numpy(T_left_thumb),
            'right_thumb': to_numpy(T_right_thumb),
            'left_toe': to_numpy(T_left_toe),
            'right_toe': to_numpy(T_right_toe),
        })
    serial_time = time.time() - start_time

    # Display results
    beauty_print(f"Batch Processing Results:", type="module")
    end_effector_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']

    for name, display_name in zip(end_effector_names, display_names):
        beauty_print(f"{display_name} Positions (first 3 samples):")
        for i in range(min(3, num_batch)):
            pos = results_batch[i][name][:3, 3]
            print(f"  Sample {i+1}: {beauty_print_array(pos)}")

    beauty_print(f"Performance Comparison:", type="module")
    print(f"  Serial Time:   {serial_time*1000:.4f} ms ({serial_time/num_batch*1000:.4f} ms/sample)")
    print(f"  Batch Time:    {batch_time*1000:.4f} ms ({batch_time/num_batch*1000:.4f} ms/sample)")
    if batch_time > 0:
        print(f"  Speedup:       {serial_time / batch_time:.2f}x")


if __name__ == "__main__":
    from openrd import get_model_path

    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")

    parser = argparse.ArgumentParser(description="Humanoid Forward Kinematics Parallel Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--num-configs', type=int, default=50, help='Number of joint configurations to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generating configurations')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for joint limits (0.0 to 1.0)')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch', 'cpp'],
                        help='Backend to use for computation (default: numpy)')
    args = parser.parse_args()
    main(args)
