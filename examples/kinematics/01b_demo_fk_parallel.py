"""Forward Kinematics Parallel Demo

This demo demonstrates parallel/batch forward kinematics computation.
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

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import *


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    # Parse joint configurations
    joint_configs = args.joint_angles
    if isinstance(joint_configs[0], list):
        # Already a list of lists
        pass
    else:
        # Single configuration, wrap it
        joint_configs = [joint_configs]

    num_batch = len(joint_configs)
    beauty_print(f"Processing {num_batch} joint configuration(s) using {backend} backend")

    # Load robot model
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    # Single configuration example
    beauty_print("Single Configuration Example", type="module", centered=True)
    q_single = joint_configs[0]
    start_time = time.time()
    T_single = forward_kinematics(robot_model, q_single, return_end=True)
    single_time = time.time() - start_time
    beauty_print(f"Single FK time: {single_time:.6f} seconds")
    
    position_single = T_single[:3, 3]
    beauty_print(f"End-Effector Position: {beauty_print_array(position_single)}")

    # Batch processing example
    beauty_print("Batch Processing Example", type="module", centered=True)
    q_batch = np.array(joint_configs)
    start_time = time.time()
    T_batch = forward_kinematics(robot_model, q_batch, return_end=True)
    batch_time = time.time() - start_time
    beauty_print(f"Batch FK time: {batch_time:.6f} seconds")
    beauty_print(f"Average time per configuration: {batch_time/num_batch:.6f} seconds")
    
    if num_batch > 1:
        speedup = (single_time * num_batch) / batch_time
        beauty_print(f"Effective speedup: {speedup:.2f}x")

    # Display results for each configuration
    beauty_print("Results for Each Configuration", type="module", centered=True)
    for i in range(num_batch):
        T_fk = T_batch[i] if T_batch.ndim == 3 else T_batch
        q_config = joint_configs[i]
        
        beauty_print(f"\nConfiguration {i+1}:")
        beauty_print(f"  Joint angles: {beauty_print_array(np.array(q_config))}")
        
        position_fk = T_fk[:3, 3]
        rotation_fk = T_fk[:3, :3]
        euler_fk = matrix_to_euler(rotation_fk, seq='xyz')
        quat_fk = matrix_to_quaternion(rotation_fk)

        beauty_print(f"  End-Effector Position (m):")
        print(f"    p = {beauty_print_array(position_fk)}")
        beauty_print(f"  End-Effector Orientation (Euler XYZ, degrees):")
        print(f"    rpy = {beauty_print_array(np.rad2deg(euler_fk))}")
        beauty_print(f"  End-Effector Orientation (Quaternion xyzw):")
        print(f"    quat = {beauty_print_array(quat_fk, precision=6)}")
        
        if args.show_matrices:
            beauty_print(f"  Homogeneous Transformation Matrix:")
            print(beauty_print_array(T_fk, precision=6))


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Forward Kinematics Parallel Demo - Compare serial vs parallel/batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Use default configurations (6 identical configs)
        python 01c_demo_fk_parallel.py

        # Use torch backend for better batch performance
        python 01c_demo_fk_parallel.py --backend torch

        # Provide custom joint configurations
        python 01c_demo_fk_parallel.py --joint-angles \\
            0.1 0.2 -0.3 0.0 0.5 -0.2 \\
            0.2 0.3 -0.4 0.1 0.6 -0.3 \\
            0.0 0.1 -0.2 0.0 0.4 -0.1

        # Show transformation matrices for each configuration
        python 01c_demo_fk_parallel.py --show-matrices
        """
    )
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', 
                        default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians (flattened list, will be reshaped)')
    parser.add_argument('--num-joints', type=int, default=6,
                        help='Number of joints per configuration (default: 6)')
    parser.add_argument('--backend', type=str, default='numpy',
                        choices=['numpy', 'torch'],
                        help='Backend to use for computation (default: numpy, torch recommended for batch)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-matrices', action='store_true',
                        help='Show full transformation matrices for each configuration')
    args = parser.parse_args()
    
    # Reshape joint angles into list of configurations
    num_joints = args.num_joints
    joint_angles_flat = args.joint_angles
    if len(joint_angles_flat) % num_joints != 0:
        raise ValueError(f"Total number of joint angles ({len(joint_angles_flat)}) must be divisible by num-joints ({num_joints})")
    
    num_batch = len(joint_angles_flat) // num_joints
    args.joint_angles = [joint_angles_flat[i*num_joints:(i+1)*num_joints] 
                        for i in range(num_batch)]
    
    main(args)