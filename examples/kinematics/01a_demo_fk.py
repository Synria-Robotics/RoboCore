"""Forward Kinematics Demo

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
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_fk(robot_model, backend, joint_angles):
    """Compute forward kinematics for given backend.
    
    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy' or 'torch')
    :param joint_angles: Joint angles in radians
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    T_fk = forward_kinematics(robot_model, joint_angles, return_end=True)
    position_fk = T_fk[:3, 3]
    rotation_fk = T_fk[:3, :3]
    euler_fk = matrix_to_euler(rotation_fk, seq='XYZ')
    quat_fk = matrix_to_quaternion(rotation_fk)

    elapsed_time = time.time() - start_time

    return {
        'position': position_fk,
        'euler': euler_fk,
        'quat': quat_fk,
        'rotation': rotation_fk,
        'transform': T_fk,
        'time': elapsed_time
    }


def main(args):
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    robot_model.summary(show_chain=True)
    robot_model.print_tree(show_fixed=True)

    # Compute with both backends
    results_np = compute_fk(robot_model, 'numpy', args.joint_angles)
    results_torch = compute_fk(robot_model, 'torch', args.joint_angles)

    # Convert to numpy for comparison
    pos_np = to_numpy(results_np['position'])
    pos_torch = to_numpy(results_torch['position'])
    euler_np = to_numpy(results_np['euler'])
    euler_torch = to_numpy(results_torch['euler'])
    quat_np = to_numpy(results_np['quat'])
    quat_torch = to_numpy(results_torch['quat'])

    # Display results
    beauty_print(f"End-Effector Position (m):")
    print(f"  NumPy:  {beauty_print_array(pos_np)}")
    print(f"  Torch:  {beauty_print_array(pos_torch)}")
    pos_diff = np.linalg.norm(pos_np - pos_torch)
    print(f"  Diff:   {pos_diff:.6e}")

    beauty_print(f"End-Effector Orientation (Euler XYZ, radians):")
    print(f"  NumPy:  {beauty_print_array(euler_np)}")
    print(f"  Torch:  {beauty_print_array(euler_torch)}")
    euler_diff = np.linalg.norm(euler_np - euler_torch)
    print(f"  Diff:   {euler_diff:.6e}")

    beauty_print(f"End-Effector Orientation (Euler XYZ, degrees):")
    print(f"  NumPy:  {beauty_print_array(np.rad2deg(euler_np))}")
    print(f"  Torch:  {beauty_print_array(np.rad2deg(euler_torch))}")

    beauty_print(f"End-Effector Orientation (Quaternion xyzw):")
    print(f"  NumPy:  {beauty_print_array(quat_np, precision=6)}")
    print(f"  Torch:  {beauty_print_array(quat_torch, precision=6)}")
    quat_diff = np.linalg.norm(quat_np - quat_torch)
    print(f"  Diff:   {quat_diff:.6e}")

    beauty_print(f"Rotation Matrix (NumPy):")
    print(beauty_print_array(to_numpy(results_np['rotation']), precision=6))
    beauty_print(f"Homogeneous Transformation Matrix (NumPy):")
    print(beauty_print_array(to_numpy(results_np['transform']), precision=6))

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']:.6f} seconds")
    print(f"  Torch:  {results_torch['time']:.6f} seconds")
    print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to robot model file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians')
    args = parser.parse_args()
    main(args)
    
    
    """_results_
    [RoboCore:INFO] End-Effector Position (m):
    p = [-0.17003, -0.01731, +0.20533]
    [RoboCore:INFO] End-Effector Orientation (Euler XYZ, radians):
    rpy = [+2.90030, -1.17327, -0.06082]
    [RoboCore:INFO] End-Effector Orientation (Euler XYZ, degrees):
    rpy = [+166.17499, -67.22318, -3.48471]
    [RoboCore:INFO] End-Effector Orientation (Quaternion xyzw):
    quat = [+0.828399, -0.041455, -0.552330, +0.083476]
    Note: q and -q represent the same rotation
    -quat = [-0.828399, +0.041455, +0.552330, -0.083476] (equivalent)
    [RoboCore:INFO] Rotation Matrix:
    [
    [+0.386427  +0.023531  -0.922020]
    [-0.160895  -0.982626  -0.092511]
    [-0.908178  +0.184097  -0.375927]
    ]
    [RoboCore:INFO] Homogeneous Transformation Matrix:
    [
    [+0.386427  +0.023531  -0.922020  -0.170033]
    [-0.160895  -0.982626  -0.092511  -0.017311]
    [-0.908178  +0.184097  -0.375927  +0.205325]
    [+0.000000  +0.000000  +0.000000  +1.000000]
    ]
    [RoboCore:INFO] Computation Time:  0.219255 seconds
    """