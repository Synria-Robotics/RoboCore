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
from robocore.transform.conversions import *


def main(args):
    backend = args.backend
    rc.set_backend(backend)

    start_time = time.time()

    robot_model = RobotModel(str(args.model_path), end_link=args.end_link)
    robot_model.summary(show_chain=True)
    robot_model.print_tree(show_fixed=True)

    T_fk = forward_kinematics(robot_model, args.joint_angles, return_end=True)
    position_fk = T_fk[:3, 3]
    rotation_fk = T_fk[:3, :3]

    results = {}
    euler_fk = matrix_to_euler(rotation_fk, seq='xyz')
    quat_fk = matrix_to_quaternion(rotation_fk)

    results['fk'] = {
        'transform': T_fk,
        'position': position_fk,
        'rotation': rotation_fk,
        'euler_xyz': euler_fk,
        'quaternion_xyzw': quat_fk  # Quaternion in xyzw order
        }

    beauty_print(f"End-Effector Position (m):")
    print(f"  p = {beauty_print_array(position_fk)}")
    beauty_print(f"End-Effector Orientation (Euler XYZ, radians):")
    print(f"  rpy = {beauty_print_array(euler_fk)}")
    beauty_print(f"End-Effector Orientation (Euler XYZ, degrees):")
    print(f"  rpy = {beauty_print_array(np.rad2deg(euler_fk))}")
    beauty_print(f"End-Effector Orientation (Quaternion xyzw):")
    print(f"  quat = {beauty_print_array(quat_fk, precision=6)}")
    # Add note about quaternion sign ambiguity
    quat_neg = -quat_fk
    print(f"  Note: q and -q represent the same rotation")
    print(f"  -quat = {beauty_print_array(quat_neg, precision=6)} (equivalent)")
    beauty_print(f"Rotation Matrix:")
    print(beauty_print_array(rotation_fk, precision=6))
    beauty_print(f"Homogeneous Transformation Matrix:")
    print(beauty_print_array(T_fk, precision=6))
    end_time = time.time()
    beauty_print(f"Computation Time: {end_time - start_time: .6f} seconds")


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians')
    parser.add_argument('--backend', type=str, default='numpy',
                        help='Backend to use for computation (default: numpy)')
    args = parser.parse_args()
    main(args)
    
    
    """_results_
    [RoboCore:INFO] End-Effector Position (m):
    p = [+0.17006, +0.01704, +0.20533]
    [RoboCore:INFO] End-Effector Orientation (Euler XYZ, radians):
    rpy = [+2.68621, +1.13891, +2.74547]
    [RoboCore:INFO] End-Effector Orientation (Euler XYZ, degrees):
    rpy = [+153.90843, +65.25471, +157.30364]
    [RoboCore:INFO] End-Effector Orientation (Quaternion xyzw):
    quat = [+0.042114, +0.828366, +0.083037, +0.552396]
    Note: q and -q represent the same rotation
    -quat = [-0.042114, -0.828366, -0.083037, -0.552396] (equivalent)
    [RoboCore:INFO] Rotation Matrix:
    [
    [-0.386171  -0.021966  +0.922166]
    [+0.161510  +0.982663  +0.091042]
    [-0.908178  +0.184097  -0.375928]
    ]
    [RoboCore:INFO] Homogeneous Transformation Matrix:
    [
    [-0.386171  -0.021966  +0.922166  +0.170060]
    [+0.161510  +0.982663  +0.091042  +0.017041]
    [-0.908178  +0.184097  -0.375928  +0.205325]
    [+0.000000  +0.000000  +0.000000  +1.000000]
    ]
    [RoboCore:INFO] Computation Time:  0.002994 seconds
    """