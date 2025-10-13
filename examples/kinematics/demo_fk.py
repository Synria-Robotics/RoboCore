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
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.path import get_robocore_path
from robocore.transform.conversions import *


def main(args):
    start_time = time.time()
    model_path = args.model_path
    end_link = args.end_link
    joint_angles = args.joint_angles

    robot_model = RobotModel(str(model_path), end_link=end_link)
    robot_model.summary(show_chain=True)
    robot_model.print_tree(show_fixed=True)

    T_fk = forward_kinematics(robot_model, joint_angles, backend='numpy', return_end=True)
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
    parser = argparse.ArgumentParser(description="Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        # default=get_robocore_path("assets/robot/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"),
                        default=get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml"),
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians') 
    args = parser.parse_args()
    main(args)