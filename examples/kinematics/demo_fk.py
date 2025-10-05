import numpy as np
import argparse

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.path import get_robocore_path
from robocore.transform.conversions import *


def main(args):
    urdf_path = args.urdf
    end_link = args.end_link
    joint_angles = args.joint_angles

    robot_model = RobotModel(str(urdf_path), end_link=end_link)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Kinematics Demo")
    parser.add_argument('--urdf', type=str, default=get_robocore_path("assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf"),
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='tool0',
                        help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians') 
    args = parser.parse_args()
    main(args)