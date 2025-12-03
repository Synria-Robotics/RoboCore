"""
Forward Kinematics Demo with Pytorch Kinematics

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
import torch
import pytorch_kinematics as pk
import argparse
import time
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import matrix_to_euler, matrix_to_quaternion


def main(args):
    start_time = time.time()
    import os
    model_path = str(args.model_path)  # Ensure it's a string path
    end_link = args.end_link
    joint_angles = args.joint_angles

    # Verify file path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"URDF file not found: {model_path}")

    # Convert joint angles to torch tensor
    joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0)

    # Load robot description from URDF and specify end effector link
    # pytorch_kinematics can accept file path or file content
    # Read file as bytes to avoid encoding declaration issues
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    # pytorch_kinematics can handle bytes input
    # Specify base_link as root to match robocore's coordinate system
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name='base_link')
    # Prints out the (nested) tree of links
    print(chain)
    # Prints out list of joint names
    print(chain.get_joint_parameter_names())
    
    ret = chain.forward_kinematics(joint_angles_tensor, end_only=False)

    # Look up the transform for a specific link
    tg = ret[end_link]
    # Get transform matrix (1,4,4), then convert to separate position and rotation matrix
    m = tg.get_matrix()  # Shape: (1, 4, 4)
    
    # Extract position and rotation matrix
    position_fk = m[0, :3, 3].detach().cpu().numpy()  # Shape: (3,)
    rotation_fk = m[0, :3, :3].detach().cpu().numpy()  # Shape: (3, 3)
    
    # Build homogeneous transformation matrix
    T_fk = np.eye(4)
    T_fk[:3, :3] = rotation_fk
    T_fk[:3, 3] = position_fk

    # Convert rotation matrix to Euler angles and quaternion
    euler_fk = matrix_to_euler(rotation_fk, seq='xyz')
    quat_fk = matrix_to_quaternion(rotation_fk)  # Returns [x, y, z, w]

    results = {
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
    args = parser.parse_args()
    main(args)