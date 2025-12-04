"""Cartesian Position Controller Demo

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

import robocore as rc
from robocore.control import CartesianPositionController
from robocore.modeling import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.transform.conversions import matrix_to_quaternion
from robocore.utils.beauty_logger import beauty_print


def demo_basic_position_control():
    """Demo basic cartesian position control."""
    beauty_print("=" * 70)
    beauty_print("Cartesian Position Controller Demo")
    beauty_print("=" * 70)
    
    # Load robot model
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        beauty_print("\nNote: synriard not available, using dummy model path")
        model_path = "path/to/robot.urdf"
    
    beauty_print(f"\nLoading robot model: {model_path}")
    robot = RobotModel(model_path)
    
    # Create controller
    controller = CartesianPositionController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),  # Position and orientation gains
        Kd=np.diag([10, 10, 10, 5, 5, 5])
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Kp (diagonal): {np.diag(controller.Kp)}")
    print(f"  Kd (diagonal): {np.diag(controller.Kd)}")
    print("  Control law: τ = J^T·[Kp·(xd - x) + Kd·(ẋd - ẋ)] + g(q)")
    
    # Current joint state
    q = np.zeros(robot.num_chain_dof)
    qd = np.zeros(robot.num_chain_dof)
    
    # Get current end-effector pose
    T_current = forward_kinematics(robot, q, return_end=True)
    pos_current = T_current[:3, 3]
    quat_current = matrix_to_quaternion(T_current[:3, :3])
    
    beauty_print("\nCurrent End-Effector Pose:")
    print(f"  Position: {pos_current}")
    print(f"  Quaternion: {quat_current}")
    
    # Desired pose: move 0.1m in x direction, same orientation
    pos_desired = pos_current + np.array([0.1, 0.0, 0.0])
    quat_desired = quat_current  # Keep same orientation
    xd_desired = np.concatenate([pos_desired, quat_desired])
    
    beauty_print("\nDesired End-Effector Pose:")
    print(f"  Position: {pos_desired}")
    print(f"  Quaternion: {quat_desired}")
    
    # Compute control torque
    tau = controller.compute(
        q=q,
        qd=qd,
        xd_desired=xd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")
    print(f"  Torque magnitude: {np.linalg.norm(tau)}")


def demo_orientation_control():
    """Demo orientation control."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Orientation Control Demo")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    controller = CartesianPositionController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5])
    )
    
    # Current state
    q = np.zeros(robot.num_chain_dof)
    qd = np.zeros(robot.num_chain_dof)
    
    T_current = forward_kinematics(robot, q, return_end=True)
    pos_current = T_current[:3, 3]
    quat_current = matrix_to_quaternion(T_current[:3, :3])
    
    # Desired: same position, rotated 90 degrees around z-axis
    from robocore.transform.conversions import rpy_to_quaternion
    quat_desired = rpy_to_quaternion(0, 0, np.pi/2)
    xd_desired = np.concatenate([pos_current, quat_desired])
    
    beauty_print("\nOrientation Change:")
    print(f"  Current quaternion: {quat_current}")
    print(f"  Desired quaternion: {quat_desired}")
    print("  (90 degree rotation around z-axis)")
    
    tau = controller.compute(
        q=q,
        qd=qd,
        xd_desired=xd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")


def demo_different_gains():
    """Demo using different gains for position and orientation."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Different Gains for Position and Orientation")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    # Higher gains for position, lower for orientation
    controller = CartesianPositionController(
        robot_model=robot,
        Kp=np.diag([200, 200, 200, 30, 30, 30]),  # High position, low orientation
        Kd=np.diag([20, 20, 20, 3, 3, 3])
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Position gains (Kp[:3]): {np.diag(controller.Kp)[:3]}")
    print(f"  Orientation gains (Kp[3:]): {np.diag(controller.Kp)[3:]}")
    print("  Note: Higher position gains for faster position response")
    
    q = np.zeros(robot.num_chain_dof)
    qd = np.zeros(robot.num_chain_dof)
    
    T_current = forward_kinematics(robot, q, return_end=True)
    pos_current = T_current[:3, 3]
    quat_current = matrix_to_quaternion(T_current[:3, :3])
    
    # Desired: move and rotate
    pos_desired = pos_current + np.array([0.1, 0.05, 0.0])
    from robocore.transform.conversions import rpy_to_quaternion
    quat_desired = rpy_to_quaternion(0, 0, np.pi/4)
    xd_desired = np.concatenate([pos_desired, quat_desired])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        xd_desired=xd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Cartesian Position Controller Demo")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to robot model (URDF or MJCF)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='numpy',
        choices=['numpy', 'torch'],
        help='Backend to use (numpy or torch)'
    )
    args = parser.parse_args()
    
    rc.set_backend(args.backend)
    
    # Run demos
    demo_basic_position_control()
    demo_orientation_control()
    demo_different_gains()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

