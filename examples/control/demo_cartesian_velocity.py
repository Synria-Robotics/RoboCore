"""Cartesian Velocity Controller Demo

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
from robocore.control import CartesianVelocityController
from robocore.modeling import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print


def demo_velocity_control():
    """Demo basic cartesian velocity control."""
    beauty_print("=" * 70)
    beauty_print("Cartesian Velocity Controller Demo")
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
    controller = CartesianVelocityController(
        robot_model=robot,
        Kp=np.diag([50, 50, 50, 30, 30, 30])  # Velocity gains
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Kp (diagonal): {np.diag(controller.Kp)}")
    print("  Control law: τ = J^T·Kp·(ẋd - ẋ) + g(q)")
    
    # Current joint state
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    
    # Compute current end-effector velocity
    from robocore.kinematics.jacobian import jacobian
    J = jacobian(robot, q)
    xd_current = J @ qd
    
    beauty_print("\nCurrent State:")
    print(f"  Joint velocity: {qd}")
    print(f"  End-effector velocity: {xd_current}")
    
    # Desired velocity: move in x direction at 0.1 m/s
    xdd_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    beauty_print("\nDesired End-Effector Velocity:")
    print(f"  Linear velocity: {xdd_desired[:3]} m/s")
    print(f"  Angular velocity: {xdd_desired[3:]} rad/s")
    
    # Compute control torque
    tau = controller.compute(
        q=q,
        qd=qd,
        xdd_desired=xdd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")
    print(f"  Torque magnitude: {np.linalg.norm(tau)}")


def demo_constant_velocity():
    """Demo maintaining constant velocity."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Constant Velocity Tracking Demo")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    controller = CartesianVelocityController(
        robot_model=robot,
        Kp=np.diag([50, 50, 50, 30, 30, 30])
    )
    
    # Desired constant velocity
    xdd_desired = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.1])
    
    beauty_print("\nSimulation (maintaining constant velocity):")
    print(f"  Target velocity: {xdd_desired}")
    
    # Simulate multiple steps
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    
    from robocore.kinematics.jacobian import jacobian
    
    for step in range(5):
        J = jacobian(robot, q)
        xd_current = J @ qd
        
        tau = controller.compute(
            q=q,
            qd=qd,
            xdd_desired=xdd_desired
        )
        
        print(f"\n  Step {step + 1}:")
        print(f"    Current velocity: {xd_current}")
        print(f"    Velocity error: {xdd_desired - xd_current}")
        print(f"    Control torque: {tau}")
        
        # Simulate velocity change (simplified)
        qd = qd + 0.01 * np.linalg.pinv(J) @ (xdd_desired - xd_current)


def demo_angular_velocity():
    """Demo angular velocity control."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Angular Velocity Control Demo")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    controller = CartesianVelocityController(
        robot_model=robot,
        Kp=np.diag([50, 50, 50, 30, 30, 30])
    )
    
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    
    # Desired: rotate around z-axis at 0.5 rad/s
    xdd_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    
    beauty_print("\nDesired Angular Velocity:")
    print(f"  Angular velocity: {xdd_desired[3:]} rad/s")
    print("  (Rotation around z-axis)")
    
    tau = controller.compute(
        q=q,
        qd=qd,
        xdd_desired=xdd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Cartesian Velocity Controller Demo")
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
    demo_velocity_control()
    demo_constant_velocity()
    demo_angular_velocity()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

