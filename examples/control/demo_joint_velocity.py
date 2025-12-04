"""Joint Velocity Controller Demo

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
from robocore.control import JointVelocityController
from robocore.utils.beauty_logger import beauty_print


def demo_velocity_control():
    """Demo basic velocity control."""
    beauty_print("=" * 70)
    beauty_print("Joint Velocity Controller Demo")
    beauty_print("=" * 70)
    
    # Create velocity controller
    controller = JointVelocityController(Kp=50.0)
    
    beauty_print("\nController Configuration:")
    print(f"  Kp: {controller.Kp}")
    print("  Control law: τ = Kp·(q̇d - q̇)")
    
    # Simulate control loop
    beauty_print("\nSimulation:")
    print("  Current velocity: qd = [0.1, -0.2, 0.05]")
    print("  Desired velocity: qdd = [0.5, 0.3, -0.1]")
    
    q = np.array([0.0, 0.0, 0.0])  # Position not used
    qd = np.array([0.1, -0.2, 0.05])
    qdd_desired = np.array([0.5, 0.3, -0.1])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        qdd_desired=qdd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  τ = {tau}")
    print(f"  Velocity error: {qdd_desired - qd}")
    print(f"  Control effort: {tau}")


def demo_constant_velocity():
    """Demo maintaining constant velocity."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Constant Velocity Tracking Demo")
    beauty_print("=" * 70)
    
    controller = JointVelocityController(Kp=50.0)
    
    # Desired constant velocity
    qdd_desired = np.array([0.5, 0.3, -0.1])
    
    beauty_print("\nSimulation (maintaining constant velocity):")
    print(f"  Target velocity: {qdd_desired}")
    
    # Simulate multiple steps
    qd = np.array([0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0])
    
    for step in range(5):
        tau = controller.compute(
            q=q,
            qd=qd,
            qdd_desired=qdd_desired
        )
        
        print(f"\n  Step {step + 1}:")
        print(f"    Current velocity: {qd}")
        print(f"    Velocity error: {qdd_desired - qd}")
        print(f"    Control torque: {tau}")
        
        # Simulate velocity change (simplified)
        qd = qd + 0.1 * (qdd_desired - qd)


def demo_matrix_gains():
    """Demo controller with matrix gains."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Matrix Gains Demo")
    beauty_print("=" * 70)
    
    # Different velocity gains for each joint
    Kp = np.diag([50.0, 80.0, 40.0])
    
    controller = JointVelocityController(Kp=Kp)
    
    beauty_print("\nController Configuration:")
    print(f"  Kp (diagonal): {np.diag(controller.Kp)}")
    print("  Note: Different gains allow different response speeds per joint")
    
    q = np.array([0.0, 0.0, 0.0])
    qd = np.array([0.1, 0.1, 0.1])
    qdd_desired = np.array([0.5, 0.5, 0.5])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        qdd_desired=qdd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  τ = {tau}")
    print("  Note: Joint 2 has highest gain, so receives more control effort")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Joint Velocity Controller Demo")
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
    demo_matrix_gains()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

