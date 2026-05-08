"""Joint Position Controller Demo

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse

import robocore as rc
from robocore.control import JointPositionController
from robocore.utils.beauty_logger import beauty_print


def demo_pd_controller():
    """Demo PD controller (no integral term)."""
    beauty_print("=" * 70)
    beauty_print("Joint Position Controller - PD Mode Demo")
    beauty_print("=" * 70)
    
    # Create PD controller
    controller = JointPositionController(
        Kp=100.0,  # Position gain
        Kd=10.0,   # Velocity gain
        use_integral=False  # PD mode
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Mode: PD (Proportional-Derivative)")
    print(f"  Kp: {controller.Kp}")
    print(f"  Kd: {controller.Kd}")
    print(f"  Ki: None (integral disabled)")
    
    # Simulate control loop
    beauty_print("\nSimulation:")
    print("  Current state: q = [0.0, 0.0, 0.0], qd = [0.0, 0.0, 0.0]")
    print("  Desired state: qd = [1.0, 0.5, -0.3], qdd = [0.0, 0.0, 0.0]")
    
    q = np.array([0.0, 0.0, 0.0])
    qd = np.array([0.0, 0.0, 0.0])
    qd_desired = np.array([1.0, 0.5, -0.3])
    qdd_desired = np.array([0.0, 0.0, 0.0])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        qd_desired=qd_desired,
        qdd_desired=qdd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  τ = {tau}")
    print(f"  Error: {qd_desired - q}")
    # Get normalized gains for display
    Kp_display = controller.Kp if controller.Kp.ndim > 0 else np.eye(3) * controller.Kp
    Kd_display = controller.Kd if controller.Kd.ndim > 0 else np.eye(3) * controller.Kd
    print(f"  Position term (Kp·e): {Kp_display @ (qd_desired - q)}")
    print(f"  Velocity term (Kd·ė): {Kd_display @ (qdd_desired - qd)}")


def demo_pid_controller():
    """Demo PID controller (with integral term)."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Joint Position Controller - PID Mode Demo")
    beauty_print("=" * 70)
    
    # Create PID controller
    controller = JointPositionController(
        Kp=100.0,  # Position gain
        Kd=10.0,   # Velocity gain
        Ki=1.0,    # Integral gain
        use_integral=True,  # PID mode
        anti_windup=True,   # Enable anti-windup
        integral_limit=10.0  # Integral saturation limit
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Mode: PID (Proportional-Integral-Derivative)")
    print(f"  Kp: {controller.Kp}")
    print(f"  Kd: {controller.Kd}")
    print(f"  Ki: {controller.Ki}")
    print(f"  Anti-windup: Enabled (limit: {controller.integral_limit})")
    
    # Simulate control loop with steady-state error
    beauty_print("\nSimulation (with steady-state error):")
    print("  Current state: q = [0.9, 0.45, -0.27], qd = [0.0, 0.0, 0.0]")
    print("  Desired state: qd = [1.0, 0.5, -0.3], qdd = [0.0, 0.0, 0.0]")
    print("  Note: Small error persists (e.g., due to gravity)")
    
    q = np.array([0.9, 0.45, -0.27])
    qd = np.array([0.0, 0.0, 0.0])
    qd_desired = np.array([1.0, 0.5, -0.3])
    qdd_desired = np.array([0.0, 0.0, 0.0])
    
    # Simulate multiple steps to show integral accumulation
    beauty_print("\nControl Loop (multiple steps):")
    for step in range(5):
        tau = controller.compute(
            q=q,
            qd=qd,
            qd_desired=qd_desired,
            qdd_desired=qdd_desired
        )
        
        error = qd_desired - q
        print(f"\n  Step {step + 1}:")
        print(f"    Error: {error}")
        print(f"    Integral: {controller.integral_error}")
        print(f"    Torque: {tau}")
        
        # Simulate robot moving (simplified)
        q = q + 0.01 * (qd_desired - q)  # Move towards desired


def demo_matrix_gains():
    """Demo controller with matrix gains (different gains per joint)."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Joint Position Controller - Matrix Gains Demo")
    beauty_print("=" * 70)
    
    # Create controller with diagonal matrix gains
    # Different gains for each joint
    Kp = np.diag([100.0, 150.0, 80.0])  # Different position gains
    Kd = np.diag([10.0, 15.0, 8.0])    # Different velocity gains
    
    controller = JointPositionController(
        Kp=Kp,
        Kd=Kd,
        use_integral=False
    )
    
    beauty_print("\nController Configuration:")
    print(f"  Kp (diagonal): {np.diag(controller.Kp)}")
    print(f"  Kd (diagonal): {np.diag(controller.Kd)}")
    print("  Note: Different gains allow fine-tuning per joint")
    
    # Compute control
    q = np.array([0.0, 0.0, 0.0])
    qd = np.array([0.0, 0.0, 0.0])
    qd_desired = np.array([1.0, 0.5, -0.3])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        qd_desired=qd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  τ = {tau}")
    print("  Note: Each joint receives different control effort based on its gain")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Joint Position Controller Demo")
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
    demo_pd_controller()
    demo_pid_controller()
    demo_matrix_gains()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

