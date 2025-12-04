"""Joint Trajectory Tracking Controller Demo

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
from robocore.control import JointTrajectoryController
from robocore.planning import (
    CubicPolynomialPlanner,
    QuinticPolynomialPlanner,
)
from robocore.utils.beauty_logger import beauty_print


def demo_trajectory_data_mode():
    """Demo using trajectory data dictionary."""
    beauty_print("=" * 70)
    beauty_print("Joint Trajectory Controller - Trajectory Data Mode")
    beauty_print("=" * 70)
    
    # Generate trajectory
    beauty_print("\nGenerating cubic polynomial trajectory...")
    planner = CubicPolynomialPlanner()
    result = planner.plan(
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.5, -0.3]),
        duration=2.0,
        num_points=100
    )
    t = result['t']
    q = result['q']
    qd = result['qd']
    qdd = result['qdd']
    
    print(f"  Trajectory duration: {t[-1]:.2f} seconds")
    print(f"  Number of waypoints: {len(t)}")
    print(f"  Start position: {q[0]}")
    print(f"  End position: {q[-1]}")
    
    # Create controller
    controller = JointTrajectoryController(
        Kp=100.0,
        Kd=10.0,
        Kff=50.0  # Feedforward gain
    )
    
    # Set trajectory
    controller.set_trajectory(trajectory_data={
        't': t,
        'q': q,
        'qd': qd,
        'qdd': qdd
    })
    
    beauty_print("\nController Configuration:")
    print(f"  Kp: {controller.Kp}")
    print(f"  Kd: {controller.Kd}")
    print(f"  Kff: {controller.Kff}")
    print("  Control law: τ = Kp·e + Kd·ė + Kff·q̈d")
    
    # Simulate tracking at different time points
    beauty_print("\nTrajectory Tracking Simulation:")
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for t_current in time_points:
        # Current state (simulated - in real scenario, this comes from sensors)
        idx = int(t_current / t[-1] * (len(t) - 1))
        q_current = q[idx] + 0.01 * np.random.randn(3)  # Add small noise
        qd_current = qd[idx] + 0.01 * np.random.randn(3)
        
        # Compute control torque
        tau = controller.compute(
            q=q_current,
            qd=qd_current,
            t=t_current
        )
        
        print(f"\n  Time: {t_current:.2f}s")
        print(f"    Desired: q={q[idx]}, qd={qd[idx]}")
        print(f"    Current: q={q_current}, qd={qd_current}")
        print(f"    Control torque: {tau}")


def demo_direct_mode():
    """Demo direct mode (providing desired states directly)."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Joint Trajectory Controller - Direct Mode")
    beauty_print("=" * 70)
    
    controller = JointTrajectoryController(
        Kp=100.0,
        Kd=10.0,
        Kff=50.0
    )
    
    beauty_print("\nDirect Mode (no trajectory function needed):")
    print("  Provide desired states directly in compute()")
    
    # Current state
    q = np.array([0.5, 0.25, -0.15])
    qd = np.array([0.2, 0.1, -0.05])
    
    # Desired state
    qd_desired = np.array([1.0, 0.5, -0.3])
    qdd_desired = np.array([0.5, 0.25, -0.15])
    qddd_desired = np.array([0.0, 0.0, 0.0])
    
    tau = controller.compute(
        q=q,
        qd=qd,
        qd_desired=qd_desired,
        qdd_desired=qdd_desired,
        qddd_desired=qddd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Current: q={q}, qd={qd}")
    print(f"  Desired: qd={qd_desired}, qdd={qdd_desired}, qddd={qddd_desired}")
    print(f"  Control torque: {tau}")


def demo_trajectory_function_mode():
    """Demo using trajectory function."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Joint Trajectory Controller - Trajectory Function Mode")
    beauty_print("=" * 70)
    
    # Define custom trajectory function
    def my_trajectory(t):
        """Custom trajectory: sinusoidal motion."""
        qd = np.array([
            0.5 + 0.5 * np.sin(t),
            0.3 + 0.3 * np.cos(t),
            -0.2 + 0.2 * np.sin(2 * t)
        ])
        qdd = np.array([
            0.5 * np.cos(t),
            -0.3 * np.sin(t),
            0.4 * np.cos(2 * t)
        ])
        qddd = np.array([
            -0.5 * np.sin(t),
            -0.3 * np.cos(t),
            -0.8 * np.sin(2 * t)
        ])
        return (qd, qdd, qddd)
    
    controller = JointTrajectoryController(
        Kp=100.0,
        Kd=10.0,
        Kff=50.0
    )
    
    # Set trajectory function
    controller.set_trajectory(trajectory_func=my_trajectory)
    
    beauty_print("\nCustom Trajectory Function:")
    print("  qd(t) = [0.5 + 0.5*sin(t), 0.3 + 0.3*cos(t), -0.2 + 0.2*sin(2t)]")
    
    # Simulate tracking
    beauty_print("\nTrajectory Tracking:")
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for t in time_points:
        qd_desired, qdd_desired, qddd_desired = my_trajectory(t)
        
        # Simulated current state
        q = qd_desired - 0.05 * np.random.randn(3)
        qd = qdd_desired - 0.02 * np.random.randn(3)
        
        tau = controller.compute(q=q, qd=qd, t=t)
        
        print(f"\n  Time: {t:.2f}s")
        print(f"    Desired: qd={qd_desired}, qdd={qdd_desired}")
        print(f"    Control torque: {tau}")


def demo_feedforward_benefit():
    """Demo showing benefit of feedforward term."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Feedforward vs Feedback Only Comparison")
    beauty_print("=" * 70)
    
    # Generate trajectory with acceleration
    planner = QuinticPolynomialPlanner()
    result = planner.plan(
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([1.0, 0.5, -0.3]),
        duration=2.0,
        num_points=50
    )
    t = result['t']
    q = result['q']
    qd = result['qd']
    qdd = result['qdd']
    
    # Controller with feedforward
    controller_ff = JointTrajectoryController(
        Kp=100.0,
        Kd=10.0,
        Kff=50.0  # With feedforward
    )
    
    # Controller without feedforward (Kff=0)
    controller_no_ff = JointTrajectoryController(
        Kp=100.0,
        Kd=10.0,
        Kff=0.0  # No feedforward
    )
    
    controller_ff.set_trajectory(trajectory_data={'t': t, 'q': q, 'qd': qd, 'qdd': qdd})
    controller_no_ff.set_trajectory(trajectory_data={'t': t, 'q': q, 'qd': qd, 'qdd': qdd})
    
    beauty_print("\nComparison at t=1.0s (mid-trajectory, high acceleration):")
    t_current = 1.0
    
    # Current state
    idx = int(t_current / t[-1] * (len(t) - 1))
    q_current = q[idx]
    qd_current = qd[idx]
    
    # Compute torques
    tau_ff = controller_ff.compute(q=q_current, qd=qd_current, t=t_current)
    tau_no_ff = controller_no_ff.compute(q=q_current, qd=qd_current, t=t_current)
    
    print(f"  Desired acceleration: {qdd[idx]}")
    print(f"  With feedforward: τ = {tau_ff}")
    print(f"  Without feedforward: τ = {tau_no_ff}")
    print(f"  Difference: {tau_ff - tau_no_ff}")
    print("  Note: Feedforward provides proactive compensation for acceleration")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Joint Trajectory Controller Demo")
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
    demo_trajectory_data_mode()
    demo_direct_mode()
    demo_trajectory_function_mode()
    demo_feedforward_benefit()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

