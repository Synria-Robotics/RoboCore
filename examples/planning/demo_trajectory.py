#!/usr/bin/env python3
"""RoboCore Module

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

import os
import numpy as np
import argparse
from pathlib import Path

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics

# Trajectory planning
from robocore.planning.trajectory import (
    cubic_polynomial_trajectory,
    quintic_polynomial_trajectory,
    linear_joint_trajectory,
    multi_waypoint_trajectory,
    linear_cartesian_trajectory,
    circular_cartesian_trajectory,
    trapezoidal_velocity_profile,
    s_curve_velocity_profile,
)


def demo_joint_space_trajectories(model, q_start, q_end, plot=False):
    """Demonstrate joint space polynomial trajectories."""
    print("\n" + "="*70)
    print("JOINT SPACE POLYNOMIAL TRAJECTORIES")
    print("="*70)
    
    duration = 2.0
    num_points = 100
    
    # 1. Linear trajectory
    print("\n1. Linear Joint Trajectory (constant velocity)")
    t_lin, q_lin, qd_lin, qdd_lin = linear_joint_trajectory(
        q_start, q_end, duration, num_points
    )
    print(f"   Duration: {t_lin[-1]:.3f}s")
    print(f"   Max velocity: {np.max(np.abs(qd_lin)):.3f} rad/s")
    print(f"   Max acceleration: {np.max(np.abs(qdd_lin)):.3f} rad/s²")
    print(f"   ⚠️  Note: Infinite acceleration at endpoints!")
    
    # 2. Cubic polynomial trajectory
    print("\n2. Cubic Polynomial Trajectory (C1 continuous)")
    t_cub, q_cub, qd_cub, qdd_cub = cubic_polynomial_trajectory(
        q_start, q_end, duration, num_points
    )
    print(f"   Duration: {t_cub[-1]:.3f}s")
    print(f"   Max velocity: {np.max(np.abs(qd_cub)):.3f} rad/s")
    print(f"   Max acceleration: {np.max(np.abs(qdd_cub)):.3f} rad/s²")
    print(f"   ✓ Continuous position and velocity")
    
    # 3. Quintic polynomial trajectory
    print("\n3. Quintic Polynomial Trajectory (C2 continuous)")
    t_qui, q_qui, qd_qui, qdd_qui = quintic_polynomial_trajectory(
        q_start, q_end, duration, num_points
    )
    print(f"   Duration: {t_qui[-1]:.3f}s")
    print(f"   Max velocity: {np.max(np.abs(qd_qui)):.3f} rad/s")
    print(f"   Max acceleration: {np.max(np.abs(qdd_qui)):.3f} rad/s²")
    print(f"   ✓ Continuous position, velocity, and acceleration")
    
    # Plot comparison if requested
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            fig.suptitle('Joint Space Trajectory Comparison (Joint 0)')
            
            # Position
            axes[0].plot(t_lin, q_lin[:, 0], 'b--', label='Linear', linewidth=2)
            axes[0].plot(t_cub, q_cub[:, 0], 'g-', label='Cubic', linewidth=2)
            axes[0].plot(t_qui, q_qui[:, 0], 'r-', label='Quintic', linewidth=2)
            axes[0].set_ylabel('Position (rad)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Velocity
            axes[1].plot(t_lin, qd_lin[:, 0], 'b--', label='Linear', linewidth=2)
            axes[1].plot(t_cub, qd_cub[:, 0], 'g-', label='Cubic', linewidth=2)
            axes[1].plot(t_qui, qd_qui[:, 0], 'r-', label='Quintic', linewidth=2)
            axes[1].set_ylabel('Velocity (rad/s)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Acceleration
            axes[2].plot(t_lin, qdd_lin[:, 0], 'b--', label='Linear', linewidth=2)
            axes[2].plot(t_cub, qdd_cub[:, 0], 'g-', label='Cubic', linewidth=2)
            axes[2].plot(t_qui, qdd_qui[:, 0], 'r-', label='Quintic', linewidth=2)
            axes[2].set_ylabel('Acceleration (rad/s²)')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n⚠️  matplotlib not installed. Skipping plots.")
    
    return t_qui, q_qui, qd_qui, qdd_qui


def demo_multi_waypoint(model, waypoints, plot=False):
    """Demonstrate multi-waypoint trajectory."""
    print("\n" + "="*70)
    print("MULTI-WAYPOINT TRAJECTORY")
    print("="*70)
    
    print(f"\nWaypoints: {len(waypoints)} points")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i}: {wp}")
    
    # Generate trajectory through waypoints
    print("\nUsing quintic polynomial between waypoints...")
    t_multi, q_multi, qd_multi, qdd_multi = multi_waypoint_trajectory(
        waypoints,
        durations=0.5,  # 0.5s per segment
        num_points_per_segment=50,
        method='quintic'
    )
    
    print(f"\nTotal duration: {t_multi[-1]:.3f}s")
    print(f"Total points: {len(t_multi)}")
    print(f"Max velocity: {np.max(np.abs(qd_multi)):.3f} rad/s")
    print(f"Max acceleration: {np.max(np.abs(qdd_multi)):.3f} rad/s²")
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            fig.suptitle('Multi-Waypoint Trajectory (All Joints)')
            
            for j in range(waypoints.shape[1]):
                axes[0].plot(t_multi, q_multi[:, j], label=f'Joint {j}', alpha=0.7)
                axes[1].plot(t_multi, qd_multi[:, j], label=f'Joint {j}', alpha=0.7)
                axes[2].plot(t_multi, qdd_multi[:, j], label=f'Joint {j}', alpha=0.7)
            
            axes[0].set_ylabel('Position (rad)')
            axes[0].legend(ncol=3)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_ylabel('Velocity (rad/s)')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].set_ylabel('Acceleration (rad/s²)')
            axes[2].set_xlabel('Time (s)')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass
    
    return t_multi, q_multi, qd_multi, qdd_multi


def demo_velocity_profiles(plot=False):
    """Demonstrate velocity profiles."""
    print("\n" + "="*70)
    print("VELOCITY PROFILES")
    print("="*70)
    
    distance = 1.0
    v_max = 0.5
    a_max = 1.0
    j_max = 5.0
    
    # 1. Trapezoidal profile
    print("\n1. Trapezoidal Velocity Profile")
    t_trap, s_trap, v_trap, a_trap = trapezoidal_velocity_profile(
        distance, v_max, a_max
    )
    print(f"   Duration: {t_trap[-1]:.3f}s")
    print(f"   Max velocity: {np.max(v_trap):.3f} m/s")
    print(f"   Max acceleration: {np.max(np.abs(a_trap)):.3f} m/s²")
    
    # 2. S-curve profile
    print("\n2. S-Curve (Jerk-Limited) Velocity Profile")
    t_scurve, s_scurve, v_scurve, a_scurve, j_scurve = s_curve_velocity_profile(
        distance, v_max, a_max, j_max
    )
    print(f"   Duration: {t_scurve[-1]:.3f}s")
    print(f"   Max velocity: {np.max(v_scurve):.3f} m/s")
    print(f"   Max acceleration: {np.max(np.abs(a_scurve)):.3f} m/s²")
    print(f"   Max jerk: {np.max(np.abs(j_scurve)):.3f} m/s³")
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Velocity Profile Comparison')
            
            # Trapezoidal - Position
            axes[0, 0].plot(t_trap, s_trap, 'b-', linewidth=2)
            axes[0, 0].set_title('Trapezoidal Profile')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Trapezoidal - Velocity
            axes[1, 0].plot(t_trap, v_trap, 'g-', linewidth=2)
            axes[1, 0].set_ylabel('Velocity (m/s)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # S-curve - Position
            axes[0, 1].plot(t_scurve, s_scurve, 'b-', linewidth=2)
            axes[0, 1].set_title('S-Curve Profile')
            axes[0, 1].set_ylabel('Position (m)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # S-curve - Velocity & Acceleration
            ax_v = axes[1, 1]
            ax_a = ax_v.twinx()
            
            l1 = ax_v.plot(t_scurve, v_scurve, 'g-', linewidth=2, label='Velocity')
            l2 = ax_a.plot(t_scurve, a_scurve, 'r--', linewidth=2, label='Acceleration')
            
            ax_v.set_ylabel('Velocity (m/s)', color='g')
            ax_v.set_xlabel('Time (s)')
            ax_a.set_ylabel('Acceleration (m/s²)', color='r')
            
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax_v.legend(lines, labels, loc='upper right')
            
            ax_v.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass


def demo_cartesian_trajectory(model, end_link, q_init, plot=False):
    """Demonstrate Cartesian space trajectory."""
    print("\n" + "="*70)
    print("CARTESIAN SPACE TRAJECTORY")
    print("="*70)
    
    # Define start and end poses
    # Start pose: current configuration
    T_start = forward_kinematics(model, q_init, backend='numpy', return_end=True)
    
    # End pose: translate +0.1m in X, rotate 45° around Z
    T_end = T_start.copy()
    T_end[0, 3] += 0.1  # +0.1m in X
    
    # Add rotation
    from scipy.spatial.transform import Rotation
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_delta = Rotation.from_euler('z', 45, degrees=True)
    R_end = R_delta * R_start
    T_end[:3, :3] = R_end.as_matrix()
    
    print(f"\nStart position: {T_start[:3, 3]}")
    print(f"End position:   {T_end[:3, 3]}")
    
    # Generate linear Cartesian trajectory
    print("\nGenerating linear Cartesian trajectory...")
    try:
        t_cart, poses_cart, q_cart = linear_cartesian_trajectory(
            model,
            T_start,
            T_end,
            duration=2.0,
            num_points=50,
            q_init=q_init,
            ik_backend='numpy',
            ik_method='dls'
        )
        
        print(f"Duration: {t_cart[-1]:.3f}s")
        print(f"Points: {len(t_cart)}")
        print(f"IK success rate: {np.sum(~np.isnan(q_cart[:, 0])) / len(q_cart) * 100:.1f}%")
        
        # Verify linearity in Cartesian space
        positions = poses_cart[:, :3, 3]
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"Position step size: {np.mean(distances):.4f} ± {np.std(distances):.4f} m")
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(12, 5))
                
                # 3D Cartesian path
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
                ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                           c='g', s=100, marker='o', label='Start')
                ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                           c='r', s=100, marker='s', label='End')
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_zlabel('Z (m)')
                ax1.set_title('End-Effector Path in Cartesian Space')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Joint space trajectory
                ax2 = fig.add_subplot(122)
                for j in range(q_cart.shape[1]):
                    ax2.plot(t_cart, q_cart[:, j], label=f'Joint {j}', alpha=0.7)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Joint Angle (rad)')
                ax2.set_title('Joint Space Trajectory')
                ax2.legend(ncol=2)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            except ImportError:
                pass
        
    except Exception as e:
        print(f"⚠️  Cartesian trajectory failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Trajectory Planning Demo')
    parser.add_argument('--robot', type=str, default='bessica',
                       choices=['alicia', 'bessica'],
                       help='Robot model')
    parser.add_argument('--arm', type=str, default='left',
                       choices=['left', 'right'],
                       help='Arm to use (for bessica)')
    parser.add_argument('--plot', action='store_true',
                       help='Show matplotlib plots')
    args = parser.parse_args()
    
    # Load robot model
    print("Loading robot model...")
    if args.robot == 'alicia':
        urdf_path = os.path.join(Path(__file__).parent.parent,
                                 '../robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf')
        end_link = 'tool0'
        dof = 6
    else:  # bessica
        urdf_path = os.path.join(Path(__file__).parent.parent,
                                 '../robocore/assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf')
        end_link = f'{args.arm}_arm_gripper_{args.arm}_finger'
        dof = 7
    
    model = RobotModel(str(urdf_path))
    print(f"✓ Loaded {args.robot} ({dof}-DOF)")
    print(f"  End link: {end_link}")
    
    # Define test configurations
    q_start = np.zeros(dof)
    q_end = np.array([0.5, 0.3, -0.2, 0.4, -0.3, 0.2] + ([0.0] if dof == 7 else []))
    
    # Demo 1: Joint space trajectories
    demo_joint_space_trajectories(model, q_start, q_end, plot=args.plot)
    
    # Demo 2: Multi-waypoint
    waypoints = np.array([
        q_start,
        q_start + 0.2,
        q_start + 0.4,
        q_end,
    ])
    demo_multi_waypoint(model, waypoints, plot=args.plot)
    
    # Demo 3: Velocity profiles
    demo_velocity_profiles(plot=args.plot)
    
    # Demo 4: Cartesian trajectory
    demo_cartesian_trajectory(model, end_link, q_start, plot=args.plot)
    
    print("\n" + "="*70)
    print("✓ Trajectory Planning Demo Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
