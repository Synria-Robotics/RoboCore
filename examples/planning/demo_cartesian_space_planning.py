#!/usr/bin/env python3
"""Cartesian Space Trajectory Planning Examples

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robocore.planning import (
    LinearPositionPlanner,
    SLERPPlanner,
    CircularArcPlanner,
    SplineCurvePlanner,
)
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import euler_to_matrix


def demo_linear_position():
    """Example: Linear Position Trajectory Planning"""
    print("\n" + "="*70)
    print("1. Linear Position Trajectory Planning")
    print("="*70)
    
    planner = LinearPositionPlanner()
    
    # Define start and end positions
    p_start = np.array([0.3, 0.2, 0.1])
    p_end = np.array([0.5, 0.4, 0.3])
    
    result = planner.plan(
        start=p_start,
        end=p_end,
        duration=2.0,
        num_points=100
    )
    
    print(f"Trajectory generated:")
    print(f"  Start position: {p_start}")
    print(f"  End position: {p_end}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.linalg.norm(result['velocities'], axis=1)):.3f} m/s")
    print(f"  ✓ Straight-line motion")
    
    return result


def demo_slerp():
    """Example: SLERP Orientation Trajectory Planning"""
    print("\n" + "="*70)
    print("2. SLERP Orientation Trajectory Planning")
    print("="*70)
    
    planner = SLERPPlanner()
    
    # Define start and end orientations (rotation matrices)
    # Start: identity rotation
    R_start = np.eye(3)
    
    # End: 90 degree rotation around Z axis
    R_end = euler_to_matrix(0, 0, np.pi/2, seq='xyz')
    
    result = planner.plan(
        start=R_start,
        end=R_end,
        duration=2.0,
        num_points=100
    )
    
    print(f"Trajectory generated:")
    print(f"  Start: Identity rotation")
    print(f"  End: 90° rotation around Z")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['angular_velocities'], axis=1)):.3f} rad/s")
    print(f"  ✓ Spherical linear interpolation (shortest path)")
    
    return result


def demo_circular_arc():
    """Example: Circular Arc Trajectory Planning"""
    print("\n" + "="*70)
    print("3. Circular Arc Trajectory Planning")
    print("="*70)
    
    planner = CircularArcPlanner()
    
    # Define three points: start, via, end
    p_start = np.array([0.3, 0.2, 0.1])
    p_via = np.array([0.4, 0.3, 0.25])  # Middle point
    p_end = np.array([0.5, 0.4, 0.3])
    
    # Define orientations
    R_start = np.eye(3)
    R_end = euler_to_matrix(0, 0, np.pi/4, seq='xyz')
    
    # Create transformation matrices
    T_start = make_transform(R_start, p_start)
    T_via = make_transform(R_start, p_via)  # Same orientation at via point
    T_end = make_transform(R_end, p_end)
    
    result = planner.plan(
        start=T_start,
        via=T_via,
        end=T_end,
        duration=2.0,
        num_points=100
    )
    
    print(f"Trajectory generated:")
    print(f"  Start position: {p_start}")
    print(f"  Via position: {p_via}")
    print(f"  End position: {p_end}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(result['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['velocities'][:, 3:], axis=1)):.3f} rad/s")
    print(f"  ✓ Circular arc path with SLERP orientation")
    
    return result


def demo_spline_curve():
    """Example: Spline Curve Trajectory Planning"""
    print("\n" + "="*70)
    print("4. Spline Curve Trajectory Planning")
    print("="*70)
    
    planner = SplineCurvePlanner()
    
    # Define multiple waypoints (positions)
    waypoints = np.array([
        [0.3, 0.2, 0.1],
        [0.35, 0.25, 0.15],
        [0.4, 0.3, 0.2],
        [0.45, 0.35, 0.25],
        [0.5, 0.4, 0.3]
    ])
    
    # Or use transformation matrices
    # waypoints = np.array([
    #     make_transform(R1, p1),
    #     make_transform(R2, p2),
    #     ...
    # ])
    
    result = planner.plan(
        waypoints=waypoints,
        duration=3.0,
        num_points=150
    )
    
    print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(result['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['velocities'][:, 3:], axis=1)):.3f} rad/s")
    print(f"  ✓ Smooth spline curve through waypoints")
    
    return result


def plot_trajectories(results_dict, plot=False):
    """Plot Cartesian space trajectories"""
    if not plot:
        return
    
    try:
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(131, projection='3d')
        
        colors = ['blue', 'green', 'red', 'orange']
        labels = ['Linear', 'SLERP', 'Circular Arc', 'Spline Curve']
        
        for idx, (name, result) in enumerate(results_dict.items()):
            if result is not None:
                if 'positions' in result:
                    positions = result['positions']
                    color = colors[idx % len(colors)]
                    label = labels[idx] if idx < len(labels) else name
                    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                            color=color, label=label, linewidth=2, alpha=0.8)
                    
                    # Mark start and end
                    if idx == 0:  # Only for first trajectory
                        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                                   c='green', s=100, marker='o', label='Start')
                        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                                   c='red', s=100, marker='s', label='End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Cartesian Space Trajectories')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Velocity plot
        ax2 = fig.add_subplot(132)
        for idx, (name, result) in enumerate(results_dict.items()):
            if result is not None and 'velocities' in result:
                t = result['t']
                v = result['velocities']
                v_norm = np.linalg.norm(v[:, :3], axis=1)  # Linear velocity
                color = colors[idx % len(colors)]
                label = labels[idx] if idx < len(labels) else name
                ax2.plot(t, v_norm, color=color, label=label, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Linear Velocity (m/s)')
        ax2.set_title('Linear Velocity Profiles')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Position components
        ax3 = fig.add_subplot(133)
        if 'linear' in results_dict and results_dict['linear'] is not None:
            result = results_dict['linear']
            t = result['t']
            positions = result['positions']
            ax3.plot(t, positions[:, 0], 'r-', label='X', linewidth=2)
            ax3.plot(t, positions[:, 1], 'g-', label='Y', linewidth=2)
            ax3.plot(t, positions[:, 2], 'b-', label='Z', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Position (m)')
            ax3.set_title('Position Components (Linear)')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping plots.")


def main():
    parser = argparse.ArgumentParser(description='Cartesian Space Trajectory Planning Examples')
    parser.add_argument('--plot', action='store_true', help='Show matplotlib plots')
    args = parser.parse_args()
    
    print("="*70)
    print("Cartesian Space Trajectory Planning Examples")
    print("="*70)
    
    results = {}
    
    # Run all examples
    try:
        results['linear'] = demo_linear_position()
        results['slerp'] = demo_slerp()
        results['circular'] = demo_circular_arc()
        results['spline'] = demo_spline_curve()
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Plot trajectories
    plot_trajectories(results, plot=args.plot)
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print("\nUsage tips:")
    print("  • Use --plot to visualize 3D trajectories")
    print("  • Linear planner: Simple straight-line motion")
    print("  • SLERP planner: Smooth orientation interpolation")
    print("  • Circular arc: Three points define a circle")
    print("  • Spline curve: Smooth path through multiple waypoints")


if __name__ == '__main__':
    main()

