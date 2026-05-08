#!/usr/bin/env python3
"""Cartesian Space Trajectory Planning Examples

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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
    draw_axis,
)
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import euler_to_matrix
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_euler
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def demo_linear_position():
    """Example: Linear Position Trajectory Planning"""
    beauty_print("[1] Linear Position Trajectory Planning", type="module", centered=False)

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

    beauty_print(f"Trajectory generated:")
    print(f"  Start position: {beauty_print_array(p_start)}")
    print(f"  End position: {beauty_print_array(p_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.linalg.norm(result['velocities'], axis=1)):.3f} m/s")
    beauty_print("  ✓ Straight-line motion")

    return result


def demo_slerp():
    """Example: SLERP Orientation Trajectory Planning"""
    beauty_print("[2] SLERP Orientation Trajectory Planning", type="module", centered=False)

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

    beauty_print(f"Trajectory generated:")
    print(f"  Start: Identity rotation")
    print(f"  End: 90° rotation around Z")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['angular_velocities'], axis=1)):.3f} rad/s")
    beauty_print("  ✓ Spherical linear interpolation (shortest path)")

    return result


def demo_circular_arc():
    """Example: Circular Arc Trajectory Planning"""
    beauty_print("[3] Circular Arc Trajectory Planning", type="module", centered=False)

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

    beauty_print(f"Trajectory generated:")
    print(f"  Start position: {beauty_print_array(p_start)}")
    print(f"  Via position: {beauty_print_array(p_via)}")
    print(f"  End position: {beauty_print_array(p_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(result['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['velocities'][:, 3:], axis=1)):.3f} rad/s")
    beauty_print("  ✓ Circular arc path with SLERP orientation")

    return result


def demo_spline_curve():
    """Example: Spline Curve Trajectory Planning"""
    beauty_print("[4] Spline Curve Trajectory Planning", type="module", centered=False)

    planner = SplineCurvePlanner()

    # Define multiple waypoints as transformation matrices
    # Positions
    p1 = np.array([0.3, 0.2, 0.1])
    p2 = np.array([0.2, 0.25, 0.15])
    p3 = np.array([0.2, 0.4, 0.2])
    p4 = np.array([0.45, -0.1, 0.25])
    p5 = np.array([0.5, 0.4, 0.3])

    # Orientations (rotation matrices)
    R1 = np.eye(3)  # Identity
    R2 = euler_to_matrix(np.pi, 0, np.pi/6, seq='xyz')  # 30° around Z
    R3 = euler_to_matrix(0, np.pi/8, 0, seq='xyz')  # 22.5° around Y
    R4 = euler_to_matrix(np.pi/6, 0, 0, seq='xyz')  # 30° around X
    R5 = euler_to_matrix(0, 0, np.pi/4, seq='xyz')  # 45° around Z

    # Create transformation matrices
    waypoints = np.array([
        make_transform(R1, p1),
        make_transform(R2, p2),
        make_transform(R3, p3),
        make_transform(R4, p4),
        make_transform(R5, p5)
    ])

    result = planner.plan(
        waypoints=waypoints,
        duration=3.0,
        num_points=150
    )

    # Extract positions from waypoints for plotting
    waypoint_positions = np.array([T[:3, 3] for T in waypoints])
    result['waypoints'] = waypoint_positions

    beauty_print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(result['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['velocities'][:, 3:], axis=1)):.3f} rad/s")
    beauty_print("  ✓ Smooth spline curve through waypoints")

    return result


def plot_trajectories(results_dict, show_list=['spline']):
    """Plot Cartesian space trajectories"""
    try:
        fig = plt.figure(figsize=(15, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(131, projection='3d')

        colors = ['blue', 'green', 'red', 'orange']
        labels = ['Linear', 'SLERP', 'Circular Arc', 'Spline Curve']

        for idx, (name, result) in enumerate(results_dict.items()):
            if result is not None:
                if name not in show_list:
                    continue
                color = colors[idx % len(colors)]
                label = labels[idx] if idx < len(labels) else name

                # Plot position trajectory if available
                if 'positions' in result:
                    positions = result['positions']
                    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                             color=color, label=label, linewidth=2, alpha=0.8)

                    # Mark start and end
                    if idx == 0:  # Only for first trajectory
                        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                                    c='green', s=100, marker='o', label='Start')
                        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                                    c='red', s=100, marker='s', label='End')

                    # Plot waypoints for spline curve
                    if name == 'spline' and 'waypoints' in result:
                        waypoints = result['waypoints']
                        ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                                    c='purple', s=150, marker='*', label='Waypoints',
                                    edgecolors='black', linewidths=1, zorder=5)

                # Plot orientation axes (sampled every 10 points)
                sample_step = 5
                if 'orientations' in result:
                    orientations = result['orientations']

                    # Convert to numpy if needed
                    if hasattr(orientations, 'cpu'):
                        orientations = orientations.cpu().numpy()

                    # Convert quaternions to rotation matrices if needed (for SLERP)
                    if len(orientations.shape) == 2 and orientations.shape[1] == 4:
                        # Quaternions: (num_points, 4)
                        R_matrices = np.zeros((len(orientations), 3, 3))
                        for i, q in enumerate(orientations):
                            R_matrices[i] = quaternion_to_matrix(q)
                        orientations = R_matrices
                    elif len(orientations.shape) == 3 and orientations.shape[1:] == (3, 3):
                        # Already rotation matrices: (num_points, 3, 3)
                        pass
                    else:
                        # Skip if shape is unexpected
                        continue

                    # Determine positions for drawing axes
                    if 'positions' in result:
                        # Use actual positions
                        axis_positions = result['positions']
                        if hasattr(axis_positions, 'cpu'):
                            axis_positions = axis_positions.cpu().numpy()
                    else:
                        # For SLERP (no positions), use a fixed position
                        # Place at origin or a visible location
                        axis_positions = np.zeros((len(orientations), 3))
                        # Offset slightly for visibility
                        axis_positions[:, 0] = 0.2 + idx * 0.1

                    # Sample and draw axes
                    for i in range(0, len(orientations), sample_step):
                        pos = axis_positions[i]
                        R = orientations[i]
                        # Use slightly smaller scale for sampled axes
                        draw_axis(ax1, pos, R, scale=0.03, alpha=0.6)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Cartesian Space Trajectories')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Velocity plot
        ax2 = fig.add_subplot(132)
        for idx, (name, result) in enumerate(results_dict.items()):
            if result is not None:
                if name not in show_list:
                    continue
                t = result['t']
                color = colors[idx % len(colors)]
                label = labels[idx] if idx < len(labels) else name

                if 'velocities' in result:
                    v = result['velocities']
                    v_norm = np.linalg.norm(v[:, :3], axis=1)  # Linear velocity
                    ax2.plot(t, v_norm, color=color, label=label, linewidth=2, alpha=0.8)
                elif 'angular_velocities' in result and name == 'slerp':
                    # For SLERP, plot angular velocity magnitude
                    angular_v = result['angular_velocities']
                    angular_v_norm = np.linalg.norm(angular_v, axis=1)
                    ax2.plot(t, angular_v_norm, color=color, label=label, linewidth=2, alpha=0.8, linestyle='--')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s or rad/s)')
        ax2.set_title('Velocity Profiles')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Position/Orientation components
        ax3 = fig.add_subplot(133)
        # Plot linear position if available
        if 'linear' in results_dict and results_dict['linear'] is not None:
            result = results_dict['linear']
            t = result['t']
            positions = result['positions']
            ax3.plot(t, positions[:, 0], 'r-', label='X', linewidth=2)
            ax3.plot(t, positions[:, 1], 'g-', label='Y', linewidth=2)
            ax3.plot(t, positions[:, 2], 'b-', label='Z', linewidth=2)
            ax3.set_ylabel('Position (m)')
            ax3.set_title('Position Components (Linear)')
        # Plot SLERP orientation (Euler angles) if available
        elif 'slerp' in results_dict and results_dict['slerp'] is not None:
            result = results_dict['slerp']
            t = result['t']
            orientations = result['orientations']  # Quaternions
            # Convert quaternions to Euler angles
            euler_angles = np.zeros((len(orientations), 3))
            for i, q in enumerate(orientations):
                R = quaternion_to_matrix(q)
                euler_angles[i] = matrix_to_euler(R, seq='xyz')
            ax3.plot(t, euler_angles[:, 0], 'r-', label='Roll (X)', linewidth=2)
            ax3.plot(t, euler_angles[:, 1], 'g-', label='Pitch (Y)', linewidth=2)
            ax3.plot(t, euler_angles[:, 2], 'b-', label='Yaw (Z)', linewidth=2)
            ax3.set_ylabel('Angle (rad)')
            ax3.set_title('Orientation Components (SLERP)')

        ax3.set_xlabel('Time (s)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping plots.")


if __name__ == '__main__':
    beauty_print("Cartesian Space Trajectory Planning Examples", type="module")

    results = {}

    # Run all examples
    results['linear'] = demo_linear_position()
    results['slerp'] = demo_slerp()
    results['circular'] = demo_circular_arc()
    results['spline'] = demo_spline_curve()

    # Plot trajectories
    plot_trajectories(results, show_list=['spline'])

    beauty_print("✓ All examples completed!", type="success")
