#!/usr/bin/env python3
"""Joint Space Trajectory Planning Examples

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

from robocore.planning import (
    CubicPolynomialPlanner,
    QuinticPolynomialPlanner,
    SepticPolynomialPlanner,
    BSplinePlanner,
    MultiSegmentPlanner,
)
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def demo_cubic_polynomial():
    """Example: Cubic Polynomial Trajectory Planning"""
    beauty_print("[1] Cubic Polynomial Trajectory Planning", type="module", centered=False)

    planner = CubicPolynomialPlanner()

    # Define start and end joint positions
    q_start = np.array([0.0, 0.0, 0.0])
    q_end = np.array([1.0, 0.5, 0.8])

    # Optional: specify start and end velocities
    qd_start = np.array([0.0, 0.0, 0.0])
    qd_end = np.array([0.0, 0.0, 0.0])

    # Generate trajectory
    result = planner.plan(
        start=q_start,
        end=q_end,
        duration=2.0,
        num_points=100,
        qd_start=qd_start,
        qd_end=qd_end
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Start: {beauty_print_array(q_start)}")
    print(f"  End: {beauty_print_array(q_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")

    return result


def demo_quintic_polynomial():
    """Example: Quintic Polynomial Trajectory Planning"""
    beauty_print("[2] Quintic Polynomial Trajectory Planning", type="module", centered=False)

    planner = QuinticPolynomialPlanner()

    q_start = np.array([0.0, 0.0, 0.0])
    q_end = np.array([1.0, 0.5, 0.8])

    # Optional: specify velocities and accelerations
    qd_start = np.array([0.0, 0.0, 0.0])
    qd_end = np.array([0.0, 0.0, 0.0])
    qdd_start = np.array([0.0, 0.0, 0.0])
    qdd_end = np.array([0.0, 0.0, 0.0])

    result = planner.plan(
        start=q_start,
        end=q_end,
        duration=2.0,
        num_points=100,
        qd_start=qd_start,
        qd_end=qd_end,
        qdd_start=qdd_start,
        qdd_end=qdd_end
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Start: {beauty_print_array(q_start)}")
    print(f"  End: {beauty_print_array(q_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    beauty_print("  ✓ C2 continuous (position, velocity, acceleration)")

    return result


def demo_septic_polynomial():
    """Example: Septic Polynomial Trajectory Planning"""
    beauty_print("[3] Septic Polynomial Trajectory Planning", type="module", centered=False)

    planner = SepticPolynomialPlanner()

    q_start = np.array([0.0, 0.0, 0.0])
    q_end = np.array([1.0, 0.5, 0.8])

    # Specify all boundary conditions
    qd_start = np.array([0.0, 0.0, 0.0])
    qd_end = np.array([0.0, 0.0, 0.0])
    qdd_start = np.array([0.0, 0.0, 0.0])
    qdd_end = np.array([0.0, 0.0, 0.0])
    qddd_start = np.array([0.0, 0.0, 0.0])  # Jerk
    qddd_end = np.array([0.0, 0.0, 0.0])

    result = planner.plan(
        start=q_start,
        end=q_end,
        duration=2.0,
        num_points=100,
        qd_start=qd_start,
        qd_end=qd_end,
        qdd_start=qdd_start,
        qdd_end=qdd_end,
        qddd_start=qddd_start,
        qddd_end=qddd_end
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Start: {beauty_print_array(q_start)}")
    print(f"  End: {beauty_print_array(q_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    print(f"  Max jerk: {np.max(np.abs(result['qddd'])):.3f} rad/s³")
    beauty_print("  ✓ C3 continuous (position, velocity, acceleration, jerk)")

    return result


def demo_b_spline():
    """Example: B-Spline Trajectory Planning"""
    beauty_print("[4] B-Spline Trajectory Planning", type="module", centered=False)

    # Create planner with cubic B-spline (degree=3)
    planner = BSplinePlanner(degree=3)

    # Define multiple waypoints
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.1],
        [0.6, 0.4, 0.3],
        [1.2, 0.5, 0.6],
        [1.0, 0.5, 0.8]
    ])

    result = planner.plan(
        waypoints=waypoints,
        duration=3.0,
        num_points=150
    )

    # Add waypoints to result for plotting
    result['waypoints'] = waypoints

    beauty_print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    beauty_print("  ✓ Smooth curve through waypoints")

    return result


def demo_multi_segment():
    """Example: Multi-Segment Trajectory Planning"""
    beauty_print("[5] Multi-Segment Trajectory Planning", type="module", centered=False)

    # Create planner with quintic polynomial segments
    planner = MultiSegmentPlanner(method='quintic')

    # Define waypoints
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.2, 0.1],
        [0.6, 0.4, 0.3],
        [1.0, 0.5, 0.8]
    ])

    # Duration per segment (can be scalar or array)
    durations = 1.0  # Same duration for all segments
    # Or: durations = np.array([1.0, 1.5, 1.0])  # Different durations

    result = planner.plan(
        waypoints=waypoints,
        durations=durations,
        num_points_per_segment=50
    )

    # Add waypoints to result for plotting
    result['waypoints'] = waypoints

    beauty_print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Segments: {len(waypoints) - 1}")
    print(f"  Total duration: {result['t'][-1]:.3f} s")
    print(f"  Total points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    beauty_print("  ✓ Continuous segments with quintic polynomials")

    return result


def plot_comparison(results_dict):
    """Plot comparison of different planning methods"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Joint Space Trajectory Planning Comparison (Joint 0)', fontsize=14)

        colors = ['blue', 'green', 'red', 'orange', 'purple']
        labels = ['Cubic', 'Quintic', 'Septic', 'B-Spline', 'Multi-Segment']

        for idx, (name, result) in enumerate(results_dict.items()):
            if result is not None:
                t = result['t']
                q = result['q']
                qd = result['qd']
                qdd = result['qdd']

                color = colors[idx % len(colors)]
                label = labels[idx] if idx < len(labels) else name

                # Position
                axes[0].plot(t, q[:, 0], color=color, label=label, linewidth=2, alpha=0.8)

                # Plot waypoints if available
                if 'waypoints' in result:
                    waypoints = result['waypoints']
                    # For B-spline and multi-segment, display waypoints
                    # Distribute waypoints across time span
                    waypoint_times = np.linspace(t[0], t[-1], len(waypoints))
                    axes[0].scatter(waypoint_times, waypoints[:, 0],
                                    c='purple', s=150, marker='*',
                                    edgecolors='black', linewidths=1, zorder=5,
                                    label='Waypoints' if idx == 0 else '')

                # Velocity
                axes[1].plot(t, qd[:, 0], color=color, label=label, linewidth=2, alpha=0.8)

                # Acceleration
                axes[2].plot(t, qdd[:, 0], color=color, label=label, linewidth=2, alpha=0.8)

        axes[0].set_ylabel('Position (rad)', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ImportError:
        beauty_print("matplotlib not installed. Skipping plots.", type="warning")


if __name__ == '__main__':
    beauty_print("Joint Space Trajectory Planning Examples", type="module")

    results = {}

    # Run all examples
    results['cubic'] = demo_cubic_polynomial()
    results['quintic'] = demo_quintic_polynomial()
    results['septic'] = demo_septic_polynomial()
    results['b_spline'] = demo_b_spline()
    results['multi_segment'] = demo_multi_segment()

    # Plot comparison
    plot_comparison(results)

    beauty_print("✓ All examples completed!", type="success")
