#!/usr/bin/env python3
"""Joint Space Trajectory Planning Examples

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

from robocore.planning import (
    CubicPolynomialPlanner,
    QuinticPolynomialPlanner,
    SepticPolynomialPlanner,
    BSplinePlanner,
    MultiSegmentPlanner,
)


def demo_cubic_polynomial():
    """Example: Cubic Polynomial Trajectory Planning"""
    print("\n" + "="*70)
    print("1. Cubic Polynomial Trajectory Planning")
    print("="*70)
    
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
    
    print(f"Trajectory generated:")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    
    return result


def demo_quintic_polynomial():
    """Example: Quintic Polynomial Trajectory Planning"""
    print("\n" + "="*70)
    print("2. Quintic Polynomial Trajectory Planning")
    print("="*70)
    
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
    
    print(f"Trajectory generated:")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    print(f"  ✓ C2 continuous (position, velocity, acceleration)")
    
    return result


def demo_septic_polynomial():
    """Example: Septic Polynomial Trajectory Planning"""
    print("\n" + "="*70)
    print("3. Septic Polynomial Trajectory Planning")
    print("="*70)
    
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
    
    print(f"Trajectory generated:")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    print(f"  Max jerk: {np.max(np.abs(result['qddd'])):.3f} rad/s³")
    print(f"  ✓ C3 continuous (position, velocity, acceleration, jerk)")
    
    return result


def demo_b_spline():
    """Example: B-Spline Trajectory Planning"""
    print("\n" + "="*70)
    print("4. B-Spline Trajectory Planning")
    print("="*70)
    
    # Create planner with cubic B-spline (degree=3)
    planner = BSplinePlanner(degree=3)
    
    # Define multiple waypoints
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.2, 0.1],
        [0.6, 0.4, 0.3],
        [0.8, 0.5, 0.6],
        [1.0, 0.5, 0.8]
    ])
    
    result = planner.plan(
        waypoints=waypoints,
        duration=3.0,
        num_points=150
    )
    
    print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    print(f"  ✓ Smooth curve through waypoints")
    
    return result


def demo_multi_segment():
    """Example: Multi-Segment Trajectory Planning"""
    print("\n" + "="*70)
    print("5. Multi-Segment Trajectory Planning")
    print("="*70)
    
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
    
    print(f"Trajectory generated:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Segments: {len(waypoints) - 1}")
    print(f"  Total duration: {result['t'][-1]:.3f} s")
    print(f"  Total points: {len(result['t'])}")
    print(f"  Max velocity: {np.max(np.abs(result['qd'])):.3f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(result['qdd'])):.3f} rad/s²")
    print(f"  ✓ Continuous segments with quintic polynomials")
    
    return result


def plot_comparison(results_dict, plot=False):
    """Plot comparison of different planning methods"""
    if not plot:
        return
    
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
        print("\n⚠️  matplotlib not installed. Skipping plots.")


def main():
    parser = argparse.ArgumentParser(description='Joint Space Trajectory Planning Examples')
    parser.add_argument('--plot', action='store_true', help='Show matplotlib plots')
    args = parser.parse_args()
    
    print("="*70)
    print("Joint Space Trajectory Planning Examples")
    print("="*70)
    
    results = {}
    
    # Run all examples
    try:
        results['cubic'] = demo_cubic_polynomial()
        results['quintic'] = demo_quintic_polynomial()
        results['septic'] = demo_septic_polynomial()
        results['b_spline'] = demo_b_spline()
        results['multi_segment'] = demo_multi_segment()
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Plot comparison
    plot_comparison(results, plot=args.plot)
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print("\nUsage tips:")
    print("  • Use --plot to visualize trajectories")
    print("  • Adjust duration and num_points for different resolutions")
    print("  • Specify boundary conditions (velocities, accelerations) for smoother motion")


if __name__ == '__main__':
    main()

