#!/usr/bin/env python3
"""Velocity Profile Generation Examples

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
    TrapezoidalVelocityProfile,
    SCurveVelocityProfile,
)
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def demo_trapezoidal():
    """Example: Trapezoidal Velocity Profile"""
    beauty_print("[1] Trapezoidal Velocity Profile", type="module", centered=False)

    planner = TrapezoidalVelocityProfile()

    # Method 1: Specify duration
    beauty_print("Method 1: Specify duration", type="info", centered=False)
    result1 = planner.plan(
        start=0.0,
        end=1.0,
        duration=2.0,
        num_points=100
    )

    beauty_print(f"  Duration: {result1['t'][-1]:.3f} s")
    print(f"  Distance: {result1['s'][-1] - result1['s'][0]:.3f} m")
    print(f"  Max velocity: {np.max(result1['v']):.3f} m/s")
    print(f"  Max acceleration: {np.max(np.abs(result1['a'])):.3f} m/s²")

    # Method 2: Specify v_max and a_max
    print("\nMethod 2: Specify v_max and a_max")
    result2 = planner.plan(
        start=0.0,
        end=1.0,
        num_points=100,
        v_max=0.5,
        a_max=1.0
    )

    print(f"  Duration: {result2['t'][-1]:.3f} s")
    print(f"  Distance: {result2['s'][-1] - result2['s'][0]:.3f} m")
    print(f"  Max velocity: {np.max(result2['v']):.3f} m/s")
    print(f"  Max acceleration: {np.max(np.abs(result2['a'])):.3f} m/s²")
    print(f"  ✓ Three phases: acceleration, constant velocity, deceleration")

    return result1, result2


def demo_s_curve():
    """Example: S-Curve Velocity Profile"""
    beauty_print("[2] S-Curve (Jerk-Limited) Velocity Profile", type="module", centered=False)

    planner = SCurveVelocityProfile()

    # Generate S-curve profile
    result = planner.plan(
        start=0.0,
        end=1.0,
        duration=2.0,
        num_points=100,
        v_max=0.5,
        a_max=1.0,
        j_max=5.0  # Maximum jerk
    )

    print(f"Trajectory generated:")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Distance: {result['s'][-1] - result['s'][0]:.3f} m")
    print(f"  Max velocity: {np.max(result['v']):.3f} m/s")
    print(f"  Max acceleration: {np.max(np.abs(result['a'])):.3f} m/s²")
    print(f"  Max jerk: {np.max(np.abs(result['j'])):.3f} m/s³")
    print(f"  ✓ Smooth acceleration (jerk-limited)")

    return result


def plot_velocity_profiles(trap_result1, trap_result2, scurve_result):
    """Plot velocity profile comparison"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Velocity Profile Comparison', fontsize=14)

        # Trapezoidal - Position
        axes[0, 0].plot(trap_result1['t'], trap_result1['s'], 'b-', linewidth=2, label='Trapezoidal (duration)')
        axes[0, 0].plot(trap_result2['t'], trap_result2['s'], 'b--', linewidth=2, label='Trapezoidal (v_max, a_max)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position Profile')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Trapezoidal - Velocity
        axes[0, 1].plot(trap_result1['t'], trap_result1['v'], 'g-', linewidth=2, label='Trapezoidal (duration)')
        axes[0, 1].plot(trap_result2['t'], trap_result2['v'], 'g--', linewidth=2, label='Trapezoidal (v_max, a_max)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Profile')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # S-Curve - Position and Velocity
        ax_v = axes[1, 0]
        ax_a = ax_v.twinx()

        l1 = ax_v.plot(scurve_result['t'], scurve_result['s'], 'r-', linewidth=2, label='Position')
        l2 = ax_a.plot(scurve_result['t'], scurve_result['v'], 'b--', linewidth=2, label='Velocity')

        ax_v.set_xlabel('Time (s)')
        ax_v.set_ylabel('Position (m)', color='r')
        ax_a.set_ylabel('Velocity (m/s)', color='b')
        ax_v.set_title('S-Curve: Position and Velocity')
        ax_v.tick_params(axis='y', labelcolor='r')
        ax_a.tick_params(axis='y', labelcolor='b')

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax_v.legend(lines, labels, loc='upper left')
        ax_v.grid(True, alpha=0.3)

        # S-Curve - Acceleration and Jerk
        ax_acc = axes[1, 1]
        ax_jerk = ax_acc.twinx()

        l3 = ax_acc.plot(scurve_result['t'], scurve_result['a'], 'g-', linewidth=2, label='Acceleration')
        l4 = ax_jerk.plot(scurve_result['t'], scurve_result['j'], 'm--', linewidth=2, label='Jerk')

        ax_acc.set_xlabel('Time (s)')
        ax_acc.set_ylabel('Acceleration (m/s²)', color='g')
        ax_jerk.set_ylabel('Jerk (m/s³)', color='m')
        ax_acc.set_title('S-Curve: Acceleration and Jerk')
        ax_acc.tick_params(axis='y', labelcolor='g')
        ax_jerk.tick_params(axis='y', labelcolor='m')

        lines = l3 + l4
        labels = [l.get_label() for l in lines]
        ax_acc.legend(lines, labels, loc='upper left')
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping plots.")


if __name__ == '__main__':
    beauty_print("Velocity Profile Generation Examples", type="module")

    trap_result1, trap_result2 = demo_trapezoidal()
    scurve_result = demo_s_curve()

    # Plot comparison
    plot_velocity_profiles(trap_result1, trap_result2, scurve_result)

    beauty_print("✓ All examples completed!", type="success")
