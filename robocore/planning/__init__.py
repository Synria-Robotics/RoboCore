"""
RoboCore Planning Module
========================

This module provides trajectory planning and motion planning capabilities:

Trajectory Planning:
- Joint space trajectory planning (polynomial, spline, velocity profiles)
- Cartesian space trajectory planning (linear, circular, spline)
- Time-optimal trajectory planning

Motion Planning:
- Sampling-based planning (RRT, RRT*)
- Optimization-based planning (CHOMP, TrajOpt)

Author: RoboCore Team
Date: 2025-10-03
"""

from robocore.planning.trajectory.joint_space import (
    cubic_polynomial_trajectory,
    quintic_polynomial_trajectory,
    linear_joint_trajectory
)

from robocore.planning.trajectory.cartesian_space import (
    linear_cartesian_trajectory,
    circular_cartesian_trajectory
)

from robocore.planning.trajectory.velocity_profile import (
    trapezoidal_velocity_profile,
    s_curve_velocity_profile
)

__all__ = [
    # Joint space
    'cubic_polynomial_trajectory',
    'quintic_polynomial_trajectory',
    'linear_joint_trajectory',
    
    # Cartesian space
    'linear_cartesian_trajectory',
    'circular_cartesian_trajectory',
    
    # Velocity profiles
    'trapezoidal_velocity_profile',
    's_curve_velocity_profile',
]
