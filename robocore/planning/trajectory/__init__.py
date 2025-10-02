"""
Trajectory Planning Module
===========================

Provides trajectory generation in joint and Cartesian space with velocity profiles.
"""

from robocore.planning.trajectory.joint_space import (
    cubic_polynomial_trajectory,
    quintic_polynomial_trajectory,
    linear_joint_trajectory,
    multi_waypoint_trajectory
)

from robocore.planning.trajectory.cartesian_space import (
    linear_cartesian_trajectory,
    circular_cartesian_trajectory,
    cartesian_waypoint_trajectory
)

from robocore.planning.trajectory.velocity_profile import (
    trapezoidal_velocity_profile,
    s_curve_velocity_profile,
    constant_velocity_profile,
    scale_trajectory_to_profile
)

__all__ = [
    # Joint space trajectories
    'cubic_polynomial_trajectory',
    'quintic_polynomial_trajectory',
    'linear_joint_trajectory',
    'multi_waypoint_trajectory',
    # Cartesian space trajectories
    'linear_cartesian_trajectory',
    'circular_cartesian_trajectory',
    'cartesian_waypoint_trajectory',
    # Velocity profiles
    'trapezoidal_velocity_profile',
    's_curve_velocity_profile',
    'constant_velocity_profile',
    'scale_trajectory_to_profile',
]
