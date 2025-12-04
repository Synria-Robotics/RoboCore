"""Trajectory Planning Module

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
