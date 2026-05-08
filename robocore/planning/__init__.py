"""Trajectory Planning Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from .base import BaseTrajectoryPlanner

# Joint space planners
from .joint_space.polynomial import (
    CubicPolynomialPlanner,
    QuinticPolynomialPlanner,
    SepticPolynomialPlanner,
)
from .joint_space.spline import BSplinePlanner
from .joint_space.multi_segment import MultiSegmentPlanner

# Cartesian space planners
from .cartesian_space.position import LinearPositionPlanner
from .cartesian_space.orientation import SLERPPlanner
from .cartesian_space.circular import CircularArcPlanner
from .cartesian_space.spline import SplineCurvePlanner

# Velocity profiles
from .velocity_profile.trapezoidal import TrapezoidalVelocityProfile
from .velocity_profile.s_curve import SCurveVelocityProfile

# Utility functions
from .utils import (
    draw_axis,
    plot_cartesian_trajectory,
    plot_joint_trajectory,
    plot_cartesian_with_ik,
)

__all__ = [
    # Base
    'BaseTrajectoryPlanner',
    # Joint space
    'CubicPolynomialPlanner',
    'QuinticPolynomialPlanner',
    'SepticPolynomialPlanner',
    'BSplinePlanner',
    'MultiSegmentPlanner',
    # Cartesian space
    'LinearPositionPlanner',
    'SLERPPlanner',
    'CircularArcPlanner',
    'SplineCurvePlanner',
    # Velocity profiles
    'TrapezoidalVelocityProfile',
    'SCurveVelocityProfile',
    # Utilities
    'draw_axis',
    'plot_cartesian_trajectory',
    'plot_joint_trajectory',
    'plot_cartesian_with_ik',
]

