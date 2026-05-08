"""Velocity Profile Generation

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from .trapezoidal import TrapezoidalVelocityProfile
from .s_curve import SCurveVelocityProfile

__all__ = [
    'TrapezoidalVelocityProfile',
    'SCurveVelocityProfile',
]

