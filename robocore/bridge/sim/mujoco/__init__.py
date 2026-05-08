"""MuJoCo physics simulation modules.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from .physics_simulator import PhysicsSimulator
from .trajectory_executor import TrajectoryExecutor
from .trajectory_evaluator import TrajectoryEvaluator
from .comparison_analyzer import ComparisonAnalyzer
from .utils import (
    interpolate_trajectory,
    interpolate_trajectory_with_derivatives,
    compute_jerk,
    fft_analysis,
    detect_vibration,
    compute_rms_error,
    compute_max_error
)

__all__ = [
    'PhysicsSimulator',
    'TrajectoryExecutor',
    'TrajectoryEvaluator',
    'ComparisonAnalyzer',
    'interpolate_trajectory',
    'interpolate_trajectory_with_derivatives',
    'compute_jerk',
    'fft_analysis',
    'detect_vibration',
    'compute_rms_error',
    'compute_max_error',
]

