"""Position Trajectory Planning in Cartesian Space

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict

from robocore.planning.base import BaseTrajectoryPlanner


class LinearPositionPlanner(BaseTrajectoryPlanner):
    """Linear position trajectory planner in Cartesian space.
    
    Generates straight-line trajectories for end-effector position.
    """
    
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate linear position trajectory.
        
        :param start: Start position [3] or transformation matrix [4, 4]
        :param end: End position [3] or transformation matrix [4, 4]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :return: Dictionary with 't', 'positions', 'velocities', 'accelerations'
        """
        start = self._ensure_array(start)
        end = self._ensure_array(end)
        
        # Extract positions from transformation matrices if needed
        if start.shape == (4, 4):
            start_pos = start[:3, 3]
        elif len(start) == 3:
            start_pos = start
        else:
            raise ValueError("Start must be [3] position or [4, 4] transformation matrix")
        
        if end.shape == (4, 4):
            end_pos = end[:3, 3]
        elif len(end) == 3:
            end_pos = end
        else:
            raise ValueError("End must be [3] position or [4, 4] transformation matrix")
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        
        # Normalized time [0, 1]
        s = t / duration
        
        # Linear interpolation: p(t) = p0 + (pf - p0) * s
        # Velocity: v(t) = (pf - p0) / T
        # Acceleration: a(t) = 0
        
        if self._backend_manager.is_numpy:
            positions = start_pos + (end_pos - start_pos) * s.reshape(-1, 1)
            velocity = (end_pos - start_pos) / duration
            velocities = velocity.reshape(1, 3).repeat(num_points, axis=0)
            accelerations = self._zeros((num_points, 3))
        else:
            import torch
            positions = start_pos + (end_pos - start_pos) * s.unsqueeze(-1)
            velocity = (end_pos - start_pos) / duration
            velocities = velocity.unsqueeze(0).repeat(num_points, 1)
            accelerations = self._zeros((num_points, 3))
        
        return {
            't': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
        }

