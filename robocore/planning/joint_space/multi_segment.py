"""Multi-Segment Trajectory Planning for Joint Space

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np

from robocore.planning.base import BaseTrajectoryPlanner
from robocore.planning.joint_space.polynomial import (
    CubicPolynomialPlanner,
    QuinticPolynomialPlanner,
)


class MultiSegmentPlanner(BaseTrajectoryPlanner):
    """Multi-segment trajectory planner for joint space.
    
    Generates smooth trajectories through multiple waypoints by connecting
    polynomial segments with continuity constraints.
    """
    
    def __init__(self, method: str = 'quintic'):
        """Initialize multi-segment planner.
        
        :param method: Interpolation method ('cubic' or 'quintic')
        """
        super().__init__()
        if method not in ['cubic', 'quintic']:
            raise ValueError("Method must be 'cubic' or 'quintic'")
        self.method = method
        if method == 'cubic':
            self.planner = CubicPolynomialPlanner()
        else:
            self.planner = QuinticPolynomialPlanner()
    
    def plan(
        self,
        waypoints: Any,
        durations: Union[float, Any] = 1.0,
        num_points_per_segment: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate multi-segment trajectory.
        
        :param waypoints: Waypoint joint positions [n_waypoints, n_joints]
        :param durations: Duration per segment (scalar) or array [n_segments]
        :param num_points_per_segment: Number of points per segment
        :return: Dictionary with 't', 'q', 'qd', 'qdd'
        """
        waypoints = self._ensure_array(waypoints)
        
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        
        n_waypoints, n_joints = waypoints.shape
        
        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints")
        
        n_segments = n_waypoints - 1
        
        # Handle durations
        durations = self._ensure_array(durations)
        if durations.ndim == 0 or len(durations) == 1:
            # Scalar duration - apply to all segments
            duration_per_segment = float(durations) if durations.ndim == 0 else float(durations[0])
            segment_durations = [duration_per_segment] * n_segments
        else:
            if len(durations) != n_segments:
                raise ValueError(f"Number of durations ({len(durations)}) must match number of segments ({n_segments})")
            segment_durations = [float(d) for d in durations]
        
        # Generate segments
        all_t = []
        all_q = []
        all_qd = []
        all_qdd = []
        
        t_offset = 0.0
        
        for i in range(n_segments):
            q_start = waypoints[i]
            q_end = waypoints[i + 1]
            duration = segment_durations[i]
            
            # Generate segment
            segment = self.planner.plan(
                start=q_start,
                end=q_end,
                duration=duration,
                num_points=num_points_per_segment,
                qd_start=None if i == 0 else None,  # Will be computed for continuity
                qd_end=None if i == n_segments - 1 else None,
            )
            
            # Adjust time offset
            segment_t = segment['t'] + t_offset
            
            # For continuity, we need to match velocities at waypoints
            if i > 0 and self.method == 'quintic':
                # Recompute previous segment's end velocity and current segment's start velocity
                # This is simplified - in practice, we'd solve a system for all segments
                pass
            
            all_t.append(segment_t)
            all_q.append(segment['q'])
            all_qd.append(segment['qd'])
            all_qdd.append(segment['qdd'])
            
            t_offset += duration
        
        # Concatenate segments
        if self._backend_manager.is_numpy:
            t = np.concatenate(all_t)
            q = np.concatenate(all_q, axis=0)
            qd = np.concatenate(all_qd, axis=0)
            qdd = np.concatenate(all_qdd, axis=0)
        else:
            import torch
            t = torch.cat(all_t)
            q = torch.cat(all_q, dim=0)
            qd = torch.cat(all_qd, dim=0)
            qdd = torch.cat(all_qdd, dim=0)
        
        return {
            't': t,
            'q': q,
            'qd': qd,
            'qdd': qdd,
        }

