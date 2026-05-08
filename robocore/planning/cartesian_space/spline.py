"""Spline Curve Interpolation for Cartesian Space Trajectory Planning

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from robocore.planning.base import BaseTrajectoryPlanner
from robocore.transform.se3 import get_translation, get_rotation, make_transform
from robocore.transform.utils import rotation_interpolate, slerp


class SplineCurvePlanner(BaseTrajectoryPlanner):
    """Spline curve trajectory planner in Cartesian space.
    
    Generates smooth spline curves through multiple waypoints.
    Position interpolation automatically selects method based on waypoint count:
    - 4+ waypoints: cubic spline (smooth C2 continuous)
    - 3 waypoints: quadratic spline (smooth C1 continuous)
    - 2 waypoints: linear interpolation
    Orientation uses SLERP for smooth rotation interpolation.
    """
    
    def plan(
        self,
        waypoints: Any,
        duration: Optional[float] = None,
        num_points: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate spline curve trajectory.
        
        :param waypoints: Waypoint poses - array of [3] positions or [4, 4] transformations [n_waypoints, ...]
        :param duration: Total trajectory duration in seconds (optional)
        :param num_points: Number of points in trajectory
        :return: Dictionary with 't', 'poses', 'positions', 'orientations', 'velocities', 'accelerations'
        """
        waypoints = self._ensure_array(waypoints)
        
        if waypoints.ndim == 2 and waypoints.shape[1] == 3:
            # Array of positions
            positions = waypoints
            n_waypoints = len(positions)
            # Default orientations (identity)
            if self._backend_manager.is_numpy:
                orientations = np.tile(np.eye(3), (n_waypoints, 1, 1))
            else:
                import torch
                orientations = torch.eye(3).unsqueeze(0).repeat(n_waypoints, 1, 1)
        elif waypoints.ndim == 3 and waypoints.shape[1:] == (4, 4):
            # Array of transformation matrices
            n_waypoints = len(waypoints)
            positions = self._zeros((n_waypoints, 3))
            orientations = self._zeros((n_waypoints, 3, 3))
            for i in range(n_waypoints):
                positions[i] = get_translation(waypoints[i])
                orientations[i] = get_rotation(waypoints[i])
        else:
            raise ValueError("Waypoints must be array of [3] positions or [4, 4] transformations")
        
        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints")
        
        # Estimate duration if not provided
        if duration is None:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
            duration = max(1.0, total_distance * 2.0)
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        
        # Interpolate positions using cubic spline
        positions_interp = self._zeros((num_points, 3))
        
        if self._backend_manager.is_numpy:
            from scipy.interpolate import interp1d
            
            # Parameterize by arc length
            u_waypoints = np.zeros(n_waypoints)
            for i in range(1, n_waypoints):
                u_waypoints[i] = u_waypoints[i - 1] + np.linalg.norm(positions[i] - positions[i - 1])
            u_waypoints = u_waypoints / u_waypoints[-1] if u_waypoints[-1] > 0 else np.linspace(0, 1, n_waypoints)
            
            # Choose interpolation kind based on number of waypoints
            # cubic requires at least 4 points, quadratic requires at least 3 points
            if n_waypoints >= 4:
                interp_kind = 'cubic'
            elif n_waypoints == 3:
                interp_kind = 'quadratic'
            else:  # n_waypoints == 2
                interp_kind = 'linear'

            # Interpolate each dimension
            u_interp = t / duration
            for dim in range(3):
                interp_func = interp1d(u_waypoints, positions[:, dim], kind=interp_kind,
                                      bounds_error=False, fill_value='extrapolate')
                positions_interp[:, dim] = interp_func(u_interp)
        else:
            import torch
            # For torch, use linear interpolation (cubic not directly available)
            u_waypoints = torch.zeros(n_waypoints, device=positions.device, dtype=positions.dtype)
            for i in range(1, n_waypoints):
                u_waypoints[i] = u_waypoints[i - 1] + torch.norm(positions[i] - positions[i - 1])
            u_waypoints = u_waypoints / u_waypoints[-1] if u_waypoints[-1] > 0 else torch.linspace(0, 1, n_waypoints, device=positions.device)
            
            u_interp = t / duration
            for dim in range(3):
                indices = torch.searchsorted(u_waypoints, u_interp)
                indices = torch.clamp(indices, 0, n_waypoints - 2)
                
                u0 = u_waypoints[indices]
                u1 = u_waypoints[indices + 1]
                p0 = positions[indices, dim]
                p1 = positions[indices + 1, dim]
                
                alpha = (u_interp - u0) / (u1 - u0 + 1e-10)
                positions_interp[:, dim] = p0 + alpha * (p1 - p0)
        
        # Interpolate orientations using SLERP between waypoints
        orientations_interp = self._zeros((num_points, 3, 3))
        
        # Find which segment each point belongs to
        u_interp = t / duration
        u_waypoints_normalized = np.linspace(0, 1, n_waypoints) if self._backend_manager.is_numpy else torch.linspace(0, 1, n_waypoints, device=t.device)
        
        for i in range(num_points):
            u = u_interp[i] if self._backend_manager.is_numpy else float(u_interp[i])
            
            # Find segment
            if u <= 0:
                orientations_interp[i] = orientations[0]
            elif u >= 1:
                orientations_interp[i] = orientations[-1]
            else:
                # Find which waypoint segment
                segment_idx = 0
                for j in range(n_waypoints - 1):
                    u_start = float(u_waypoints_normalized[j])
                    u_end = float(u_waypoints_normalized[j + 1])
                    if u_start <= u <= u_end:
                        segment_idx = j
                        break
                
                # Interpolate within segment
                u_start = float(u_waypoints_normalized[segment_idx])
                u_end = float(u_waypoints_normalized[segment_idx + 1])
                s = (u - u_start) / (u_end - u_start + 1e-10)
                
                R1 = orientations[segment_idx]
                R2 = orientations[segment_idx + 1]
                R_interp = rotation_interpolate(R1, R2, s, method='slerp')
                orientations_interp[i] = R_interp
        
        # Build transformation matrices
        poses = self._zeros((num_points, 4, 4))
        for i in range(num_points):
            poses[i] = make_transform(orientations_interp[i], positions_interp[i])
        
        # Compute velocities and accelerations
        velocities = self._zeros((num_points, 6))
        accelerations = self._zeros((num_points, 6))
        
        if num_points > 1:
            dt = duration / (num_points - 1)
            for i in range(num_points - 1):
                # Linear velocity
                velocities[i, :3] = (positions_interp[i + 1] - positions_interp[i]) / dt
                # Angular velocity
                R1 = orientations_interp[i]
                R2 = orientations_interp[i + 1]
                R_rel = R2 @ R1.T
                from robocore.transform.conversions import matrix_to_axis_angle
                axis, angle = matrix_to_axis_angle(R_rel)
                velocities[i, 3:] = axis * angle / dt
            
            velocities[-1] = velocities[-2]
            
            # Accelerations
            for i in range(num_points - 1):
                accelerations[i] = (velocities[i + 1] - velocities[i]) / dt
            accelerations[-1] = accelerations[-2]
        
        return {
            't': t,
            'poses': poses,
            'positions': positions_interp,
            'orientations': orientations_interp,
            'velocities': velocities,
            'accelerations': accelerations,
        }

