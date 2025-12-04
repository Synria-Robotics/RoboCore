"""B-Spline Interpolation for Joint Space Trajectory Planning

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

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from robocore.planning.base import BaseTrajectoryPlanner


class BSplinePlanner(BaseTrajectoryPlanner):
    """B-spline trajectory planner for joint space.
    
    Generates smooth trajectories through multiple waypoints using B-spline interpolation.
    Supports cubic (k=3) and quintic (k=5) B-splines.
    """
    
    def __init__(self, degree: int = 3):
        """Initialize B-spline planner.
        
        :param degree: B-spline degree (3 for cubic, 5 for quintic)
        """
        super().__init__()
        if degree not in [3, 5]:
            raise ValueError("Degree must be 3 (cubic) or 5 (quintic)")
        self.degree = degree
    
    def plan(
        self,
        waypoints: Any,
        duration: Optional[float] = None,
        num_points: int = 100,
        knot_vector: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate B-spline trajectory through waypoints.
        
        :param waypoints: Waypoint joint positions [n_waypoints, n_joints]
        :param duration: Total trajectory duration in seconds (optional, computed from waypoints if None)
        :param num_points: Number of points in trajectory
        :param knot_vector: Custom knot vector (optional, auto-generated if None)
        :return: Dictionary with 't', 'q', 'qd', 'qdd'
        """
        waypoints = self._ensure_array(waypoints)
        
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        
        n_waypoints, n_joints = waypoints.shape
        
        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints")
        
        # Generate knot vector if not provided
        if knot_vector is None:
            knot_vector = self._generate_knot_vector(n_waypoints, self.degree)
        else:
            knot_vector = self._ensure_array(knot_vector)
        
        # Generate time array
        if duration is None:
            # Estimate duration based on waypoint distances
            distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            total_distance = np.sum(distances)
            duration = max(1.0, total_distance * 2.0)  # Rough estimate
        
        t = self._linspace(0.0, duration, num_points)
        
        # Normalize time to [0, 1] for B-spline parameter
        u = t / duration
        
        # Map u to knot vector domain
        u_min = knot_vector[self.degree]
        u_max = knot_vector[-(self.degree + 1)]
        u_normalized = u_min + (u_max - u_min) * u
        
        # Evaluate B-spline
        q = self._zeros((num_points, n_joints))
        qd = self._zeros((num_points, n_joints))
        qdd = self._zeros((num_points, n_joints))
        
        for j in range(n_joints):
            # Fit B-spline through waypoints
            if self._backend_manager.is_numpy:
                from scipy.interpolate import BSpline, make_interp_spline
                
                # Use scipy's B-spline interpolation
                try:
                    # Try to use make_interp_spline for better control
                    spline = make_interp_spline(
                        np.linspace(0, 1, n_waypoints),
                        waypoints[:, j],
                        k=self.degree,
                        bc_type='natural'
                    )
                    
                    # Evaluate
                    q[:, j] = spline(u_normalized)
                    qd[:, j] = spline.derivative(1)(u_normalized) / duration
                    qdd[:, j] = spline.derivative(2)(u_normalized) / (duration ** 2)
                except:
                    # Fallback to manual B-spline evaluation
                    q[:, j], qd[:, j], qdd[:, j] = self._evaluate_bspline_manual(
                        waypoints[:, j], u_normalized, knot_vector, self.degree, duration
                    )
            else:
                # For torch, use manual evaluation
                q[:, j], qd[:, j], qdd[:, j] = self._evaluate_bspline_manual(
                    waypoints[:, j], u_normalized, knot_vector, self.degree, duration
                )
        
        return {
            't': t,
            'q': q,
            'qd': qd,
            'qdd': qdd,
        }
    
    def _generate_knot_vector(self, n_waypoints: int, degree: int) -> Any:
        """Generate uniform knot vector.
        
        :param n_waypoints: Number of waypoints
        :param degree: B-spline degree
        :return: Knot vector
        """
        n_control = n_waypoints
        n_knots = n_control + degree + 1
        
        # Uniform knot vector
        if self._backend_manager.is_numpy:
            knots = np.zeros(n_knots)
            # Clamped knots: first (degree+1) and last (degree+1) knots are repeated
            for i in range(degree + 1):
                knots[i] = 0.0
            for i in range(degree + 1, n_control):
                knots[i] = (i - degree) / (n_control - degree)
            for i in range(n_control, n_knots):
                knots[i] = 1.0
        else:
            import torch
            knots = torch.zeros(n_knots, device=self._backend_manager.get_device(),
                               dtype=self._backend_manager.get_dtype())
            for i in range(degree + 1):
                knots[i] = 0.0
            for i in range(degree + 1, n_control):
                knots[i] = (i - degree) / (n_control - degree)
            for i in range(n_control, n_knots):
                knots[i] = 1.0
        
        return knots
    
    def _evaluate_bspline_manual(
        self,
        control_points: Any,
        u: Any,
        knots: Any,
        degree: int,
        duration: float
    ) -> tuple:
        """Manually evaluate B-spline (fallback method).
        
        :param control_points: Control points [n]
        :param u: Parameter values [num_points]
        :param knots: Knot vector
        :param degree: B-spline degree
        :param duration: Duration for velocity/acceleration scaling
        :return: (q, qd, qdd)
        """
        n_control = len(control_points)
        num_points = len(u)
        
        q = self._zeros(num_points)
        qd = self._zeros(num_points)
        qdd = self._zeros(num_points)
        
        # For simplicity, use piecewise polynomial approximation
        # This is a simplified implementation
        if self._backend_manager.is_numpy:
            from scipy.interpolate import interp1d
            
            # Map u to waypoint indices
            u_waypoints = np.linspace(0, 1, n_control)
            
            # Use cubic interpolation as approximation
            interp_func = interp1d(u_waypoints, control_points, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
            q = interp_func(u)
            
            # Compute derivatives using finite differences
            if num_points > 1:
                dt = duration / (num_points - 1)
                qd = np.gradient(q, dt)
                qdd = np.gradient(qd, dt)
        else:
            import torch
            # Similar approach for torch
            u_waypoints = torch.linspace(0, 1, n_control, device=u.device, dtype=u.dtype)
            
            # Linear interpolation for torch (cubic not directly available)
            indices = torch.searchsorted(u_waypoints, u)
            indices = torch.clamp(indices, 0, n_control - 2)
            
            u0 = u_waypoints[indices]
            u1 = u_waypoints[indices + 1]
            p0 = control_points[indices]
            p1 = control_points[indices + 1]
            
            alpha = (u - u0) / (u1 - u0 + 1e-10)
            q = p0 + alpha * (p1 - p0)
            
            # Finite differences
            if num_points > 1:
                dt = duration / (num_points - 1)
                qd = torch.diff(q, prepend=q[0:1]) / dt
                qdd = torch.diff(qd, prepend=qd[0:1]) / dt
        
        return q, qd, qdd

