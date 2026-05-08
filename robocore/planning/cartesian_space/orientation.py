"""Orientation Trajectory Planning in Cartesian Space

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict

from robocore.planning.base import BaseTrajectoryPlanner
from robocore.transform.conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from robocore.transform.utils import slerp


class SLERPPlanner(BaseTrajectoryPlanner):
    """Spherical Linear Interpolation (SLERP) planner for orientation.
    
    Generates smooth orientation trajectories using quaternion SLERP.
    """
    
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SLERP orientation trajectory.
        
        :param start: Start orientation - quaternion [4], rotation matrix [3, 3], or transformation [4, 4]
        :param end: End orientation - quaternion [4], rotation matrix [3, 3], or transformation [4, 4]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :return: Dictionary with 't', 'orientations' (quaternions), 'angular_velocities', 'angular_accelerations'
        """
        start = self._ensure_array(start)
        end = self._ensure_array(end)
        
        # Convert to quaternions
        q_start = self._extract_quaternion(start)
        q_end = self._extract_quaternion(end)
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        
        # Normalized time [0, 1]
        s = t / duration
        
        # SLERP interpolation
        if self._backend_manager.is_numpy:
            # Handle batch SLERP
            q_traj = self._zeros((num_points, 4))
            for i in range(num_points):
                q_traj[i] = slerp(q_start, q_end, s[i])
        else:
            # For torch, slerp should handle batch
            q_traj = slerp(
                q_start.unsqueeze(0).repeat(num_points, 1),
                q_end.unsqueeze(0).repeat(num_points, 1),
                s
            )
        
        # Compute angular velocities (simplified - using finite differences)
        # For more accurate computation, we would need to compute quaternion derivatives
        angular_velocities = self._zeros((num_points, 3))
        angular_accelerations = self._zeros((num_points, 3))
        
        if num_points > 1:
            dt = duration / (num_points - 1) if duration > 0 else 1.0
            for i in range(num_points - 1):
                # Convert quaternion difference to angular velocity
                q1 = q_traj[i]
                q2 = q_traj[i + 1]
                
                # Relative rotation: q_rel = q2 * q1^-1
                from robocore.transform.conversions import (
                    quaternion_multiply,
                    quaternion_conjugate,
                )
                q_rel = quaternion_multiply(q2, quaternion_conjugate(q1))
                
                # Convert to axis-angle
                from robocore.transform.conversions import quaternion_to_axis_angle
                axis, angle = quaternion_to_axis_angle(q_rel)
                
                # Angular velocity = axis * angle / dt
                angular_velocities[i] = axis * angle / dt
            
            # Set last velocity same as second to last
            if num_points > 1:
                angular_velocities[-1] = angular_velocities[-2]
            
            # Compute accelerations
            if num_points > 1:
                for i in range(num_points - 1):
                    angular_accelerations[i] = (angular_velocities[i + 1] - angular_velocities[i]) / dt
                angular_accelerations[-1] = angular_accelerations[-2]
        
        return {
            't': t,
            'orientations': q_traj,
            'angular_velocities': angular_velocities,
            'angular_accelerations': angular_accelerations,
        }
    
    def _extract_quaternion(self, orientation: Any) -> Any:
        """Extract quaternion from various orientation representations.
        
        :param orientation: Quaternion [4], rotation matrix [3, 3], or transformation [4, 4]
        :return: Quaternion [4]
        """
        orientation = self._ensure_array(orientation)
        
        if orientation.shape == (4, 4):
            # Transformation matrix - extract rotation part
            R = orientation[:3, :3]
            return matrix_to_quaternion(R)
        elif orientation.shape == (3, 3):
            # Rotation matrix
            return matrix_to_quaternion(orientation)
        elif len(orientation) == 4:
            # Already a quaternion
            return orientation
        else:
            raise ValueError(
                "Orientation must be quaternion [4], rotation matrix [3, 3], "
                "or transformation matrix [4, 4]"
            )

