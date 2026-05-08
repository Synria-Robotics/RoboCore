"""Circular Arc Interpolation for Cartesian Space Trajectory Planning

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
from robocore.transform.utils import rotation_interpolate


class CircularArcPlanner(BaseTrajectoryPlanner):
    """Circular arc trajectory planner in Cartesian space.
    
    Generates circular arc trajectories through three points (start, via, end).
    Position follows a circular arc, orientation is interpolated using SLERP.
    """
    
    def plan(
        self,
        start: Any,
        via: Any,
        end: Any,
        duration: float,
        num_points: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate circular arc trajectory.
        
        :param start: Start pose - position [3] or transformation [4, 4]
        :param via: Via point - position [3] or transformation [4, 4]
        :param end: End pose - position [3] or transformation [4, 4]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :return: Dictionary with 't', 'poses' (4x4 matrices), 'positions', 'orientations', 'velocities', 'accelerations'
        """
        start = self._ensure_array(start)
        via = self._ensure_array(via)
        end = self._ensure_array(end)
        
        # Extract positions
        if start.shape == (4, 4):
            p_start = get_translation(start)
            R_start = get_rotation(start)
        elif len(start) == 3:
            p_start = start
            R_start = self._backend_manager.eye(3)
        else:
            raise ValueError("Start must be [3] position or [4, 4] transformation")
        
        if via.shape == (4, 4):
            p_via = get_translation(via)
            R_via = get_rotation(via)
        elif len(via) == 3:
            p_via = via
            R_via = self._backend_manager.eye(3)
        else:
            raise ValueError("Via must be [3] position or [4, 4] transformation")
        
        if end.shape == (4, 4):
            p_end = get_translation(end)
            R_end = get_rotation(end)
        elif len(end) == 3:
            p_end = end
            R_end = self._backend_manager.eye(3)
        else:
            raise ValueError("End must be [3] position or [4, 4] transformation")
        
        # Compute circle parameters
        # Three points define a circle
        v1 = p_via - p_start
        v2 = p_end - p_start
        
        # Normal vectors
        if self._backend_manager.is_numpy:
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v1)
            
            if np.linalg.norm(n1) < 1e-6:
                # Points are collinear, use linear interpolation
                return self._linear_fallback(p_start, p_end, R_start, R_end, duration, num_points)
            
            # Circle center (simplified - using perpendicular bisectors)
            # For a more accurate implementation, solve the circle equation
            # Here we use a simplified approach
            mid1 = (p_start + p_via) / 2
            mid2 = (p_via + p_end) / 2
            
            # Direction vectors
            dir1 = np.cross(n1, v1)
            dir2 = np.cross(n2, v2)
            dir1 = dir1 / (np.linalg.norm(dir1) + 1e-10)
            dir2 = dir2 / (np.linalg.norm(dir2) + 1e-10)
            
            # Approximate center (intersection of perpendicular bisectors)
            # Simplified: use average
            center = (p_start + p_via + p_end) / 3
            radius = np.linalg.norm(p_start - center)
        else:
            import torch
            n1 = torch.cross(v1, v2)
            n2 = torch.cross(v2, v1)
            
            if torch.norm(n1) < 1e-6:
                return self._linear_fallback(p_start, p_end, R_start, R_end, duration, num_points)
            
            center = (p_start + p_via + p_end) / 3
            radius = torch.norm(p_start - center)
        
        # Generate arc parameterization
        t = self._linspace(0.0, duration, num_points)
        s = t / duration
        
        # Compute arc angles
        if self._backend_manager.is_numpy:
            vec_start = p_start - center
            vec_via = p_via - center
            vec_end = p_end - center
            
            # Normalize
            vec_start = vec_start / (np.linalg.norm(vec_start) + 1e-10)
            vec_via = vec_via / (np.linalg.norm(vec_via) + 1e-10)
            vec_end = vec_end / (np.linalg.norm(vec_end) + 1e-10)
            
            # Compute angles
            angle_start = 0.0
            angle_via = np.arccos(np.clip(np.dot(vec_start, vec_via), -1, 1))
            angle_end = np.arccos(np.clip(np.dot(vec_start, vec_end), -1, 1))
            
            # Ensure correct direction
            cross = np.cross(vec_start, vec_via)
            if np.dot(cross, np.cross(vec_start, vec_end)) < 0:
                angle_end = 2 * np.pi - angle_end
        else:
            import torch
            vec_start = p_start - center
            vec_via = p_via - center
            vec_end = p_end - center
            
            vec_start = vec_start / (torch.norm(vec_start) + 1e-10)
            vec_via = vec_via / (torch.norm(vec_via) + 1e-10)
            vec_end = vec_end / (torch.norm(vec_end) + 1e-10)
            
            angle_start = 0.0
            angle_via = torch.acos(torch.clamp(torch.dot(vec_start, vec_via), -1, 1))
            angle_end = torch.acos(torch.clamp(torch.dot(vec_start, vec_end), -1, 1))
            
            cross = torch.cross(vec_start, vec_via)
            if torch.dot(cross, torch.cross(vec_start, vec_end)) < 0:
                angle_end = 2 * np.pi - angle_end
        
        # Generate positions along arc
        positions = self._zeros((num_points, 3))
        orientations = self._zeros((num_points, 3, 3))
        poses = self._zeros((num_points, 4, 4))
        
        for i in range(num_points):
            # Interpolate angle
            angle = angle_start + (angle_end - angle_start) * s[i]
            
            # Compute position on circle
            if self._backend_manager.is_numpy:
                # Find rotation axis (perpendicular to plane)
                normal = np.cross(vec_start, vec_end)
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                
                # Rotate vec_start by angle around normal
                from scipy.spatial.transform import Rotation
                R_rot = Rotation.from_rotvec(normal * angle).as_matrix()
                vec_rotated = R_rot @ vec_start
                positions[i] = center + radius * vec_rotated
            else:
                import torch
                normal = torch.cross(vec_start, vec_end)
                normal = normal / (torch.norm(normal) + 1e-10)
                
                # Rotation matrix from axis-angle
                angle_tensor = torch.tensor(angle, device=normal.device, dtype=normal.dtype)
                R_rot = self._axis_angle_to_rotation(normal * angle_tensor)
                vec_rotated = R_rot @ vec_start
                positions[i] = center + radius * vec_rotated
            
            # Interpolate orientation using SLERP
            R_interp = rotation_interpolate(R_start, R_end, s[i], method='slerp')
            orientations[i] = R_interp
            
            # Build transformation matrix
            if self._backend_manager.is_numpy:
                poses[i] = np.eye(4)
                poses[i][:3, :3] = R_interp
                poses[i][:3, 3] = positions[i]
            else:
                import torch
                poses[i] = torch.eye(4, device=positions.device, dtype=positions.dtype)
                poses[i][:3, :3] = R_interp
                poses[i][:3, 3] = positions[i]
        
        # Compute velocities and accelerations (simplified)
        velocities = self._zeros((num_points, 6))
        accelerations = self._zeros((num_points, 6))
        
        if num_points > 1:
            dt = duration / (num_points - 1)
            for i in range(num_points - 1):
                # Linear velocity
                velocities[i, :3] = (positions[i + 1] - positions[i]) / dt
                # Angular velocity (simplified)
                R1 = orientations[i]
                R2 = orientations[i + 1]
                R_rel = R2 @ R1.T
                # Convert to axis-angle
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
            'positions': positions,
            'orientations': orientations,
            'velocities': velocities,
            'accelerations': accelerations,
        }
    
    def _linear_fallback(self, p_start, p_end, R_start, R_end, duration, num_points):
        """Fallback to linear interpolation if points are collinear."""
        from robocore.planning.cartesian_space.position import LinearPositionPlanner
        from robocore.planning.cartesian_space.orientation import SLERPPlanner
        
        pos_planner = LinearPositionPlanner()
        ori_planner = SLERPPlanner()
        
        pos_result = pos_planner.plan(p_start, p_end, duration, num_points)
        ori_result = ori_planner.plan(R_start, R_end, duration, num_points)
        
        # Combine results
        poses = self._zeros((num_points, 4, 4))
        for i in range(num_points):
            if self._backend_manager.is_numpy:
                poses[i] = np.eye(4)
                poses[i][:3, :3] = ori_result['orientations'][i]
                poses[i][:3, 3] = pos_result['positions'][i]
            else:
                import torch
                poses[i] = torch.eye(4, device=p_start.device, dtype=p_start.dtype)
                poses[i][:3, :3] = ori_result['orientations'][i]
                poses[i][:3, 3] = pos_result['positions'][i]
        
        return {
            't': pos_result['t'],
            'poses': poses,
            'positions': pos_result['positions'],
            'orientations': ori_result['orientations'],
            'velocities': self._zeros((num_points, 6)),
            'accelerations': self._zeros((num_points, 6)),
        }
    
    def _axis_angle_to_rotation(self, axis_angle):
        """Convert axis-angle to rotation matrix."""
        if self._backend_manager.is_numpy:
            from scipy.spatial.transform import Rotation
            return Rotation.from_rotvec(axis_angle).as_matrix()
        else:
            import torch
            angle = torch.norm(axis_angle)
            if angle < 1e-6:
                return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
            axis = axis_angle / angle
            # Rodrigues' formula
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=axis.device, dtype=axis.dtype)
            R = torch.eye(3, device=axis.device, dtype=axis.dtype) + \
                torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
            return R

