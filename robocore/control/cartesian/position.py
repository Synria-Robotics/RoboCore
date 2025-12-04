"""Cartesian Position Controller

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

from typing import Any, Optional, Union

import numpy as np

from robocore.control.base import BaseController
from robocore.transform.conversions import (
    quaternion_to_rotation_matrix,
    matrix_to_quaternion,
)
from robocore.transform.so3 import rotation_matrix_to_axis_angle


class CartesianPositionController(BaseController):
    """Cartesian space position controller.
    
    Control law: τ = J^T·[Kp·(xd - x) + Kd·(ẋd - ẋ)] + g(q)
    
    where:
    - x: current end-effector pose [7×1] (position 3 + quaternion 4)
    - xd: desired end-effector pose [7×1]
    - ẋ: current end-effector velocity [6×1] (linear 3 + angular 3)
    - ẋd: desired end-effector velocity [6×1]
    - J: Jacobian matrix [6×n]
    - g(q): gravity vector [n×1]
    
    :param robot_model: RobotModel instance
    :param Kp: Position gain matrix [6×6] or diagonal vector [6×1]
    :param Kd: Velocity gain matrix [6×6] or diagonal vector [6×1]
    :param use_gravity_compensation: Enable gravity compensation, default True
    :param end_link: End-effector link name, default None (uses default end link)
    """
    
    def __init__(
        self,
        robot_model: Any,
        Kp: Union[np.ndarray, list, float],
        Kd: Union[np.ndarray, list, float],
        use_gravity_compensation: bool = True,
        end_link: Optional[str] = None,
    ):
        """Initialize cartesian position controller.
        
        :param robot_model: RobotModel instance
        :param Kp: Position gain
        :param Kd: Velocity gain
        :param use_gravity_compensation: Enable gravity compensation
        :param end_link: End-effector link name
        """
        super().__init__()
        
        self.robot_model = robot_model
        self.use_gravity_compensation = use_gravity_compensation
        self.end_link = end_link
        
        # Convert gains to arrays
        self.Kp = self._ensure_array(Kp)
        self.Kd = self._ensure_array(Kd)
        
        # Normalize gains to 6×6 matrices
        self._normalize_gains()
    
    def _normalize_gains(self):
        """Normalize gain matrices to 6×6 shape."""
        if self.Kp.ndim == 0:
            # Scalar gain - create diagonal matrix
            if self._backend_manager.is_numpy:
                self.Kp = np.eye(6) * self.Kp
                self.Kd = np.eye(6) * self.Kd
            else:
                import torch
                self.Kp = torch.eye(6, device=self.Kp.device, dtype=self.Kp.dtype) * self.Kp
                self.Kd = torch.eye(6, device=self.Kd.device, dtype=self.Kd.dtype) * self.Kd
        elif self.Kp.ndim == 1:
            # Vector gain - convert to diagonal matrix
            if len(self.Kp) != 6:
                raise ValueError("Gain vector must have length 6")
            if self._backend_manager.is_numpy:
                self.Kp = np.diag(self.Kp)
                self.Kd = np.diag(self.Kd)
            else:
                import torch
                self.Kp = torch.diag(self.Kp)
                self.Kd = torch.diag(self.Kd)
        elif self.Kp.ndim == 2:
            # Matrix gain
            if self.Kp.shape != (6, 6):
                raise ValueError("Gain matrix must be 6×6")
        else:
            raise ValueError("Gain must be scalar, vector [6], or matrix [6×6]")
    
    def _compute_pose_error(
        self,
        x_current: Any,
        xd_desired: Any,
    ) -> Any:
        """Compute pose error (position + orientation).
        
        :param x_current: Current pose [7×1] (position 3 + quaternion 4)
        :param xd_desired: Desired pose [7×1] (position 3 + quaternion 4)
        :return: Pose error [6×1] (position error 3 + orientation error 3)
        """
        # Position error
        pos_error = xd_desired[:3] - x_current[:3]
        
        # Orientation error (quaternion to axis-angle)
        q_current = x_current[3:]
        q_desired = xd_desired[3:]
        
        # Convert quaternions to rotation matrices
        R_current = quaternion_to_rotation_matrix(q_current)
        R_desired = quaternion_to_rotation_matrix(q_desired)
        
        # Compute relative rotation
        R_error = R_desired @ R_current.T
        
        # Convert to axis-angle representation
        axis_angle = rotation_matrix_to_axis_angle(R_error)
        
        # Combine errors
        if self._backend_manager.is_numpy:
            error = np.concatenate([pos_error, axis_angle])
        else:
            import torch
            error = torch.cat([pos_error, axis_angle])
        
        return error
    
    def compute(
        self,
        q: Any,
        qd: Any,
        xd_desired: Any,
        xdd_desired: Optional[Any] = None,
    ) -> Any:
        """Compute control torque.
        
        :param q: Current joint positions [n×1]
        :param qd: Current joint velocities [n×1]
        :param xd_desired: Desired end-effector pose [7×1] (position 3 + quaternion 4)
        :param xdd_desired: Desired end-effector velocity [6×1], optional
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        q = self._ensure_array(q)
        qd = self._ensure_array(qd)
        xd_desired = self._ensure_array(xd_desired)
        
        # Compute current end-effector pose
        fk_result = self.robot_model.fk(q, return_end=True)
        if isinstance(fk_result, dict):
            # Extract pose from dict
            if self.end_link:
                T_current = fk_result.get(self.end_link)
            else:
                T_current = fk_result.get('end', list(fk_result.values())[0])
        else:
            T_current = fk_result
        
        # Extract position and quaternion from 4×4 transformation matrix
        if T_current.shape == (4, 4):
            # 4×4 transformation matrix
            pos_current = T_current[:3, 3]
            R_current = T_current[:3, :3]
            # Convert rotation matrix to quaternion
            quat_current = matrix_to_quaternion(R_current)
        else:
            # Assume pose is already in [pos, quat] format
            pos_current = T_current[:3]
            quat_current = T_current[3:7] if len(T_current) >= 7 else self._array([1.0, 0.0, 0.0, 0.0])
        
        # Combine current pose
        if self._backend_manager.is_numpy:
            x_current_full = np.concatenate([pos_current, quat_current])
        else:
            import torch
            x_current_full = torch.cat([pos_current, quat_current])
        
        # Compute pose error
        pose_error = self._compute_pose_error(x_current_full, xd_desired)
        
        # Compute current end-effector velocity
        J = self.robot_model.jacobian(q, target_link=self.end_link)
        xd_current = J @ qd
        
        # Compute velocity error
        if xdd_desired is not None:
            xdd_desired = self._ensure_array(xdd_desired)
            velocity_error = xdd_desired - xd_current
        else:
            velocity_error = -xd_current  # Assume desired velocity is zero
        
        # Control law in operational space
        F = self.Kp @ pose_error + self.Kd @ velocity_error
        
        # Map to joint space via Jacobian transpose
        tau = J.T @ F
        
        # Add gravity compensation
        if self.use_gravity_compensation:
            # TODO: Implement gravity compensation when dynamics module is available
            # For now, skip gravity compensation
            pass
        
        return tau

