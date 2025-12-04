"""Cartesian Velocity Controller

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


class CartesianVelocityController(BaseController):
    """Cartesian space velocity controller.
    
    Control law: τ = J^T·Kp·(ẋd - ẋ) + g(q)
    
    where:
    - ẋ: current end-effector velocity [6×1] (linear 3 + angular 3)
    - ẋd: desired end-effector velocity [6×1]
    - J: Jacobian matrix [6×n]
    - g(q): gravity vector [n×1]
    
    :param robot_model: RobotModel instance
    :param Kp: Velocity gain matrix [6×6] or diagonal vector [6×1]
    :param use_gravity_compensation: Enable gravity compensation, default True
    :param end_link: End-effector link name, default None (uses default end link)
    """
    
    def __init__(
        self,
        robot_model: Any,
        Kp: Union[np.ndarray, list, float],
        use_gravity_compensation: bool = True,
        end_link: Optional[str] = None,
    ):
        """Initialize cartesian velocity controller.
        
        :param robot_model: RobotModel instance
        :param Kp: Velocity gain
        :param use_gravity_compensation: Enable gravity compensation
        :param end_link: End-effector link name
        """
        super().__init__()
        
        self.robot_model = robot_model
        self.use_gravity_compensation = use_gravity_compensation
        self.end_link = end_link
        
        # Convert gain to array
        self.Kp = self._ensure_array(Kp)
        
        # Normalize gain to 6×6 matrix
        self._normalize_gain()
    
    def _normalize_gain(self):
        """Normalize gain matrix to 6×6 shape."""
        if self.Kp.ndim == 0:
            # Scalar gain - create diagonal matrix
            if self._backend_manager.is_numpy:
                self.Kp = np.eye(6) * self.Kp
            else:
                import torch
                self.Kp = torch.eye(6, device=self.Kp.device, dtype=self.Kp.dtype) * self.Kp
        elif self.Kp.ndim == 1:
            # Vector gain - convert to diagonal matrix
            if len(self.Kp) != 6:
                raise ValueError("Gain vector must have length 6")
            if self._backend_manager.is_numpy:
                self.Kp = np.diag(self.Kp)
            else:
                import torch
                self.Kp = torch.diag(self.Kp)
        elif self.Kp.ndim == 2:
            # Matrix gain
            if self.Kp.shape != (6, 6):
                raise ValueError("Gain matrix must be 6×6")
        else:
            raise ValueError("Gain must be scalar, vector [6], or matrix [6×6]")
    
    def compute(
        self,
        q: Any,
        qd: Any,
        xdd_desired: Any,
    ) -> Any:
        """Compute control torque.
        
        :param q: Current joint positions [n×1]
        :param qd: Current joint velocities [n×1]
        :param xdd_desired: Desired end-effector velocity [6×1]
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        q = self._ensure_array(q)
        qd = self._ensure_array(qd)
        xdd_desired = self._ensure_array(xdd_desired)
        
        # Compute current end-effector velocity
        J = self.robot_model.jacobian(q, target_link=self.end_link)
        xd_current = J @ qd
        
        # Compute velocity error
        velocity_error = xdd_desired - xd_current
        
        # Control law in operational space
        F = self.Kp @ velocity_error
        
        # Map to joint space via Jacobian transpose
        tau = J.T @ F
        
        # Add gravity compensation
        if self.use_gravity_compensation:
            # TODO: Implement gravity compensation when dynamics module is available
            # For now, skip gravity compensation
            pass
        
        return tau

