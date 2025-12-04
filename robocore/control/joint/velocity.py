"""Joint Velocity Controller

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


class JointVelocityController(BaseController):
    """Joint space velocity controller.
    
    Control law: τ = Kp·(q̇d - q̇)
    
    :param Kp: Velocity gain matrix [n×n] or vector [n×1]
    """
    
    def __init__(
        self,
        Kp: Union[np.ndarray, list, float],
    ):
        """Initialize joint velocity controller.
        
        :param Kp: Velocity gain
        """
        super().__init__()
        
        self.Kp = self._ensure_array(Kp)
        self._num_joints = None
        
        # Normalize gain
        self._normalize_gain()
    
    def _normalize_gain(self):
        """Normalize gain matrix to proper shape."""
        if self.Kp.ndim == 0:
            # Scalar gain
            self._num_joints = None
        elif self.Kp.ndim == 1:
            # Vector gain - convert to diagonal matrix
            self._num_joints = len(self.Kp)
            if self._backend_manager.is_numpy:
                self.Kp = np.diag(self.Kp)
            else:
                import torch
                self.Kp = torch.diag(self.Kp)
        elif self.Kp.ndim == 2:
            # Matrix gain
            self._num_joints = self.Kp.shape[0]
        else:
            raise ValueError("Gain must be scalar, vector, or matrix")
    
    def compute(
        self,
        q: Any,
        qd: Any,
        qdd_desired: Any,
    ) -> Any:
        """Compute control torque.
        
        :param q: Current joint positions [n×1] (not used, for interface consistency)
        :param qd: Current joint velocities [n×1]
        :param qdd_desired: Desired joint velocities [n×1]
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        qd = self._ensure_array(qd)
        qdd_desired = self._ensure_array(qdd_desired)
        
        # Infer number of joints from input
        if self._num_joints is None:
            self._num_joints = len(qd)
            self._normalize_gain()
        
        # Compute velocity error
        error = qdd_desired - qd
        
        # Control law
        tau = self.Kp @ error
        
        return tau

