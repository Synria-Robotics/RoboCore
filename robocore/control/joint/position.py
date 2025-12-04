"""Joint Position Controller (PD/PID)

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


class JointPositionController(BaseController):
    """Joint space position controller with PD/PID feedback.
    
    Control law:
    - PD: τ = Kp·(qd - q) + Kd·(q̇d - q̇)
    - PID: τ = Kp·(qd - q) + Ki·∫(qd - q)dt + Kd·(q̇d - q̇)
    
    :param Kp: Position gain matrix [n×n] or vector [n×1]
    :param Kd: Velocity gain matrix [n×n] or vector [n×1]
    :param Ki: Integral gain matrix [n×n] or vector [n×1], optional
    :param use_integral: Enable integral term (PID mode), default False
    :param anti_windup: Enable integral anti-windup, default True
    :param integral_limit: Maximum integral value for anti-windup
    :param dt: Time step for integral computation, default 0.001
    """
    
    def __init__(
        self,
        Kp: Union[np.ndarray, list, float],
        Kd: Union[np.ndarray, list, float],
        Ki: Optional[Union[np.ndarray, list, float]] = None,
        use_integral: bool = False,
        anti_windup: bool = True,
        integral_limit: Optional[float] = None,
        dt: float = 0.001,
    ):
        """Initialize joint position controller.
        
        :param Kp: Position gain
        :param Kd: Velocity gain
        :param Ki: Integral gain (required if use_integral=True)
        :param use_integral: Enable PID mode
        :param anti_windup: Enable anti-windup
        :param integral_limit: Integral saturation limit
        :param dt: Time step
        """
        super().__init__()
        
        # Convert gains to arrays
        self.Kp = self._ensure_array(Kp)
        self.Kd = self._ensure_array(Kd)
        
        # Handle integral term
        self.use_integral = use_integral
        if use_integral:
            if Ki is None:
                raise ValueError("Ki must be provided when use_integral=True")
            self.Ki = self._ensure_array(Ki)
        else:
            self.Ki = None
        
        self.anti_windup = anti_windup
        self.integral_limit = integral_limit
        self.dt = dt
        
        # Ensure gains are matrices (handle scalar/vector inputs)
        self._normalize_gains()
        
        # State variables
        self.integral_error = None
        self.last_error = None
        self._num_joints = None
    
    def _normalize_gains(self):
        """Normalize gain matrices to proper shape."""
        # Get number of joints from first gain
        if self.Kp.ndim == 0:
            # Scalar gain - will be expanded later
            self._num_joints = None
        elif self.Kp.ndim == 1:
            # Vector gain
            self._num_joints = len(self.Kp)
            # Convert to diagonal matrix
            if self._backend_manager.is_numpy:
                self.Kp = np.diag(self.Kp)
                self.Kd = np.diag(self.Kd)
                if self.Ki is not None:
                    self.Ki = np.diag(self.Ki)
            else:
                import torch
                self.Kp = torch.diag(self.Kp)
                self.Kd = torch.diag(self.Kd)
                if self.Ki is not None:
                    self.Ki = torch.diag(self.Ki)
        elif self.Kp.ndim == 2:
            # Matrix gain
            self._num_joints = self.Kp.shape[0]
        else:
            raise ValueError("Gain must be scalar, vector, or matrix")
    
    def _reset_state(self):
        """Reset controller state."""
        if self._num_joints is not None:
            self.integral_error = self._zeros(self._num_joints)
        else:
            self.integral_error = None
        self.last_error = None
    
    def compute(
        self,
        q: Any,
        qd: Any,
        qd_desired: Any,
        qdd_desired: Optional[Any] = None,
    ) -> Any:
        """Compute control torque.
        
        :param q: Current joint positions [n×1]
        :param qd: Current joint velocities [n×1]
        :param qd_desired: Desired joint positions [n×1]
        :param qdd_desired: Desired joint velocities [n×1], optional
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        q = self._ensure_array(q)
        qd = self._ensure_array(qd)
        qd_desired = self._ensure_array(qd_desired)
        
        # Infer number of joints from input
        if self._num_joints is None:
            self._num_joints = len(q)
            self._normalize_gains()
            self._reset_state()
        
        # Compute errors
        error = qd_desired - q
        if qdd_desired is not None:
            qdd_desired = self._ensure_array(qdd_desired)
            error_dot = qdd_desired - qd
        else:
            error_dot = -qd  # Assume desired velocity is zero
        
        # PD term
        tau = self.Kp @ error + self.Kd @ error_dot
        
        # Integral term (PID)
        if self.use_integral and self.Ki is not None:
            # Update integral
            self.integral_error = self.integral_error + error * self.dt
            
            # Anti-windup
            if self.anti_windup:
                if self.integral_limit is not None:
                    # Clip integral
                    if self._backend_manager.is_numpy:
                        self.integral_error = np.clip(
                            self.integral_error,
                            -self.integral_limit,
                            self.integral_limit
                        )
                    else:
                        self.integral_error = self.integral_error.clamp(
                            -self.integral_limit,
                            self.integral_limit
                        )
            
            # Add integral term
            tau = tau + self.Ki @ self.integral_error
        
        return tau
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._reset_state()

