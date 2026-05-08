"""Joint Velocity Controller

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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
        num_joints = len(qd)
        if self._num_joints is None:
            self._num_joints = num_joints
            self._normalize_gain()
        
        # Ensure gain is matrix (if it was scalar, expand now)
        Kp = self.Kp
        if Kp.ndim == 0:
            # Scalar gain - expand to diagonal matrix
            if self._backend_manager.is_numpy:
                Kp = np.eye(num_joints) * Kp
            else:
                import torch
                Kp = torch.eye(num_joints, device=Kp.device, dtype=Kp.dtype) * Kp

        # Compute velocity error
        error = qdd_desired - qd
        
        # Control law
        tau = Kp @ error
        
        return tau

