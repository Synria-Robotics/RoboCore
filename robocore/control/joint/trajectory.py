"""Joint Trajectory Tracking Controller

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

from typing import Any, Optional, Union, Callable

import numpy as np

from robocore.control.base import BaseController


class JointTrajectoryController(BaseController):
    """Joint space trajectory tracking controller with feedforward and feedback.
    
    Control law (simplified, no dynamics):
    τ = Kp·(qd - q) + Kd·(q̇d - q̇) + Kff·q̈d
    
    Control law (full, with dynamics):
    τ = M·q̈d + C·q̇d + g + Kp·(qd - q) + Kd·(q̇d - q̇)
    
    :param Kp: Position gain matrix [n×n] or vector [n×1]
    :param Kd: Velocity gain matrix [n×n] or vector [n×1]
    :param Kff: Feedforward gain matrix [n×n] or vector [n×1], optional
    :param use_dynamics: Use full dynamics model (requires dynamics module), default False
    :param robot_model: RobotModel instance (required if use_dynamics=True)
    :param trajectory: Trajectory function or data
    """
    
    def __init__(
        self,
        Kp: Union[np.ndarray, list, float],
        Kd: Union[np.ndarray, list, float],
        Kff: Optional[Union[np.ndarray, list, float]] = None,
        use_dynamics: bool = False,
        robot_model: Optional[Any] = None,
    ):
        """Initialize joint trajectory tracking controller.
        
        :param Kp: Position gain
        :param Kd: Velocity gain
        :param Kff: Feedforward gain (default: same as Kp)
        :param use_dynamics: Use full dynamics model
        :param robot_model: RobotModel instance (required if use_dynamics=True)
        """
        super().__init__()
        
        self.use_dynamics = use_dynamics
        self.robot_model = robot_model
        
        if use_dynamics and robot_model is None:
            raise ValueError("robot_model is required when use_dynamics=True")
        
        # Convert gains to arrays
        self.Kp = self._ensure_array(Kp)
        self.Kd = self._ensure_array(Kd)
        
        if Kff is not None:
            self.Kff = self._ensure_array(Kff)
        else:
            # Default: use Kp as feedforward gain
            self.Kff = self.Kp.copy()
        
        # Normalize gains
        self._normalize_gains()
        
        # Trajectory state
        self._trajectory_func: Optional[Callable] = None
        self._trajectory_data: Optional[dict] = None
        self._current_time: float = 0.0
    
    def _normalize_gains(self):
        """Normalize gain matrices to proper shape."""
        if self.Kp.ndim == 0:
            # Scalar gain - will be expanded later
            self._num_joints = None
        elif self.Kp.ndim == 1:
            # Vector gain - convert to diagonal matrix
            self._num_joints = len(self.Kp)
            if self._backend_manager.is_numpy:
                self.Kp = np.diag(self.Kp)
                self.Kd = np.diag(self.Kd)
                self.Kff = np.diag(self.Kff) if self.Kff.ndim == 1 else self.Kff
            else:
                import torch
                self.Kp = torch.diag(self.Kp)
                self.Kd = torch.diag(self.Kd)
                self.Kff = torch.diag(self.Kff) if self.Kff.ndim == 1 else self.Kff
        elif self.Kp.ndim == 2:
            # Matrix gain
            self._num_joints = self.Kp.shape[0]
        else:
            raise ValueError("Gain must be scalar, vector, or matrix")
    
    def set_trajectory(
        self,
        trajectory_func: Optional[Callable[[float], tuple]] = None,
        trajectory_data: Optional[dict] = None,
    ):
        """Set trajectory for tracking.
        
        :param trajectory_func: Function that takes time t and returns (qd, qdd, qddd)
        :param trajectory_data: Dictionary with keys 't', 'q', 'qd', 'qdd' (arrays)
        """
        if trajectory_func is not None:
            self._trajectory_func = trajectory_func
            self._trajectory_data = None
        elif trajectory_data is not None:
            self._trajectory_data = trajectory_data
            self._trajectory_func = None
            # Validate trajectory data
            required_keys = ['t', 'q', 'qd', 'qdd']
            for key in required_keys:
                if key not in trajectory_data:
                    raise ValueError(f"trajectory_data must contain '{key}'")
        else:
            raise ValueError("Either trajectory_func or trajectory_data must be provided")
    
    def _get_trajectory_state(self, t: float) -> tuple:
        """Get desired trajectory state at time t.
        
        :param t: Current time
        :return: (qd, qdd, qddd) - desired position, velocity, acceleration
        """
        if self._trajectory_func is not None:
            # Use trajectory function
            return self._trajectory_func(t)
        elif self._trajectory_data is not None:
            # Interpolate from trajectory data
            t_array = self._trajectory_data['t']
            q_array = self._trajectory_data['q']
            qd_array = self._trajectory_data['qd']
            qdd_array = self._trajectory_data['qdd']
            
            # Find closest time index
            if self._backend_manager.is_numpy:
                idx = np.argmin(np.abs(t_array - t))
                qd = q_array[idx]
                qdd = qd_array[idx]
                qddd = qdd_array[idx] if 'qdd' in self._trajectory_data else self._zeros(len(qd))
            else:
                import torch
                idx = torch.argmin(torch.abs(t_array - t))
                qd = q_array[idx]
                qdd = qd_array[idx]
                qddd = qdd_array[idx] if 'qdd' in self._trajectory_data else self._zeros(len(qd))
            
            return (qd, qdd, qddd)
        else:
            raise RuntimeError("Trajectory not set. Call set_trajectory() first.")
    
    def compute(
        self,
        q: Any,
        qd: Any,
        t: Optional[float] = None,
        qd_desired: Optional[Any] = None,
        qdd_desired: Optional[Any] = None,
        qddd_desired: Optional[Any] = None,
    ) -> Any:
        """Compute control torque.
        
        Can be called in two modes:
        1. Time-based: Provide t, controller uses internal trajectory
        2. Direct: Provide qd_desired, qdd_desired, qddd_desired directly
        
        :param q: Current joint positions [n×1]
        :param qd: Current joint velocities [n×1]
        :param t: Current time (for time-based trajectory tracking)
        :param qd_desired: Desired joint positions [n×1] (direct mode)
        :param qdd_desired: Desired joint velocities [n×1] (direct mode)
        :param qddd_desired: Desired joint accelerations [n×1] (direct mode)
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        q = self._ensure_array(q)
        qd = self._ensure_array(qd)
        
        # Infer number of joints from input
        if self._num_joints is None:
            self._num_joints = len(q)
            self._normalize_gains()
        
        # Get desired trajectory state
        if t is not None:
            # Time-based mode
            qd_desired, qdd_desired, qddd_desired = self._get_trajectory_state(t)
            qd_desired = self._ensure_array(qd_desired)
            qdd_desired = self._ensure_array(qdd_desired)
            qddd_desired = self._ensure_array(qddd_desired) if qddd_desired is not None else self._zeros(self._num_joints)
        else:
            # Direct mode
            if qd_desired is None or qdd_desired is None:
                raise ValueError("Either t or (qd_desired, qdd_desired) must be provided")
            qd_desired = self._ensure_array(qd_desired)
            qdd_desired = self._ensure_array(qdd_desired)
            if qddd_desired is None:
                qddd_desired = self._zeros(self._num_joints)
            else:
                qddd_desired = self._ensure_array(qddd_desired)
        
        # Ensure gains are matrices (if they were scalars, expand now)
        num_joints = len(q)
        Kp = self.Kp
        Kd = self.Kd
        Kff = self.Kff
        if Kp.ndim == 0:
            # Scalar gain - expand to diagonal matrix
            if self._backend_manager.is_numpy:
                Kp = np.eye(num_joints) * Kp
                Kd = np.eye(num_joints) * Kd
                Kff = np.eye(num_joints) * Kff
            else:
                import torch
                Kp = torch.eye(num_joints, device=Kp.device, dtype=Kp.dtype) * Kp
                Kd = torch.eye(num_joints, device=Kd.device, dtype=Kd.dtype) * Kd
                Kff = torch.eye(num_joints, device=Kff.device, dtype=Kff.dtype) * Kff

        # Compute errors
        error = qd_desired - q
        error_dot = qdd_desired - qd
        
        # Feedback term
        tau_fb = Kp @ error + Kd @ error_dot
        
        # Feedforward term
        if self.use_dynamics:
            # Full dynamics feedforward (requires dynamics module)
            # TODO: Implement when dynamics module is available
            # tau_ff = M @ qddd_desired + C @ qdd_desired + g
            raise NotImplementedError("Full dynamics feedforward requires dynamics module (not yet implemented)")
        else:
            # Simple acceleration feedforward
            tau_ff = Kff @ qddd_desired
        
        # Total control torque
        tau = tau_fb + tau_ff
        
        return tau
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._current_time = 0.0

