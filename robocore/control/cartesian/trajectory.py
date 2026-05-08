"""Cartesian Trajectory Tracking Controller

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Optional, Union, Callable

import numpy as np

from robocore.control.base import BaseController
from robocore.transform.conversions import (
    matrix_to_quaternion,
    matrix_to_axis_angle,
)
from robocore.transform.so3 import quaternion_to_matrix


class CartesianTrajectoryController(BaseController):
    """Cartesian space trajectory tracking controller with feedforward and feedback.
    
    Control law (simplified, no dynamics):
    τ = J^T·[Kp·(xd - x) + Kd·(ẋd - ẋ) + Kff·ẍd] + g(q)
    
    Control law (full, with dynamics):
    τ = J^T·Λ·[ẍd + Kp·(xd - x) + Kd·(ẋd - ẋ)] + μ + p
    
    where:
    - Λ = (J·M^-1·J^T)^-1: Operational space inertia matrix [6×6]
    - μ: Operational space Coriolis force [6×1]
    - p: Operational space gravity [6×1]
    
    :param robot_model: RobotModel instance
    :param Kp: Position gain matrix [6×6] or diagonal vector [6×1]
    :param Kd: Velocity gain matrix [6×6] or diagonal vector [6×1]
    :param Kff: Feedforward gain matrix [6×6] or diagonal vector [6×1], optional
    :param use_dynamics: Use full dynamics model (requires dynamics module), default False
    :param use_gravity_compensation: Enable gravity compensation, default True
    :param end_link: End-effector link name, default None
    """
    
    def __init__(
        self,
        robot_model: Any,
        Kp: Union[np.ndarray, list, float],
        Kd: Union[np.ndarray, list, float],
        Kff: Optional[Union[np.ndarray, list, float]] = None,
        use_dynamics: bool = False,
        use_gravity_compensation: bool = True,
        end_link: Optional[str] = None,
    ):
        """Initialize cartesian trajectory tracking controller.
        
        :param robot_model: RobotModel instance
        :param Kp: Position gain
        :param Kd: Velocity gain
        :param Kff: Feedforward gain (default: same as Kp)
        :param use_dynamics: Use full dynamics model
        :param use_gravity_compensation: Enable gravity compensation
        :param end_link: End-effector link name
        """
        super().__init__()
        
        self.robot_model = robot_model
        self.use_dynamics = use_dynamics
        self.use_gravity_compensation = use_gravity_compensation
        self.end_link = end_link
        
        if use_dynamics and not hasattr(robot_model, 'mass_matrix'):
            raise ValueError("robot_model must support dynamics when use_dynamics=True")
        
        # Convert gains to arrays
        self.Kp = self._ensure_array(Kp)
        self.Kd = self._ensure_array(Kd)
        
        if Kff is not None:
            self.Kff = self._ensure_array(Kff)
        else:
            # Default: use Kp as feedforward gain
            self.Kff = self.Kp.copy()
        
        # Normalize gains to 6×6 matrices
        self._normalize_gains()
        
        # Trajectory state
        self._trajectory_func: Optional[Callable] = None
        self._trajectory_data: Optional[dict] = None
        self._current_time: float = 0.0
    
    def _normalize_gains(self):
        """Normalize gain matrices to 6×6 shape."""
        for gain_name in ['Kp', 'Kd', 'Kff']:
            gain = getattr(self, gain_name)
            if gain.ndim == 0:
                # Scalar gain - create diagonal matrix
                if self._backend_manager.is_numpy:
                    setattr(self, gain_name, np.eye(6) * gain)
                else:
                    import torch
                    setattr(self, gain_name, torch.eye(6, device=gain.device, dtype=gain.dtype) * gain)
            elif gain.ndim == 1:
                # Vector gain - convert to diagonal matrix
                if len(gain) != 6:
                    raise ValueError(f"{gain_name} vector must have length 6")
                if self._backend_manager.is_numpy:
                    setattr(self, gain_name, np.diag(gain))
                else:
                    import torch
                    setattr(self, gain_name, torch.diag(gain))
            elif gain.ndim == 2:
                # Matrix gain
                if gain.shape != (6, 6):
                    raise ValueError(f"{gain_name} matrix must be 6×6")
            else:
                raise ValueError(f"{gain_name} must be scalar, vector [6], or matrix [6×6]")
    
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
        R_current = quaternion_to_matrix(q_current)
        R_desired = quaternion_to_matrix(q_desired)
        
        # Compute relative rotation
        R_error = R_desired @ R_current.T
        
        # Convert to axis-angle representation
        axis, angle = matrix_to_axis_angle(R_error)
        # Convert to axis-angle vector (axis * angle)
        axis_angle = axis * angle
        
        # Combine errors
        if self._backend_manager.is_numpy:
            error = np.concatenate([pos_error, axis_angle])
        else:
            import torch
            error = torch.cat([pos_error, axis_angle])
        
        return error
    
    def set_trajectory(
        self,
        trajectory_func: Optional[Callable[[float], tuple]] = None,
        trajectory_data: Optional[dict] = None,
    ):
        """Set trajectory for tracking.
        
        :param trajectory_func: Function that takes time t and returns (xd, xdd, xddd)
                                where xd is [7×1] pose and xdd/xddd are [6×1] velocities/accelerations
        :param trajectory_data: Dictionary with keys 't', 'poses', 'velocities', 'accelerations'
        """
        if trajectory_func is not None:
            self._trajectory_func = trajectory_func
            self._trajectory_data = None
        elif trajectory_data is not None:
            self._trajectory_data = trajectory_data
            self._trajectory_func = None
            # Validate trajectory data
            required_keys = ['t', 'poses']
            for key in required_keys:
                if key not in trajectory_data:
                    raise ValueError(f"trajectory_data must contain '{key}'")
        else:
            raise ValueError("Either trajectory_func or trajectory_data must be provided")
    
    def _get_trajectory_state(self, t: float) -> tuple:
        """Get desired trajectory state at time t.
        
        :param t: Current time
        :return: (xd, xdd, xddd) - desired pose [7×1], velocity [6×1], acceleration [6×1]
        """
        if self._trajectory_func is not None:
            # Use trajectory function
            return self._trajectory_func(t)
        elif self._trajectory_data is not None:
            # Interpolate from trajectory data
            t_array = self._trajectory_data['t']
            poses_array = self._trajectory_data['poses']
            
            # Find closest time index
            if self._backend_manager.is_numpy:
                idx = np.argmin(np.abs(t_array - t))
                pose = poses_array[idx]
                
                # Extract pose (handle 4×4 matrix or [pos, quat] format)
                if pose.shape == (4, 4):
                    pos = pose[:3, 3]
                    R = pose[:3, :3]
                    quat = matrix_to_quaternion(R)
                    xd = np.concatenate([pos, quat])
                else:
                    xd = pose
                
                # Get velocities and accelerations if available
                if 'velocities' in self._trajectory_data:
                    xdd = self._trajectory_data['velocities'][idx]
                else:
                    xdd = self._zeros(6)
                
                if 'accelerations' in self._trajectory_data:
                    xddd = self._trajectory_data['accelerations'][idx]
                else:
                    xddd = self._zeros(6)
            else:
                import torch
                idx = torch.argmin(torch.abs(t_array - t))
                pose = poses_array[idx]
                
                # Extract pose
                if pose.shape == (4, 4):
                    pos = pose[:3, 3]
                    R = pose[:3, :3]
                    quat = matrix_to_quaternion(R)
                    xd = torch.cat([pos, quat])
                else:
                    xd = pose
                
                # Get velocities and accelerations
                if 'velocities' in self._trajectory_data:
                    xdd = self._trajectory_data['velocities'][idx]
                else:
                    xdd = self._zeros(6)
                
                if 'accelerations' in self._trajectory_data:
                    xddd = self._trajectory_data['accelerations'][idx]
                else:
                    xddd = self._zeros(6)
            
            return (xd, xdd, xddd)
        else:
            raise RuntimeError("Trajectory not set. Call set_trajectory() first.")
    
    def compute(
        self,
        q: Any,
        qd: Any,
        t: Optional[float] = None,
        xd_desired: Optional[Any] = None,
        xdd_desired: Optional[Any] = None,
        xddd_desired: Optional[Any] = None,
    ) -> Any:
        """Compute control torque.
        
        Can be called in two modes:
        1. Time-based: Provide t, controller uses internal trajectory
        2. Direct: Provide xd_desired, xdd_desired, xddd_desired directly
        
        :param q: Current joint positions [n×1]
        :param qd: Current joint velocities [n×1]
        :param t: Current time (for time-based trajectory tracking)
        :param xd_desired: Desired end-effector pose [7×1] (direct mode)
        :param xdd_desired: Desired end-effector velocity [6×1] (direct mode)
        :param xddd_desired: Desired end-effector acceleration [6×1] (direct mode)
        :return: Control torque [n×1]
        """
        # Convert inputs to arrays
        q = self._ensure_array(q)
        qd = self._ensure_array(qd)
        
        # Get desired trajectory state
        if t is not None:
            # Time-based mode
            xd_desired, xdd_desired, xddd_desired = self._get_trajectory_state(t)
            xd_desired = self._ensure_array(xd_desired)
            xdd_desired = self._ensure_array(xdd_desired)
            xddd_desired = self._ensure_array(xddd_desired) if xddd_desired is not None else self._zeros(6)
        else:
            # Direct mode
            if xd_desired is None or xdd_desired is None:
                raise ValueError("Either t or (xd_desired, xdd_desired) must be provided")
            xd_desired = self._ensure_array(xd_desired)
            xdd_desired = self._ensure_array(xdd_desired)
            if xddd_desired is None:
                xddd_desired = self._zeros(6)
            else:
                xddd_desired = self._ensure_array(xddd_desired)
        
        # Compute current end-effector pose
        fk_result = self.robot_model.fk(q, return_end=True)
        if isinstance(fk_result, dict):
            if self.end_link:
                T_current = fk_result.get(self.end_link)
            else:
                T_current = fk_result.get('end', list(fk_result.values())[0])
        else:
            T_current = fk_result
        
        # Extract position and quaternion from 4×4 transformation matrix
        if T_current.shape == (4, 4):
            pos_current = T_current[:3, 3]
            R_current = T_current[:3, :3]
            quat_current = matrix_to_quaternion(R_current)
        else:
            pos_current = T_current[:3]
            quat_current = T_current[3:7] if len(T_current) >= 7 else self._array([1.0, 0.0, 0.0, 0.0])
        
        # Combine current pose
        if self._backend_manager.is_numpy:
            x_current = np.concatenate([pos_current, quat_current])
        else:
            import torch
            x_current = torch.cat([pos_current, quat_current])
        
        # Compute pose error
        pose_error = self._compute_pose_error(x_current, xd_desired)
        
        # Compute current end-effector velocity
        J = self.robot_model.jacobian(q, target_link=self.end_link)
        xd_current = J @ qd
        
        # Compute velocity error
        velocity_error = xdd_desired - xd_current
        
        # Control law in operational space
        if self.use_dynamics:
            # Full dynamics (requires dynamics module)
            # TODO: Implement when dynamics module is available
            # Λ = (J @ M^-1 @ J.T)^-1
            # μ = Λ @ (J @ M^-1 @ C @ qd - J_dot @ qd)
            # p = Λ @ J @ M^-1 @ g
            # F = Λ @ (xddd_desired + Kp @ pose_error + Kd @ velocity_error) + μ + p
            raise NotImplementedError("Full dynamics feedforward requires dynamics module (not yet implemented)")
        else:
            # Simple feedforward + feedback
            F = self.Kp @ pose_error + self.Kd @ velocity_error + self.Kff @ xddd_desired
        
        # Map to joint space via Jacobian transpose
        tau = J.T @ F
        
        # Add gravity compensation
        if self.use_gravity_compensation:
            # TODO: Implement gravity compensation when dynamics module is available
            pass
        
        return tau
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._current_time = 0.0

