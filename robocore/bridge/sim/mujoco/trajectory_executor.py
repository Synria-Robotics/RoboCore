"""Trajectory executor for MuJoCo physics simulation.

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

from typing import Optional, Dict, Any
import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

from .physics_simulator import PhysicsSimulator
from .utils import interpolate_trajectory_with_derivatives


class TrajectoryExecutor:
    """Execute trajectory in MuJoCo physics simulation.
    
    Supports servo limits (velocity and acceleration constraints)
    for position-controlled actuators.
    """
    
    def __init__(
        self,
        simulator: PhysicsSimulator,
        max_velocity: Optional[np.ndarray] = None,
        max_acceleration: Optional[np.ndarray] = None,
        servo_kp: float = 100.0,
        servo_kv: float = 10.0
    ):
        """Initialize trajectory executor.
        
        Parameters
        ----------
        simulator : PhysicsSimulator
            Physics simulator instance
        max_velocity : np.ndarray, optional
            Maximum velocity for each joint (rad/s). If None, no limit.
        max_acceleration : np.ndarray, optional
            Maximum acceleration for each joint (rad/s²). If None, no limit.
        servo_kp : float
            Position gain for servo control (used when limits are active)
        servo_kv : float
            Velocity gain for servo control (used when limits are active)
        """
        self.simulator = simulator
        self.servo_kp = servo_kp
        self.servo_kv = servo_kv
        
        # Servo limits
        if max_velocity is not None:
            max_velocity = np.asarray(max_velocity)
            if max_velocity.ndim == 0:
                max_velocity = np.full(simulator.nu, max_velocity)
        self.max_velocity = max_velocity
        
        if max_acceleration is not None:
            max_acceleration = np.asarray(max_acceleration)
            if max_acceleration.ndim == 0:
                max_acceleration = np.full(simulator.nu, max_acceleration)
        self.max_acceleration = max_acceleration
        
        # Execution state
        self.trajectory_time = None
        self.trajectory_q = None
        self.trajectory_qd = None
        self.trajectory_qdd = None
        self.current_index = 0
        self.is_executing = False
    
    def load_trajectory(
        self,
        q_trajectory: np.ndarray,
        time: np.ndarray,
        qd_trajectory: Optional[np.ndarray] = None,
        qdd_trajectory: Optional[np.ndarray] = None
    ):
        """Load trajectory for execution.
        
        Parameters
        ----------
        q_trajectory : np.ndarray
            Joint positions, shape (N, n_joints)
        time : np.ndarray
            Time array, shape (N,)
        qd_trajectory : np.ndarray, optional
            Joint velocities, shape (N, n_joints)
        qdd_trajectory : np.ndarray, optional
            Joint accelerations, shape (N, n_joints)
        """
        q_trajectory = np.asarray(q_trajectory)
        time = np.asarray(time)
        
        if q_trajectory.ndim == 1:
            q_trajectory = q_trajectory.reshape(-1, 1)
        
        if len(time) != len(q_trajectory):
            raise ValueError(f"Time length {len(time)} != trajectory length {len(q_trajectory)}")
        
        if q_trajectory.shape[1] != self.simulator.nu:
            raise ValueError(f"Trajectory DOF {q_trajectory.shape[1]} != "
                           f"actuator count {self.simulator.nu}")
        
        self.trajectory_time = time
        self.trajectory_q = q_trajectory
        
        # Interpolate derivatives if not provided
        if qd_trajectory is None or qdd_trajectory is None:
            _, _, qd_interp, qdd_interp = interpolate_trajectory_with_derivatives(
                time, q_trajectory, qd_trajectory, qdd_trajectory, t_new=time
            )
            if qd_trajectory is None:
                qd_trajectory = qd_interp
            if qdd_trajectory is None:
                qdd_trajectory = qdd_interp
        
        self.trajectory_qd = np.asarray(qd_trajectory)
        self.trajectory_qdd = np.asarray(qdd_trajectory)
        self.current_index = 0
    
    def execute(
        self,
        q_init: Optional[np.ndarray] = None,
        record: bool = True
    ) -> Dict[str, np.ndarray]:
        """Execute loaded trajectory.
        
        Parameters
        ----------
        q_init : np.ndarray, optional
            Initial joint positions. If None, use trajectory start.
        record : bool
            Whether to record execution data
        
        Returns
        -------
        execution_data : dict
            Dictionary with execution results:
            - 'time': time array
            - 'q': actual positions
            - 'qd': actual velocities
            - 'qdd': actual accelerations
            - 'tau': torques
            - 'q_desired': desired positions
            - 'qd_desired': desired velocities
        """
        if self.trajectory_q is None:
            raise ValueError("No trajectory loaded. Call load_trajectory() first.")
        
        # Reset simulator
        if q_init is None:
            q_init = self.trajectory_q[0]
        self.simulator.reset(q_init)
        
        # Start recording
        if record:
            self.simulator.start_recording()
        
        # Interpolate trajectory to match simulation timestep
        dt = self.simulator.timestep
        t_sim = np.arange(
            self.trajectory_time[0],
            self.trajectory_time[-1] + dt,
            dt
        )
        
        # Interpolate trajectory
        from .utils import interpolate_trajectory_with_derivatives
        t_interp, q_interp, qd_interp, qdd_interp = interpolate_trajectory_with_derivatives(
            self.trajectory_time,
            self.trajectory_q,
            self.trajectory_qd,
            self.trajectory_qdd,
            t_new=t_sim
        )
        
        # Execute trajectory
        self.is_executing = True
        desired_data = {
            'time': t_interp,
            'q': q_interp,
            'qd': qd_interp,
            'qdd': qdd_interp
        }
        
        for i in range(len(t_interp)):
            q_desired = q_interp[i]
            
            # Apply control based on mode and limits
            if self.simulator.control_mode == 'position':
                if self.max_velocity is not None or self.max_acceleration is not None:
                    # Apply servo limits
                    self._step_with_servo_limits(q_desired)
                else:
                    # Direct position control - use step() to ensure recording
                    self.simulator.step(q_desired)
            elif self.simulator.control_mode == 'velocity':
                qd_desired = qd_interp[i]
                # Use step() to ensure recording
                self.simulator.step(qd_desired)
            elif self.simulator.control_mode == 'torque':
                # Compute required torque using inverse dynamics
                # First set desired position and velocity
                self.simulator.data.qpos[:] = q_desired
                self.simulator.data.qvel[:] = qd_interp[i]
                if MUJOCO_AVAILABLE:
                    mujoco.mj_forward(self.simulator.model, self.simulator.data)
                    
                    # Compute inverse dynamics
                    mujoco.mj_inverse(self.simulator.model, self.simulator.data)
                    tau_required = self.simulator.data.qfrc_inverse.copy()
                    
                    # Apply torque - use step() to ensure recording
                    self.simulator.step(tau_required)
        
        self.is_executing = False
        
        # Get recorded data
        if record:
            self.simulator.stop_recording()
            actual_data = self.simulator.get_recorded_data()
        else:
            # Get final state only
            state = self.simulator.get_current_state()
            actual_data = {
                'time': np.array([state['time']]),
                'q': state['q'].reshape(1, -1),
                'qd': state['qd'].reshape(1, -1),
                'qdd': state['qdd'].reshape(1, -1),
                'tau': state['tau'].reshape(1, -1)
            }
        
        # Combine desired and actual
        execution_data = {
            'time': desired_data['time'],
            'q_desired': desired_data['q'],
            'qd_desired': desired_data['qd'],
            'qdd_desired': desired_data['qdd'],
            'q': actual_data['q'],
            'qd': actual_data['qd'],
            'qdd': actual_data['qdd'],
            'tau': actual_data['tau']
        }
        
        return execution_data
    
    def _step_with_servo_limits(self, q_desired: np.ndarray):
        """Step with servo velocity and acceleration limits.
        
        Parameters
        ----------
        q_desired : np.ndarray
            Desired joint positions
        """
        # Get current state
        state = self.simulator.get_current_state()
        q_current = state['q']
        qd_current = state['qd']
        dt = self.simulator.timestep
        
        # Compute desired velocity (PD control)
        q_error = q_desired - q_current
        qd_desired = q_error * self.servo_kp
        
        # Limit velocity
        if self.max_velocity is not None:
            qd_desired = np.clip(qd_desired, -self.max_velocity, self.max_velocity)
        
        # Limit acceleration
        if self.max_acceleration is not None:
            qdd_desired = (qd_desired - qd_current) / dt
            qdd_desired = np.clip(qdd_desired, -self.max_acceleration, self.max_acceleration)
            qd_limited = qd_current + qdd_desired * dt
        else:
            qd_limited = qd_desired
        
        # Update position based on limited velocity
        q_limited = q_current + qd_limited * dt
        
        # Apply position control with limited target - use step() to ensure recording
        self.simulator.step(q_limited)
    
    def set_servo_limits(
        self,
        max_velocity: Optional[np.ndarray] = None,
        max_acceleration: Optional[np.ndarray] = None
    ):
        """Set servo velocity and acceleration limits.
        
        Parameters
        ----------
        max_velocity : np.ndarray, optional
            Maximum velocity for each joint (rad/s)
        max_acceleration : np.ndarray, optional
            Maximum acceleration for each joint (rad/s²)
        """
        if max_velocity is not None:
            max_velocity = np.asarray(max_velocity)
            if max_velocity.ndim == 0:
                max_velocity = np.full(self.simulator.nu, max_velocity)
            self.max_velocity = max_velocity
        
        if max_acceleration is not None:
            max_acceleration = np.asarray(max_acceleration)
            if max_acceleration.ndim == 0:
                max_acceleration = np.full(self.simulator.nu, max_acceleration)
            self.max_acceleration = max_acceleration


__all__ = ['TrajectoryExecutor']

