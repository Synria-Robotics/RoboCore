"""Physics simulator for MuJoCo-based trajectory execution.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None


class PhysicsSimulator:
    """Physics simulator using MuJoCo engine.
    
    Supports multiple control modes: position, velocity, and torque control.
    """
    
    CONTROL_MODES = ['position', 'velocity', 'torque']
    
    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        timestep: float = 0.002,
        control_mode: str = 'position'
    ):
        """Initialize physics simulator.
        
        Parameters
        ----------
        mjcf_path : str, optional
            Path to MuJoCo MJCF/XML file
        urdf_path : str, optional
            Path to URDF file (will be converted)
        timestep : float
            Simulation timestep in seconds (default: 0.002, 500Hz)
        control_mode : str
            Control mode: 'position', 'velocity', or 'torque'
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required. Install with: pip install mujoco")
        
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control mode: {control_mode}. "
                           f"Must be one of {self.CONTROL_MODES}")
        
        self.mjcf_path = mjcf_path
        self.urdf_path = urdf_path
        self.timestep = timestep
        self.control_mode = control_mode
        
        # MuJoCo model and data
        self.model = None
        self.data = None
        
        # Load model
        if mjcf_path:
            self._load_mjcf(mjcf_path)
        elif urdf_path:
            self._load_from_urdf(urdf_path)
        else:
            raise ValueError("Either mjcf_path or urdf_path must be provided")
        
        # Configure simulation
        self.model.opt.timestep = timestep
        
        # State recording
        self.state_history = []
        self.time_history = []
        self.recording = False
    
    def _load_mjcf(self, mjcf_path: str):
        """Load MuJoCo model from MJCF file."""
        mjcf_path = Path(mjcf_path)
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def _load_from_urdf(self, urdf_path: str):
        """Load model from URDF file."""
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}. "
                             f"Consider converting to MJCF format.")
    
    def reset(self, q_init: Optional[np.ndarray] = None):
        """Reset simulation to initial state.
        
        Parameters
        ----------
        q_init : np.ndarray, optional
            Initial joint positions. If None, use zero configuration.
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if q_init is not None:
            if len(q_init) != self.model.nq:
                raise ValueError(f"q_init length {len(q_init)} != model DOF {self.model.nq}")
            self.data.qpos[:] = q_init
        
        mujoco.mj_forward(self.model, self.data)
        
        # Clear history
        self.state_history = []
        self.time_history = []
    
    def set_control_mode(self, mode: str):
        """Set control mode.
        
        Parameters
        ----------
        mode : str
            Control mode: 'position', 'velocity', or 'torque'
        """
        if mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control mode: {mode}")
        self.control_mode = mode
    
    def step_position_control(self, q_desired: np.ndarray):
        """Step simulation with position control.
        
        Parameters
        ----------
        q_desired : np.ndarray
            Desired joint positions, shape (n_joints,)
        """
        if len(q_desired) != self.model.nu:
            raise ValueError(f"q_desired length {len(q_desired)} != "
                           f"actuator count {self.model.nu}")
        
        # Set control signal (target position)
        self.data.ctrl[:] = q_desired
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
    
    def step_velocity_control(self, qd_desired: np.ndarray):
        """Step simulation with velocity control.
        
        Parameters
        ----------
        qd_desired : np.ndarray
            Desired joint velocities, shape (n_joints,)
        """
        if len(qd_desired) != self.model.nu:
            raise ValueError(f"qd_desired length {len(qd_desired)} != "
                           f"actuator count {self.model.nu}")
        
        # Set control signal (target velocity)
        self.data.ctrl[:] = qd_desired
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
    
    def step_torque_control(self, tau_desired: np.ndarray):
        """Step simulation with torque control.
        
        Parameters
        ----------
        tau_desired : np.ndarray
            Desired joint torques, shape (n_joints,)
        """
        if len(tau_desired) != self.model.nv:
            raise ValueError(f"tau_desired length {len(tau_desired)} != "
                           f"joint DOF {self.model.nv}")
        
        # Set applied forces (torques)
        self.data.qfrc_applied[:] = tau_desired
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
    
    def step(self, command: np.ndarray):
        """Step simulation with current control mode.
        
        Parameters
        ----------
        command : np.ndarray
            Control command (interpreted based on control mode):
            - position mode: desired positions
            - velocity mode: desired velocities
            - torque mode: desired torques
        """
        if self.control_mode == 'position':
            self.step_position_control(command)
        elif self.control_mode == 'velocity':
            self.step_velocity_control(command)
        elif self.control_mode == 'torque':
            self.step_torque_control(command)
        
        # Record state if recording
        if self.recording:
            self._record_state()
    
    def _record_state(self):
        """Record current simulation state."""
        state = {
            'q': self.data.qpos.copy(),
            'qd': self.data.qvel.copy(),
            'qdd': self.data.qacc.copy(),
            'tau': self.data.qfrc_actuator.copy() if self.model.nu > 0 else np.zeros(self.model.nv),
            'time': self.data.time
        }
        self.state_history.append(state)
        self.time_history.append(self.data.time)
    
    def start_recording(self):
        """Start recording simulation state."""
        self.recording = True
        self.state_history = []
        self.time_history = []
    
    def stop_recording(self):
        """Stop recording simulation state."""
        self.recording = False
    
    def get_recorded_data(self) -> Dict[str, np.ndarray]:
        """Get recorded simulation data.
        
        Returns
        -------
        data : dict
            Dictionary with keys: 'time', 'q', 'qd', 'qdd', 'tau'
        """
        if not self.state_history:
            return {
                'time': np.array([]),
                'q': np.array([]),
                'qd': np.array([]),
                'qdd': np.array([]),
                'tau': np.array([])
            }
        
        return {
            'time': np.array(self.time_history),
            'q': np.array([s['q'] for s in self.state_history]),
            'qd': np.array([s['qd'] for s in self.state_history]),
            'qdd': np.array([s['qdd'] for s in self.state_history]),
            'tau': np.array([s['tau'] for s in self.state_history])
        }
    
    def get_current_state(self) -> Dict[str, np.ndarray]:
        """Get current simulation state.
        
        Returns
        -------
        state : dict
            Dictionary with current 'q', 'qd', 'qdd', 'tau', 'time'
        """
        return {
            'q': self.data.qpos.copy(),
            'qd': self.data.qvel.copy(),
            'qdd': self.data.qacc.copy(),
            'tau': self.data.qfrc_actuator.copy() if self.model.nu > 0 else np.zeros(self.model.nv),
            'time': self.data.time
        }
    
    @property
    def nq(self) -> int:
        """Number of position DOF."""
        return self.model.nq
    
    @property
    def nv(self) -> int:
        """Number of velocity DOF."""
        return self.model.nv
    
    @property
    def nu(self) -> int:
        """Number of actuators."""
        return self.model.nu


__all__ = ['PhysicsSimulator']

