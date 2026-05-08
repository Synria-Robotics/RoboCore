"""Trajectory quality evaluator for MuJoCo simulation results.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from typing import Dict, Any, Optional
import numpy as np

from .utils import (
    compute_jerk,
    detect_vibration,
    compute_rms_error,
    compute_max_error,
    fft_analysis
)


class TrajectoryEvaluator:
    """Evaluate trajectory execution quality.
    
    Computes metrics for smoothness, tracking error, torque requirements,
    and vibration analysis.
    """
    
    def __init__(self, dt: float = 0.002):
        """Initialize evaluator.
        
        Parameters
        ----------
        dt : float
            Time step for computation
        """
        self.dt = dt
    
    def evaluate(
        self,
        execution_data: Dict[str, np.ndarray],
        compute_vibration: bool = True
    ) -> Dict[str, Any]:
        """Evaluate trajectory execution.
        
        Parameters
        ----------
        execution_data : dict
            Execution data from TrajectoryExecutor with keys:
            - 'time': time array
            - 'q': actual positions
            - 'qd': actual velocities
            - 'qdd': actual accelerations
            - 'tau': torques
            - 'q_desired': desired positions
            - 'qd_desired': desired velocities
            - 'qdd_desired': desired accelerations
        compute_vibration : bool
            Whether to compute vibration analysis (FFT)
        
        Returns
        -------
        metrics : dict
            Dictionary with evaluation metrics
        """
        time = execution_data['time']
        q_actual = execution_data['q']
        qd_actual = execution_data['qd']
        qdd_actual = execution_data['qdd']
        tau = execution_data['tau']
        q_desired = execution_data['q_desired']
        qd_desired = execution_data['qd_desired']
        qdd_desired = execution_data['qdd_desired']
        
        metrics = {}
        
        # 1. Smoothness metrics
        metrics['smoothness'] = self._compute_smoothness(
            qdd_desired, qdd_actual
        )
        
        # 2. Tracking error
        metrics['tracking_error'] = self._compute_tracking_error(
            q_actual, qd_actual, q_desired, qd_desired
        )
        
        # 3. Torque analysis
        metrics['torque'] = self._compute_torque_metrics(tau)
        
        # 4. Vibration analysis
        if compute_vibration:
            metrics['vibration'] = self._compute_vibration(
                q_actual, qd_actual, tau
            )
        
        return metrics
    
    def _compute_smoothness(
        self,
        qdd_desired: np.ndarray,
        qdd_actual: np.ndarray
    ) -> Dict[str, Any]:
        """Compute smoothness metrics.
        
        Parameters
        ----------
        qdd_desired : np.ndarray
            Desired accelerations
        qdd_actual : np.ndarray
            Actual accelerations
        
        Returns
        -------
        smoothness : dict
            Smoothness metrics
        """
        # Jerk (rate of change of acceleration)
        jerk_desired = compute_jerk(qdd_desired, self.dt)
        jerk_actual = compute_jerk(qdd_actual, self.dt)
        
        # Velocity continuity (detect jumps)
        if qdd_desired.ndim > 1:
            qdd_diff_desired = np.diff(qdd_desired, axis=0)
            qdd_diff_actual = np.diff(qdd_actual, axis=0)
        else:
            qdd_diff_desired = np.diff(qdd_desired)
            qdd_diff_actual = np.diff(qdd_actual)
        
        # Acceleration continuity
        # Handle empty jerk arrays (when trajectory has too few points)
        if jerk_desired.size == 0 or jerk_actual.size == 0:
            # Return zeros if no jerk data available
            if jerk_desired.ndim > 1:
                n_joints = jerk_desired.shape[1] if jerk_desired.size > 0 else jerk_actual.shape[1] if jerk_actual.size > 0 else 1
                max_jerk_desired = np.zeros(n_joints)
                max_jerk_actual = np.zeros(n_joints)
                rms_jerk_desired = np.zeros(n_joints)
                rms_jerk_actual = np.zeros(n_joints)
            else:
                max_jerk_desired = 0.0
                max_jerk_actual = 0.0
                rms_jerk_desired = 0.0
                rms_jerk_actual = 0.0
        elif jerk_desired.ndim > 1:
            max_jerk_desired = np.max(np.abs(jerk_desired), axis=0)
            max_jerk_actual = np.max(np.abs(jerk_actual), axis=0)
            rms_jerk_desired = np.sqrt(np.mean(jerk_desired**2, axis=0))
            rms_jerk_actual = np.sqrt(np.mean(jerk_actual**2, axis=0))
        else:
            max_jerk_desired = np.max(np.abs(jerk_desired))
            max_jerk_actual = np.max(np.abs(jerk_actual))
            rms_jerk_desired = np.sqrt(np.mean(jerk_desired**2))
            rms_jerk_actual = np.sqrt(np.mean(jerk_actual**2))
        
        # Detect velocity discontinuities
        if qdd_diff_desired.size == 0 or qdd_diff_actual.size == 0:
            # Handle empty arrays
            if qdd_desired.ndim > 1:
                n_joints = qdd_desired.shape[1] if qdd_desired.size > 0 else qdd_actual.shape[1] if qdd_actual.size > 0 else 1
                max_accel_jump_desired = np.zeros(n_joints)
                max_accel_jump_actual = np.zeros(n_joints)
            else:
                max_accel_jump_desired = 0.0
                max_accel_jump_actual = 0.0
        elif qdd_desired.ndim > 1:
            max_accel_jump_desired = np.max(np.abs(qdd_diff_desired), axis=0)
            max_accel_jump_actual = np.max(np.abs(qdd_diff_actual), axis=0)
        else:
            max_accel_jump_desired = np.max(np.abs(qdd_diff_desired))
            max_accel_jump_actual = np.max(np.abs(qdd_diff_actual))
        
        return {
            'jerk': {
                'desired': {
                    'max': max_jerk_desired,
                    'rms': rms_jerk_desired
                },
                'actual': {
                    'max': max_jerk_actual,
                    'rms': rms_jerk_actual
                }
            },
            'acceleration_continuity': {
                'desired': {
                    'max_jump': max_accel_jump_desired
                },
                'actual': {
                    'max_jump': max_accel_jump_actual
                }
            }
        }
    
    def _compute_tracking_error(
        self,
        q_actual: np.ndarray,
        qd_actual: np.ndarray,
        q_desired: np.ndarray,
        qd_desired: np.ndarray
    ) -> Dict[str, Any]:
        """Compute tracking error metrics.
        
        Parameters
        ----------
        q_actual : np.ndarray
            Actual positions
        qd_actual : np.ndarray
            Actual velocities
        q_desired : np.ndarray
            Desired positions
        qd_desired : np.ndarray
            Desired velocities
        
        Returns
        -------
        tracking_error : dict
            Tracking error metrics
        """
        # Position error
        pos_error = q_actual - q_desired
        rms_pos_error = compute_rms_error(q_actual, q_desired)
        max_pos_error = compute_max_error(q_actual, q_desired)
        
        # Velocity error
        vel_error = qd_actual - qd_desired
        rms_vel_error = compute_rms_error(qd_actual, qd_desired)
        max_vel_error = compute_max_error(qd_actual, qd_desired)
        
        # Per-joint errors
        if pos_error.ndim > 1:
            rms_pos_error_per_joint = np.sqrt(np.mean(pos_error**2, axis=0))
            max_pos_error_per_joint = np.max(np.abs(pos_error), axis=0)
            rms_vel_error_per_joint = np.sqrt(np.mean(vel_error**2, axis=0))
            max_vel_error_per_joint = np.max(np.abs(vel_error), axis=0)
        else:
            rms_pos_error_per_joint = rms_pos_error
            max_pos_error_per_joint = max_pos_error
            rms_vel_error_per_joint = rms_vel_error
            max_vel_error_per_joint = max_vel_error
        
        return {
            'position': {
                'rms': rms_pos_error,
                'max': max_pos_error,
                'rms_per_joint': rms_pos_error_per_joint,
                'max_per_joint': max_pos_error_per_joint,
                'error_trajectory': pos_error
            },
            'velocity': {
                'rms': rms_vel_error,
                'max': max_vel_error,
                'rms_per_joint': rms_vel_error_per_joint,
                'max_per_joint': max_vel_error_per_joint,
                'error_trajectory': vel_error
            }
        }
    
    def _compute_torque_metrics(self, tau: np.ndarray) -> Dict[str, Any]:
        """Compute torque metrics.
        
        Parameters
        ----------
        tau : np.ndarray
            Torque array, shape (N, n_joints)
        
        Returns
        -------
        torque_metrics : dict
            Torque metrics
        """
        if tau.ndim == 1:
            tau = tau.reshape(-1, 1)
        
        # Torque magnitude
        tau_abs = np.abs(tau)
        max_tau = np.max(tau_abs, axis=0) if tau_abs.ndim > 1 else np.max(tau_abs)
        rms_tau = np.sqrt(np.mean(tau**2, axis=0)) if tau.ndim > 1 else np.sqrt(np.mean(tau**2))
        mean_tau = np.mean(tau_abs, axis=0) if tau_abs.ndim > 1 else np.mean(tau_abs)
        
        # Torque change rate
        tau_diff = np.diff(tau, axis=0)
        max_tau_rate = np.max(np.abs(tau_diff), axis=0) / self.dt if tau_diff.ndim > 1 else np.max(np.abs(tau_diff)) / self.dt
        rms_tau_rate = np.sqrt(np.mean(tau_diff**2, axis=0)) / self.dt if tau_diff.ndim > 1 else np.sqrt(np.mean(tau_diff**2)) / self.dt
        
        return {
            'max': max_tau,
            'rms': rms_tau,
            'mean': mean_tau,
            'rate': {
                'max': max_tau_rate,
                'rms': rms_tau_rate
            }
        }
    
    def _compute_vibration(
        self,
        q_actual: np.ndarray,
        qd_actual: np.ndarray,
        tau: np.ndarray
    ) -> Dict[str, Any]:
        """Compute vibration analysis.
        
        Parameters
        ----------
        q_actual : np.ndarray
            Actual positions
        qd_actual : np.ndarray
            Actual velocities
        tau : np.ndarray
            Torques
        
        Returns
        -------
        vibration : dict
            Vibration analysis results
        """
        vibration_results = {}
        
        # Analyze each joint
        n_joints = q_actual.shape[1] if q_actual.ndim > 1 else 1
        
        for j in range(n_joints):
            joint_idx = j if n_joints > 1 else None
            
            # Position vibration
            pos_vib = detect_vibration(
                q_actual[:, j] if n_joints > 1 else q_actual,
                self.dt,
                min_frequency=1.0,
                threshold=0.01
            )
            
            # Velocity vibration
            vel_vib = detect_vibration(
                qd_actual[:, j] if n_joints > 1 else qd_actual,
                self.dt,
                min_frequency=1.0,
                threshold=0.01
            )
            
            # Torque vibration
            tau_vib = detect_vibration(
                tau[:, j] if n_joints > 1 else tau,
                self.dt,
                min_frequency=1.0,
                threshold=0.1
            )
            
            vibration_results[f'joint_{j}'] = {
                'position': pos_vib,
                'velocity': vel_vib,
                'torque': tau_vib
            }
        
        # Overall vibration indicator
        has_vibration = any(
            v['position']['has_vibration'] or
            v['velocity']['has_vibration'] or
            v['torque']['has_vibration']
            for v in vibration_results.values()
        )
        
        return {
            'has_vibration': has_vibration,
            'joints': vibration_results
        }
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """Generate evaluation report.
        
        Parameters
        ----------
        metrics : dict
            Evaluation metrics from evaluate()
        output_file : str, optional
            File path to save report
        
        Returns
        -------
        report : str
            Text report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Trajectory Execution Evaluation Report")
        lines.append("=" * 70)
        lines.append("")
        
        # Smoothness
        lines.append("1. Smoothness Metrics")
        lines.append("-" * 70)
        smooth = metrics['smoothness']
        lines.append(f"  Jerk (desired):")
        lines.append(f"    Max: {smooth['jerk']['desired']['max']}")
        lines.append(f"    RMS: {smooth['jerk']['desired']['rms']}")
        lines.append(f"  Jerk (actual):")
        lines.append(f"    Max: {smooth['jerk']['actual']['max']}")
        lines.append(f"    RMS: {smooth['jerk']['actual']['rms']}")
        lines.append("")
        
        # Tracking error
        lines.append("2. Tracking Error")
        lines.append("-" * 70)
        tracking = metrics['tracking_error']
        lines.append(f"  Position Error:")
        lines.append(f"    RMS: {tracking['position']['rms']:.6f}")
        lines.append(f"    Max: {tracking['position']['max']:.6f}")
        lines.append(f"  Velocity Error:")
        lines.append(f"    RMS: {tracking['velocity']['rms']:.6f}")
        lines.append(f"    Max: {tracking['velocity']['max']:.6f}")
        lines.append("")
        
        # Torque
        lines.append("3. Torque Requirements")
        lines.append("-" * 70)
        torque = metrics['torque']
        lines.append(f"  Max Torque: {torque['max']}")
        lines.append(f"  RMS Torque: {torque['rms']}")
        lines.append(f"  Mean Torque: {torque['mean']}")
        lines.append(f"  Max Torque Rate: {torque['rate']['max']}")
        lines.append("")
        
        # Vibration
        if 'vibration' in metrics:
            lines.append("4. Vibration Analysis")
            lines.append("-" * 70)
            vib = metrics['vibration']
            lines.append(f"  Has Vibration: {vib['has_vibration']}")
            for joint_name, joint_vib in vib['joints'].items():
                lines.append(f"  {joint_name}:")
                if joint_vib['position']['has_vibration']:
                    lines.append(f"    Position: freq={joint_vib['position']['dominant_frequency']:.2f} Hz, "
                               f"magnitude={joint_vib['position']['max_magnitude']:.4f}")
                if joint_vib['velocity']['has_vibration']:
                    lines.append(f"    Velocity: freq={joint_vib['velocity']['dominant_frequency']:.2f} Hz, "
                               f"magnitude={joint_vib['velocity']['max_magnitude']:.4f}")
            lines.append("")
        
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


__all__ = ['TrajectoryEvaluator']

