"""Utility functions for MuJoCo physics simulation.

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

import numpy as np
from typing import Tuple, Optional
from scipy import interpolate
from scipy.fft import fft, fftfreq


def interpolate_trajectory(
    t_original: np.ndarray,
    q_original: np.ndarray,
    t_new: np.ndarray,
    kind: str = 'cubic'
) -> np.ndarray:
    """Interpolate trajectory to new time points.
    
    Parameters
    ----------
    t_original : np.ndarray
        Original time array, shape (N,)
    q_original : np.ndarray
        Original joint positions, shape (N, n_joints)
    t_new : np.ndarray
        New time array, shape (M,)
    kind : str
        Interpolation kind ('linear', 'cubic', 'quintic')
    
    Returns
    -------
    q_new : np.ndarray
        Interpolated joint positions, shape (M, n_joints)
    """
    t_original = np.asarray(t_original)
    q_original = np.asarray(q_original)
    t_new = np.asarray(t_new)
    
    if q_original.ndim == 1:
        q_original = q_original.reshape(-1, 1)
    
    n_joints = q_original.shape[1]
    q_new = np.zeros((len(t_new), n_joints))
    
    for j in range(n_joints):
        if kind == 'linear':
            interp_func = interpolate.interp1d(
                t_original, q_original[:, j],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
        elif kind == 'cubic':
            interp_func = interpolate.interp1d(
                t_original, q_original[:, j],
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
        elif kind == 'quintic':
            # Use B-spline for quintic interpolation
            tck = interpolate.splrep(t_original, q_original[:, j], k=5, s=0)
            q_new[:, j] = interpolate.splev(t_new, tck)
            continue
        else:
            raise ValueError(f"Unknown interpolation kind: {kind}")
        
        q_new[:, j] = interp_func(t_new)
    
    return q_new


def interpolate_trajectory_with_derivatives(
    t_original: np.ndarray,
    q_original: np.ndarray,
    qd_original: Optional[np.ndarray] = None,
    qdd_original: Optional[np.ndarray] = None,
    t_new: np.ndarray = None,
    dt: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate trajectory with velocity and acceleration.
    
    Parameters
    ----------
    t_original : np.ndarray
        Original time array, shape (N,)
    q_original : np.ndarray
        Original joint positions, shape (N, n_joints)
    qd_original : np.ndarray, optional
        Original joint velocities, shape (N, n_joints)
    qdd_original : np.ndarray, optional
        Original joint accelerations, shape (N, n_joints)
    t_new : np.ndarray, optional
        New time array. If None, generate from dt
    dt : float, optional
        Time step for new trajectory. Used if t_new is None
    
    Returns
    -------
    t_new : np.ndarray
        New time array
    q_new : np.ndarray
        Interpolated positions
    qd_new : np.ndarray
        Interpolated velocities
    qdd_new : np.ndarray
        Interpolated accelerations
    """
    t_original = np.asarray(t_original)
    q_original = np.asarray(q_original)
    
    if q_original.ndim == 1:
        q_original = q_original.reshape(-1, 1)
    
    # Generate new time array if needed
    if t_new is None:
        if dt is None:
            dt = (t_original[-1] - t_original[0]) / (len(t_original) - 1)
        t_new = np.arange(t_original[0], t_original[-1] + dt, dt)
    
    n_joints = q_original.shape[1]
    n_points = len(t_new)
    
    q_new = np.zeros((n_points, n_joints))
    qd_new = np.zeros((n_points, n_joints))
    qdd_new = np.zeros((n_points, n_joints))
    
    # Use cubic spline for smooth derivatives
    for j in range(n_joints):
        # Fit cubic spline
        tck = interpolate.splrep(t_original, q_original[:, j], k=3, s=0)
        
        # Evaluate position
        q_new[:, j] = interpolate.splev(t_new, tck)
        
        # Evaluate velocity (first derivative)
        qd_new[:, j] = interpolate.splev(t_new, tck, der=1)
        
        # Evaluate acceleration (second derivative)
        qdd_new[:, j] = interpolate.splev(t_new, tck, der=2)
    
    return t_new, q_new, qd_new, qdd_new


def compute_jerk(qdd: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (rate of change of acceleration).
    
    Parameters
    ----------
    qdd : np.ndarray
        Acceleration array, shape (N, n_joints)
    dt : float
        Time step
    
    Returns
    -------
    jerk : np.ndarray
        Jerk array, shape (N-1, n_joints)
    """
    qdd = np.asarray(qdd)
    if qdd.ndim == 1:
        qdd = qdd.reshape(-1, 1)
    
    jerk = np.diff(qdd, axis=0) / dt
    return jerk


def fft_analysis(
    signal: np.ndarray,
    dt: float,
    joint_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform FFT analysis on signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal array, shape (N,) or (N, n_joints)
    dt : float
        Time step
    joint_idx : int, optional
        Joint index if signal is multi-dimensional
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency array
    magnitude : np.ndarray
        FFT magnitude
    """
    signal = np.asarray(signal)
    
    if signal.ndim > 1:
        if joint_idx is None:
            # Use first joint
            signal = signal[:, 0]
        else:
            signal = signal[:, joint_idx]
    
    n = len(signal)
    fft_vals = fft(signal)
    frequencies = fftfreq(n, dt)
    
    # Only return positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(fft_vals[positive_freq_idx])
    
    return frequencies, magnitude


def detect_vibration(
    signal: np.ndarray,
    dt: float,
    min_frequency: float = 1.0,
    threshold: float = 0.1
) -> dict:
    """Detect vibration in signal using FFT.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal array, shape (N,) or (N, n_joints)
    dt : float
        Time step
    min_frequency : float
        Minimum frequency to consider (Hz)
    threshold : float
        Magnitude threshold for detection
    
    Returns
    -------
    result : dict
        Dictionary with vibration information:
        - has_vibration: bool
        - dominant_frequency: float
        - max_magnitude: float
        - frequencies: np.ndarray
        - magnitudes: np.ndarray
    """
    frequencies, magnitudes = fft_analysis(signal, dt)
    
    # Filter by minimum frequency
    valid_idx = frequencies >= min_frequency
    frequencies = frequencies[valid_idx]
    magnitudes = magnitudes[valid_idx]
    
    if len(magnitudes) == 0:
        return {
            'has_vibration': False,
            'dominant_frequency': 0.0,
            'max_magnitude': 0.0,
            'frequencies': np.array([]),
            'magnitudes': np.array([])
        }
    
    # Find dominant frequency
    max_idx = np.argmax(magnitudes)
    dominant_freq = frequencies[max_idx]
    max_magnitude = magnitudes[max_idx]
    
    has_vibration = max_magnitude > threshold
    
    return {
        'has_vibration': has_vibration,
        'dominant_frequency': dominant_freq,
        'max_magnitude': max_magnitude,
        'frequencies': frequencies,
        'magnitudes': magnitudes
    }


def compute_rms_error(
    actual: np.ndarray,
    desired: np.ndarray
) -> float:
    """Compute RMS error between actual and desired.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values, shape (N, n_joints) or (N,)
    desired : np.ndarray
        Desired values, shape (N, n_joints) or (N,)
    
    Returns
    -------
    rms_error : float
        Root mean square error
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    
    error = actual - desired
    if error.ndim > 1:
        error = error.flatten()
    
    rms_error = np.sqrt(np.mean(error**2))
    return rms_error


def compute_max_error(
    actual: np.ndarray,
    desired: np.ndarray
) -> float:
    """Compute maximum absolute error.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    desired : np.ndarray
        Desired values
    
    Returns
    -------
    max_error : float
        Maximum absolute error
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    
    error = np.abs(actual - desired)
    if error.ndim > 1:
        error = error.flatten()
    
    max_error = np.max(error)
    return max_error


__all__ = [
    'interpolate_trajectory',
    'interpolate_trajectory_with_derivatives',
    'compute_jerk',
    'fft_analysis',
    'detect_vibration',
    'compute_rms_error',
    'compute_max_error',
]

