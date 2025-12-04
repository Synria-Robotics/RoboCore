"""Velocity Profile Generation

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


def trapezoidal_velocity_profile(
    distance: float,
    v_max: float,
    a_max: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trapezoidal velocity profile.
    
    Profile has three phases:
    1. Acceleration phase: linear ramp-up
    2. Cruise phase: constant velocity
    3. Deceleration phase: linear ramp-down
    
    Parameters
    ----------
    distance : float
        Total distance to travel
    v_max : float
        Maximum velocity
    a_max : float
        Maximum acceleration (positive)
    v_start : float
        Starting velocity (default: 0)
    v_end : float
        Ending velocity (default: 0)
    num_points : int
        Number of points in profile
    
    Returns
    -------
    t : np.ndarray
        Time array, shape (num_points,)
    s : np.ndarray
        Position array, shape (num_points,)
    v : np.ndarray
        Velocity array, shape (num_points,)
    a : np.ndarray
        Acceleration array, shape (num_points,)
    
    Examples
    --------
    >>> t, s, v, a = trapezoidal_velocity_profile(
    ...     distance=1.0, v_max=0.5, a_max=1.0
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3,1,1); plt.plot(t, s); plt.ylabel('Position')
    >>> plt.subplot(3,1,2); plt.plot(t, v); plt.ylabel('Velocity')
    >>> plt.subplot(3,1,3); plt.plot(t, a); plt.ylabel('Acceleration')
    """
    if distance <= 0:
        raise ValueError("distance must be positive")
    if v_max <= 0:
        raise ValueError("v_max must be positive")
    if a_max <= 0:
        raise ValueError("a_max must be positive")
    
    # Check if cruise phase exists
    # Distance needed to accelerate from v_start to v_max
    d_accel = (v_max**2 - v_start**2) / (2 * a_max)
    # Distance needed to decelerate from v_max to v_end
    d_decel = (v_max**2 - v_end**2) / (2 * a_max)
    
    # Check if we can reach v_max
    if d_accel + d_decel > distance:
        # No cruise phase - triangular profile
        # Find peak velocity
        v_peak = np.sqrt((2 * a_max * distance + v_start**2 + v_end**2) / 2)
        d_accel = (v_peak**2 - v_start**2) / (2 * a_max)
        d_decel = (v_peak**2 - v_end**2) / (2 * a_max)
        d_cruise = 0.0
        v_cruise = v_peak
    else:
        # Normal trapezoidal profile
        d_cruise = distance - d_accel - d_decel
        v_cruise = v_max
    
    # Calculate phase durations
    t_accel = (v_cruise - v_start) / a_max
    t_cruise = d_cruise / v_cruise if v_cruise > 0 else 0.0
    t_decel = (v_cruise - v_end) / a_max
    
    t_total = t_accel + t_cruise + t_decel
    
    # Generate time array
    t = np.linspace(0, t_total, num_points)
    
    # Initialize arrays
    s = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    
    # Fill arrays
    for i, ti in enumerate(t):
        if ti <= t_accel:
            # Acceleration phase
            a[i] = a_max
            v[i] = v_start + a_max * ti
            s[i] = v_start * ti + 0.5 * a_max * ti**2
        elif ti <= t_accel + t_cruise:
            # Cruise phase
            a[i] = 0.0
            v[i] = v_cruise
            t_cruise_elapsed = ti - t_accel
            s[i] = (v_start * t_accel + 0.5 * a_max * t_accel**2 +
                    v_cruise * t_cruise_elapsed)
        else:
            # Deceleration phase
            a[i] = -a_max
            t_decel_elapsed = ti - t_accel - t_cruise
            v[i] = v_cruise - a_max * t_decel_elapsed
            s[i] = (v_start * t_accel + 0.5 * a_max * t_accel**2 +
                    v_cruise * t_cruise +
                    v_cruise * t_decel_elapsed - 0.5 * a_max * t_decel_elapsed**2)
    
    return t, s, v, a


def s_curve_velocity_profile(
    distance: float,
    v_max: float,
    a_max: float,
    j_max: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate S-curve (jerk-limited) velocity profile.
    
    Profile has seven phases:
    1. Jerk-in (acceleration increases)
    2. Constant acceleration
    3. Jerk-out (acceleration decreases)
    4. Constant velocity
    5. Jerk-in (deceleration increases)
    6. Constant deceleration
    7. Jerk-out (deceleration decreases)
    
    This profile minimizes jerk, resulting in smoother motion.
    
    Parameters
    ----------
    distance : float
        Total distance to travel
    v_max : float
        Maximum velocity
    a_max : float
        Maximum acceleration
    j_max : float
        Maximum jerk (rate of change of acceleration)
    v_start : float
        Starting velocity
    v_end : float
        Ending velocity
    num_points : int
        Number of points
    
    Returns
    -------
    t : np.ndarray
        Time array
    s : np.ndarray
        Position array
    v : np.ndarray
        Velocity array
    a : np.ndarray
        Acceleration array
    j : np.ndarray
        Jerk array
    
    Examples
    --------
    >>> t, s, v, a, j = s_curve_velocity_profile(
    ...     distance=1.0, v_max=0.5, a_max=1.0, j_max=5.0
    ... )
    """
    if distance <= 0:
        raise ValueError("distance must be positive")
    if v_max <= 0:
        raise ValueError("v_max must be positive")
    if a_max <= 0:
        raise ValueError("a_max must be positive")
    if j_max <= 0:
        raise ValueError("j_max must be positive")
    
    # Time to reach max acceleration
    t_j = a_max / j_max
    
    # Check if we can reach max acceleration
    # Distance during jerk phases
    d_j = j_max * t_j**3 / 6
    
    # Simplified S-curve (assuming symmetric accel/decel)
    # For full implementation, need to solve complex system of equations
    
    # Use trapezoidal approximation with smoothed corners
    # This is a simplified version - full S-curve is more complex
    
    # Generate trapezoidal base
    t_trap, s_trap, v_trap, a_trap = trapezoidal_velocity_profile(
        distance, v_max, a_max, v_start, v_end, num_points
    )
    
    # Smooth acceleration transitions using jerk limit
    a_smoothed = np.zeros_like(a_trap)
    j = np.zeros_like(a_trap)
    
    # Simple smoothing filter
    window_size = max(3, int(num_points * t_j / t_trap[-1]))
    for i in range(len(a_trap)):
        i_start = max(0, i - window_size // 2)
        i_end = min(len(a_trap), i + window_size // 2 + 1)
        a_smoothed[i] = np.mean(a_trap[i_start:i_end])
    
    # Calculate jerk as derivative of acceleration
    dt = t_trap[1] - t_trap[0]
    j[1:-1] = (a_smoothed[2:] - a_smoothed[:-2]) / (2 * dt)
    j[0] = (a_smoothed[1] - a_smoothed[0]) / dt
    j[-1] = (a_smoothed[-1] - a_smoothed[-2]) / dt
    
    # Clamp jerk
    j = np.clip(j, -j_max, j_max)
    
    # Recalculate acceleration from jerk
    a_final = np.zeros_like(j)
    a_final[0] = a_smoothed[0]
    for i in range(1, len(j)):
        a_final[i] = a_final[i-1] + j[i] * dt
    
    # Recalculate velocity
    v_final = np.zeros_like(a_final)
    v_final[0] = v_start
    for i in range(1, len(a_final)):
        v_final[i] = v_final[i-1] + a_final[i] * dt
    
    # Recalculate position
    s_final = np.zeros_like(v_final)
    s_final[0] = 0
    for i in range(1, len(v_final)):
        s_final[i] = s_final[i-1] + v_final[i] * dt
    
    # Scale to match desired distance
    s_final = s_final * (distance / s_final[-1])
    
    return t_trap, s_final, v_final, a_final, j


def constant_velocity_profile(
    distance: float,
    velocity: float,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate constant velocity profile.
    
    Simple linear motion with constant velocity (infinite acceleration).
    
    Parameters
    ----------
    distance : float
        Total distance
    velocity : float
        Constant velocity
    num_points : int
        Number of points
    
    Returns
    -------
    t : np.ndarray
        Time array
    s : np.ndarray
        Position array
    v : np.ndarray
        Velocity array
    a : np.ndarray
        Acceleration array (all zeros)
    """
    if distance <= 0:
        raise ValueError("distance must be positive")
    if velocity <= 0:
        raise ValueError("velocity must be positive")
    
    duration = distance / velocity
    t = np.linspace(0, duration, num_points)
    
    s = velocity * t
    v = np.full_like(t, velocity)
    a = np.zeros_like(t)
    
    return t, s, v, a


def scale_trajectory_to_profile(
    q_path: np.ndarray,
    velocity_profile: np.ndarray,
    duration: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale a geometric path to follow a velocity profile.
    
    Given a path in joint space (e.g., from multi-waypoint interpolation)
    and a velocity profile, compute time-parameterized trajectory.
    
    Parameters
    ----------
    q_path : np.ndarray
        Geometric path, shape (num_path_points, n_joints)
    velocity_profile : np.ndarray
        Velocity at each point, shape (num_path_points,)
    duration : float
        Total duration
    
    Returns
    -------
    t : np.ndarray
        Time array
    q : np.ndarray
        Joint positions
    qd : np.ndarray
        Joint velocities
    qdd : np.ndarray
        Joint accelerations
    
    Examples
    --------
    >>> # Generate geometric path
    >>> q_path = np.linspace(q_start, q_end, 100)
    >>> 
    >>> # Generate velocity profile
    >>> _, _, v, _ = trapezoidal_velocity_profile(
    ...     distance=1.0, v_max=0.5, a_max=1.0
    ... )
    >>> 
    >>> # Combine
    >>> t, q, qd, qdd = scale_trajectory_to_profile(q_path, v, duration=2.0)
    """
    q_path = np.asarray(q_path)
    velocity_profile = np.asarray(velocity_profile)
    
    num_points = q_path.shape[0]
    n_joints = q_path.shape[1]
    
    if len(velocity_profile) != num_points:
        raise ValueError("velocity_profile length must match q_path length")
    
    # Compute path parameter s (arc length along path)
    s = np.zeros(num_points)
    for i in range(1, num_points):
        ds = np.linalg.norm(q_path[i] - q_path[i-1])
        s[i] = s[i-1] + ds
    
    # Normalize s to [0, 1]
    if s[-1] > 0:
        s = s / s[-1]
    
    # Time parameterization
    t = np.linspace(0, duration, num_points)
    dt = t[1] - t[0]
    
    # Position (already have from q_path)
    q = q_path.copy()
    
    # Velocity: qd = dq/dt = (dq/ds) * (ds/dt)
    qd = np.zeros_like(q)
    for i in range(1, num_points - 1):
        dq_ds = (q[i+1] - q[i-1]) / (s[i+1] - s[i-1] + 1e-10)
        ds_dt = velocity_profile[i]
        qd[i] = dq_ds * ds_dt
    
    # Boundary conditions
    qd[0] = qd[1]
    qd[-1] = qd[-2]
    
    # Acceleration: qdd = dqd/dt
    qdd = np.zeros_like(q)
    for i in range(1, num_points - 1):
        qdd[i] = (qd[i+1] - qd[i-1]) / (2 * dt)
    qdd[0] = qdd[1]
    qdd[-1] = qdd[-2]
    
    return t, q, qd, qdd
