"""Joint Space Trajectory Planning

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
from typing import Union, Optional, Tuple


def cubic_polynomial_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    num_points: int = 50,
    v_start: Optional[np.ndarray] = None,
    v_end: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate cubic polynomial trajectory in joint space.
    
    Cubic polynomial ensures C1 continuity (position and velocity continuous).
    Coefficients: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    
    Parameters
    ----------
    q_start : np.ndarray
        Start joint configuration, shape (n_joints,)
    q_end : np.ndarray
        End joint configuration, shape (n_joints,)
    duration : float
        Total duration of trajectory (seconds)
    num_points : int
        Number of waypoints to generate
    v_start : np.ndarray, optional
        Start velocity (default: zero)
    v_end : np.ndarray, optional
        End velocity (default: zero)
    
    Returns
    -------
    t : np.ndarray
        Time array, shape (num_points,)
    q : np.ndarray
        Joint positions, shape (num_points, n_joints)
    qd : np.ndarray
        Joint velocities, shape (num_points, n_joints)
    qdd : np.ndarray
        Joint accelerations, shape (num_points, n_joints)
    
    Examples
    --------
    >>> q_start = np.array([0.0, 0.0, 0.0])
    >>> q_end = np.array([1.0, -0.5, 0.8])
    >>> t, q, qd, qdd = cubic_polynomial_trajectory(q_start, q_end, duration=2.0)
    >>> print(q.shape)  # (50, 3)
    """
    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)
    n_joints = len(q_start)
    
    if len(q_end) != n_joints:
        raise ValueError(f"q_start and q_end must have same dimension, got {n_joints} and {len(q_end)}")
    
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    
    # Default velocities
    if v_start is None:
        v_start = np.zeros(n_joints)
    if v_end is None:
        v_end = np.zeros(n_joints)
    
    v_start = np.asarray(v_start)
    v_end = np.asarray(v_end)
    
    # Time array
    t = np.linspace(0, duration, num_points)
    
    # Solve for cubic coefficients for each joint
    # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    # q(0) = a0 = q_start
    # q(T) = a0 + a1*T + a2*T^2 + a3*T^3 = q_end
    # q'(0) = a1 = v_start
    # q'(T) = a1 + 2*a2*T + 3*a3*T^2 = v_end
    
    T = duration
    a0 = q_start
    a1 = v_start
    
    # Solve for a2, a3:
    # a0 + a1*T + a2*T^2 + a3*T^3 = q_end
    # a1 + 2*a2*T + 3*a3*T^2 = v_end
    #
    # a2*T^2 + a3*T^3 = q_end - a0 - a1*T
    # 2*a2*T + 3*a3*T^2 = v_end - a1
    #
    # a2 = (v_end - a1) / (2*T) - (3*a3*T) / 2
    # Substitute into first equation and solve for a3:
    
    a3 = (2 * (q_end - q_start - v_start * T) - T * (v_end - v_start)) / (T ** 3)
    a2 = (3 * (q_end - q_start) - T * (2 * v_start + v_end)) / (T ** 2)
    
    # Generate trajectory
    q = np.zeros((num_points, n_joints))
    qd = np.zeros((num_points, n_joints))
    qdd = np.zeros((num_points, n_joints))
    
    for i, ti in enumerate(t):
        q[i] = a0 + a1 * ti + a2 * ti**2 + a3 * ti**3
        qd[i] = a1 + 2 * a2 * ti + 3 * a3 * ti**2
        qdd[i] = 2 * a2 + 6 * a3 * ti
    
    return t, q, qd, qdd


def quintic_polynomial_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    num_points: int = 50,
    v_start: Optional[np.ndarray] = None,
    v_end: Optional[np.ndarray] = None,
    a_start: Optional[np.ndarray] = None,
    a_end: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate quintic (5th order) polynomial trajectory in joint space.
    
    Quintic polynomial ensures C2 continuity (position, velocity, acceleration continuous).
    Coefficients: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    Parameters
    ----------
    q_start : np.ndarray
        Start joint configuration, shape (n_joints,)
    q_end : np.ndarray
        End joint configuration, shape (n_joints,)
    duration : float
        Total duration of trajectory (seconds)
    num_points : int
        Number of waypoints to generate
    v_start : np.ndarray, optional
        Start velocity (default: zero)
    v_end : np.ndarray, optional
        End velocity (default: zero)
    a_start : np.ndarray, optional
        Start acceleration (default: zero)
    a_end : np.ndarray, optional
        End acceleration (default: zero)
    
    Returns
    -------
    t : np.ndarray
        Time array, shape (num_points,)
    q : np.ndarray
        Joint positions, shape (num_points, n_joints)
    qd : np.ndarray
        Joint velocities, shape (num_points, n_joints)
    qdd : np.ndarray
        Joint accelerations, shape (num_points, n_joints)
    
    Examples
    --------
    >>> q_start = np.array([0.0, 0.0, 0.0])
    >>> q_end = np.array([1.0, -0.5, 0.8])
    >>> t, q, qd, qdd = quintic_polynomial_trajectory(q_start, q_end, duration=2.0)
    >>> print(q.shape)  # (50, 3)
    """
    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)
    n_joints = len(q_start)
    
    if len(q_end) != n_joints:
        raise ValueError(f"q_start and q_end must have same dimension")
    
    if duration <= 0:
        raise ValueError(f"duration must be positive")
    
    # Default velocities and accelerations
    if v_start is None:
        v_start = np.zeros(n_joints)
    if v_end is None:
        v_end = np.zeros(n_joints)
    if a_start is None:
        a_start = np.zeros(n_joints)
    if a_end is None:
        a_end = np.zeros(n_joints)
    
    v_start = np.asarray(v_start)
    v_end = np.asarray(v_end)
    a_start = np.asarray(a_start)
    a_end = np.asarray(a_end)
    
    # Time array
    t = np.linspace(0, duration, num_points)
    
    # Solve for quintic coefficients
    # Boundary conditions:
    # q(0) = q_start, q(T) = q_end
    # q'(0) = v_start, q'(T) = v_end
    # q''(0) = a_start, q''(T) = a_end
    
    T = duration
    
    # Coefficient matrix (6x6 system for each joint)
    # [1   0    0     0      0       0    ] [a0]   [q_start]
    # [0   1    0     0      0       0    ] [a1]   [v_start]
    # [0   0    2     0      0       0    ] [a2] = [a_start/2]
    # [1   T   T^2   T^3    T^4     T^5  ] [a3]   [q_end]
    # [0   1   2T    3T^2   4T^3    5T^4 ] [a4]   [v_end]
    # [0   0    2    6T     12T^2   20T^3] [a5]   [a_end]
    
    # Direct solution:
    a0 = q_start
    a1 = v_start
    a2 = a_start / 2.0
    
    # Solve 3x3 system for a3, a4, a5
    A = np.array([
        [T**3, T**4, T**5],
        [3*T**2, 4*T**3, 5*T**4],
        [6*T, 12*T**2, 20*T**3]
    ])
    
    b = np.array([
        q_end - a0 - a1 * T - a2 * T**2,
        v_end - a1 - 2 * a2 * T,
        a_end - 2 * a2
    ])
    
    # Solve for each joint
    a3 = np.zeros(n_joints)
    a4 = np.zeros(n_joints)
    a5 = np.zeros(n_joints)
    
    for j in range(n_joints):
        b_j = np.array([b[0][j], b[1][j], b[2][j]])
        coeffs = np.linalg.solve(A, b_j)
        a3[j], a4[j], a5[j] = coeffs
    
    # Generate trajectory
    q = np.zeros((num_points, n_joints))
    qd = np.zeros((num_points, n_joints))
    qdd = np.zeros((num_points, n_joints))
    
    for i, ti in enumerate(t):
        q[i] = a0 + a1*ti + a2*ti**2 + a3*ti**3 + a4*ti**4 + a5*ti**5
        qd[i] = a1 + 2*a2*ti + 3*a3*ti**2 + 4*a4*ti**3 + 5*a5*ti**4
        qdd[i] = 2*a2 + 6*a3*ti + 12*a4*ti**2 + 20*a5*ti**3
    
    return t, q, qd, qdd


def linear_joint_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    num_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate linear interpolation trajectory in joint space.
    
    Simple linear interpolation with constant velocity and zero acceleration.
    Velocity is discontinuous at start and end.
    
    Parameters
    ----------
    q_start : np.ndarray
        Start joint configuration, shape (n_joints,)
    q_end : np.ndarray
        End joint configuration, shape (n_joints,)
    duration : float
        Total duration of trajectory (seconds)
    num_points : int
        Number of waypoints to generate
    
    Returns
    -------
    t : np.ndarray
        Time array, shape (num_points,)
    q : np.ndarray
        Joint positions, shape (num_points, n_joints)
    qd : np.ndarray
        Joint velocities, shape (num_points, n_joints)
    qdd : np.ndarray
        Joint accelerations (all zeros), shape (num_points, n_joints)
    
    Examples
    --------
    >>> q_start = np.array([0.0, 0.0, 0.0])
    >>> q_end = np.array([1.0, -0.5, 0.8])
    >>> t, q, qd, qdd = linear_joint_trajectory(q_start, q_end, duration=2.0)
    """
    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)
    n_joints = len(q_start)
    
    if duration <= 0:
        raise ValueError(f"duration must be positive")
    
    # Time array
    t = np.linspace(0, duration, num_points)
    
    # Linear interpolation
    q = np.zeros((num_points, n_joints))
    for i, ti in enumerate(t):
        alpha = ti / duration
        q[i] = (1 - alpha) * q_start + alpha * q_end
    
    # Constant velocity
    qd = np.tile((q_end - q_start) / duration, (num_points, 1))
    
    # Zero acceleration
    qdd = np.zeros((num_points, n_joints))
    
    return t, q, qd, qdd


def multi_waypoint_trajectory(
    waypoints: np.ndarray,
    durations: Union[np.ndarray, float],
    method: str = 'cubic',
    num_points_per_segment: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trajectory through multiple waypoints.
    
    Parameters
    ----------
    waypoints : np.ndarray
        Waypoints, shape (num_waypoints, n_joints)
    durations : np.ndarray or float
        Duration for each segment. If float, same duration for all segments.
        Shape (num_waypoints-1,) if array
    method : str
        Interpolation method: 'linear', 'cubic', 'quintic'
    num_points_per_segment : int
        Number of points per segment
    
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
    >>> waypoints = np.array([[0, 0, 0], [1, 0.5, 0.2], [0.5, -0.5, 0.8]])
    >>> t, q, qd, qdd = multi_waypoint_trajectory(waypoints, duration=1.0, method='cubic')
    """
    waypoints = np.asarray(waypoints)
    num_waypoints = waypoints.shape[0]
    n_joints = waypoints.shape[1]
    
    if num_waypoints < 2:
        raise ValueError("Need at least 2 waypoints")
    
    # Handle durations
    if np.isscalar(durations):
        durations = np.full(num_waypoints - 1, durations)
    else:
        durations = np.asarray(durations)
        if len(durations) != num_waypoints - 1:
            raise ValueError(f"Expected {num_waypoints-1} durations, got {len(durations)}")
    
    # Generate trajectory for each segment
    t_segments = []
    q_segments = []
    qd_segments = []
    qdd_segments = []
    
    t_offset = 0.0
    
    for i in range(num_waypoints - 1):
        q_start = waypoints[i]
        q_end = waypoints[i + 1]
        duration = durations[i]
        
        # Get velocity at waypoint (ensure continuity)
        if i == 0:
            v_start = None
        else:
            v_start = qd_segments[-1][-1]  # Use last velocity from previous segment
        
        if i == num_waypoints - 2:
            v_end = None
        else:
            # Estimate velocity for next segment
            v_end = (waypoints[i + 2] - q_end) / durations[i + 1] if i + 2 < num_waypoints else None
        
        # Generate segment
        if method == 'linear':
            t_seg, q_seg, qd_seg, qdd_seg = linear_joint_trajectory(
                q_start, q_end, duration, num_points_per_segment
            )
        elif method == 'cubic':
            t_seg, q_seg, qd_seg, qdd_seg = cubic_polynomial_trajectory(
                q_start, q_end, duration, num_points_per_segment, v_start, v_end
            )
        elif method == 'quintic':
            t_seg, q_seg, qd_seg, qdd_seg = quintic_polynomial_trajectory(
                q_start, q_end, duration, num_points_per_segment, v_start, v_end
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add time offset
        t_seg = t_seg + t_offset
        
        # Store (skip first point of non-first segments to avoid duplication)
        if i == 0:
            t_segments.append(t_seg)
            q_segments.append(q_seg)
            qd_segments.append(qd_seg)
            qdd_segments.append(qdd_seg)
        else:
            t_segments.append(t_seg[1:])
            q_segments.append(q_seg[1:])
            qd_segments.append(qd_seg[1:])
            qdd_segments.append(qdd_seg[1:])
        
        t_offset += duration
    
    # Concatenate all segments
    t = np.concatenate(t_segments)
    q = np.vstack(q_segments)
    qd = np.vstack(qd_segments)
    qdd = np.vstack(qdd_segments)
    
    return t, q, qd, qdd
