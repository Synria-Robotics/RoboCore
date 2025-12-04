"""Cartesian Space Trajectory Planning

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
from typing import TYPE_CHECKING, Tuple, Optional, Union
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


def linear_cartesian_trajectory(
    robot_model: "RobotModel",
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    duration: float,
    num_points: int = 50,
    q_init: Optional[np.ndarray] = None,
    ik_backend: str = 'numpy',
    ik_method: str = 'dls',
    **ik_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate linear trajectory in Cartesian space.
    
    End-effector moves in a straight line from start to end pose.
    Orientation is interpolated using SLERP (Spherical Linear Interpolation).
    
    Parameters
    ----------
    robot_model : RobotModel
        Robot model
    pose_start : np.ndarray
        Start pose as 4x4 homogeneous matrix
    pose_end : np.ndarray
        End pose as 4x4 homogeneous matrix
    duration : float
        Total duration (seconds)
    num_points : int
        Number of waypoints
    q_init : np.ndarray, optional
        Initial joint guess for IK (default: zeros)
    ik_backend : str
        IK backend ('numpy' or 'torch')
    ik_method : str
        IK method ('dls', 'jacobian', etc.)
    **ik_kwargs
        Additional arguments for IK solver
    
    Returns
    -------
    t : np.ndarray
        Time array, shape (num_points,)
    poses : np.ndarray
        Cartesian poses, shape (num_points, 4, 4)
    q : np.ndarray
        Joint configurations, shape (num_points, n_joints)
    
    Examples
    --------
    >>> from robocore.modeling.robot_model import RobotModel
    >>> from robocore.kinematics.fk import forward_kinematics
    >>> 
    >>> model = RobotModel('robot.urdf')
    >>> q_start = np.zeros(6)
    >>> q_end = np.array([1.0, 0.5, -0.3, 0.0, 0.5, 0.0])
    >>> 
    >>> T_start = forward_kinematics(model, q_start, backend='numpy', return_end=True)
    >>> T_end = forward_kinematics(model, q_end, backend='numpy', return_end=True)
    >>> 
    >>> t, poses, q = linear_cartesian_trajectory(model, T_start, T_end, duration=2.0)
    """
    from robocore.kinematics.ik import inverse_kinematics
    
    pose_start = np.asarray(pose_start)
    pose_end = np.asarray(pose_end)
    
    if pose_start.shape != (4, 4) or pose_end.shape != (4, 4):
        raise ValueError("Poses must be 4x4 homogeneous matrices")
    
    if duration <= 0:
        raise ValueError("duration must be positive")
    
    # Extract positions and rotations
    pos_start = pose_start[:3, 3]
    pos_end = pose_end[:3, 3]
    
    rot_start = Rotation.from_matrix(pose_start[:3, :3])
    rot_end = Rotation.from_matrix(pose_end[:3, :3])
    
    # Time array
    t = np.linspace(0, duration, num_points)
    
    # Initialize arrays
    poses = np.zeros((num_points, 4, 4))
    q = np.zeros((num_points, robot_model.num_dof()))
    
    # Initial guess for IK
    if q_init is None:
        q_init = np.zeros(robot_model.num_dof())
    
    q_current = q_init.copy()
    
    # Create SLERP interpolator for orientation
    key_times = [0, duration]
    key_rots = Rotation.concatenate([rot_start, rot_end])
    slerp = Slerp(key_times, key_rots)
    
    # Generate trajectory
    for i, ti in enumerate(t):
        alpha = ti / duration
        
        # Linear position interpolation
        pos = (1 - alpha) * pos_start + alpha * pos_end
        
        # SLERP orientation interpolation
        rot = slerp(ti)
        
        # Build pose matrix
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos
        poses[i] = T
        
        # Solve IK
        ik_result = inverse_kinematics(
            robot_model,
            T,
            q_current,
            backend=ik_backend,
            method=ik_method,
            **ik_kwargs
        )
        
        if not ik_result['success']:
            print(f"Warning: IK failed at waypoint {i}/{num_points} (t={ti:.3f}s)")
            # Use previous solution
            q[i] = q_current
        else:
            q[i] = ik_result['q']
            q_current = ik_result['q']  # Use as seed for next iteration
    
    return t, poses, q


def circular_cartesian_trajectory(
    robot_model: "RobotModel",
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    duration: float,
    num_points: int = 50,
    orientation: Union[str, np.ndarray] = 'constant',
    q_init: Optional[np.ndarray] = None,
    ik_backend: str = 'numpy',
    ik_method: str = 'dls',
    **ik_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate circular arc trajectory in Cartesian space.
    
    End-effector moves along a circular arc in 3D space.
    
    Parameters
    ----------
    robot_model : RobotModel
        Robot model
    center : np.ndarray
        Center of circle, shape (3,)
    normal : np.ndarray
        Normal vector to plane of circle, shape (3,)
    radius : float
        Radius of circle (meters)
    start_angle : float
        Start angle (radians)
    end_angle : float
        End angle (radians)
    duration : float
        Total duration (seconds)
    num_points : int
        Number of waypoints
    orientation : str or np.ndarray
        'constant': keep initial orientation
        'tangent': orient along tangent to circle
        3x3 rotation matrix: use fixed orientation
    q_init : np.ndarray, optional
        Initial joint guess for IK
    ik_backend : str
        IK backend
    ik_method : str
        IK method
    **ik_kwargs
        Additional IK arguments
    
    Returns
    -------
    t : np.ndarray
        Time array
    poses : np.ndarray
        Cartesian poses, shape (num_points, 4, 4)
    q : np.ndarray
        Joint configurations
    
    Examples
    --------
    >>> center = np.array([0.3, 0.0, 0.5])
    >>> normal = np.array([0.0, 0.0, 1.0])  # Circle in XY plane
    >>> t, poses, q = circular_cartesian_trajectory(
    ...     model, center, normal, radius=0.1,
    ...     start_angle=0, end_angle=np.pi, duration=3.0
    ... )
    """
    from robocore.kinematics.ik import inverse_kinematics
    
    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    if radius <= 0:
        raise ValueError("radius must be positive")
    
    if duration <= 0:
        raise ValueError("duration must be positive")
    
    # Create orthonormal basis for circle plane
    # Find two perpendicular vectors in the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0, 0, 1]))
    else:
        u = np.cross(normal, np.array([1, 0, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Time array
    t = np.linspace(0, duration, num_points)
    angles = np.linspace(start_angle, end_angle, num_points)
    
    # Initialize
    poses = np.zeros((num_points, 4, 4))
    q = np.zeros((num_points, robot_model.num_dof()))
    
    if q_init is None:
        q_init = np.zeros(robot_model.num_dof())
    q_current = q_init.copy()
    
    # Determine orientation handling
    if isinstance(orientation, str) and orientation == 'constant':
        # Get current orientation from FK of q_init
        from robocore.kinematics.fk import forward_kinematics
        T_init = forward_kinematics(robot_model, q_init, backend='numpy', return_end=True)
        R_constant = T_init[:3, :3]
    elif isinstance(orientation, str) and orientation == 'tangent':
        R_constant = None
    else:
        # Fixed orientation provided
        R_constant = np.asarray(orientation)
        if R_constant.shape != (3, 3):
            raise ValueError("orientation must be 3x3 rotation matrix or 'constant'/'tangent'")
    
    # Generate trajectory
    for i, (_, angle) in enumerate(zip(t, angles)):
        # Position on circle
        pos = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
        
        # Orientation
        if isinstance(orientation, str):
            if orientation == 'tangent':
                # Orient tangent to circle
                tangent = radius * (-np.sin(angle) * u + np.cos(angle) * v)
                tangent = tangent / np.linalg.norm(tangent)
                
                # Build rotation matrix with tangent as X axis
                x_axis = tangent
                z_axis = normal
                y_axis = np.cross(z_axis, x_axis)
                
                R = np.column_stack([x_axis, y_axis, z_axis])
            else:  # 'constant'
                R = R_constant
        else:
            R = orientation
        
        # Build pose
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        poses[i] = T
        
        # Solve IK
        ik_result = inverse_kinematics(
            robot_model, T, q_current,
            backend=ik_backend, method=ik_method, **ik_kwargs
        )
        
        if not ik_result['success']:
            print(f"Warning: IK failed at waypoint {i}/{num_points}")
            q[i] = q_current
        else:
            q[i] = ik_result['q']
            q_current = ik_result['q']
    
    return t, poses, q


def cartesian_waypoint_trajectory(
    robot_model: "RobotModel",
    waypoint_poses: np.ndarray,
    durations: Union[np.ndarray, float],
    num_points_per_segment: int = 50,
    q_init: Optional[np.ndarray] = None,
    ik_backend: str = 'numpy',
    ik_method: str = 'dls',
    **ik_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trajectory through multiple Cartesian waypoints.
    
    Linear interpolation between each pair of waypoints.
    
    Parameters
    ----------
    robot_model : RobotModel
        Robot model
    waypoint_poses : np.ndarray
        Waypoint poses, shape (num_waypoints, 4, 4)
    durations : np.ndarray or float
        Duration for each segment
    num_points_per_segment : int
        Points per segment
    q_init : np.ndarray, optional
        Initial joint configuration
    ik_backend : str
        IK backend
    ik_method : str
        IK method
    **ik_kwargs
        Additional IK arguments
    
    Returns
    -------
    t : np.ndarray
        Time array
    poses : np.ndarray
        Cartesian poses
    q : np.ndarray
        Joint configurations
    """
    waypoint_poses = np.asarray(waypoint_poses)
    num_waypoints = waypoint_poses.shape[0]
    
    if num_waypoints < 2:
        raise ValueError("Need at least 2 waypoints")
    
    # Handle durations
    if np.isscalar(durations):
        durations = np.full(num_waypoints - 1, durations)
    else:
        durations = np.asarray(durations)
    
    # Generate segments
    t_segments = []
    poses_segments = []
    q_segments = []
    
    t_offset = 0.0
    
    for i in range(num_waypoints - 1):
        pose_start = waypoint_poses[i]
        pose_end = waypoint_poses[i + 1]
        duration = durations[i]
        
        # Use last q as init for next segment
        if i == 0:
            q_seg_init = q_init
        else:
            q_seg_init = q_segments[-1][-1]
        
        t_seg, poses_seg, q_seg = linear_cartesian_trajectory(
            robot_model, pose_start, pose_end, duration,
            num_points=num_points_per_segment,
            q_init=q_seg_init,
            ik_backend=ik_backend,
            ik_method=ik_method,
            **ik_kwargs
        )
        
        # Add offset
        t_seg = t_seg + t_offset
        
        # Store (skip first point of non-first segments)
        if i == 0:
            t_segments.append(t_seg)
            poses_segments.append(poses_seg)
            q_segments.append(q_seg)
        else:
            t_segments.append(t_seg[1:])
            poses_segments.append(poses_seg[1:])
            q_segments.append(q_seg[1:])
        
        t_offset += duration
    
    # Concatenate
    t = np.concatenate(t_segments)
    poses = np.vstack(poses_segments)
    q = np.vstack(q_segments)
    
    return t, poses, q
