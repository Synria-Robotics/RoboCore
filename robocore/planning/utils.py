"""Utility functions for trajectory planning visualization.

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

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from robocore.transform import quaternion_to_matrix
from robocore.utils.backend import to_numpy


def draw_axis(ax, origin, R, scale=0.05, alpha=0.8):
    """Draw a coordinate frame (axis) at given origin with given rotation.

    :param ax: 3D axes object
    :param origin: Origin position [3]
    :param R: Rotation matrix [3, 3]
    :param scale: Scale of the axis
    :param alpha: Transparency
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Define unit vectors for X, Y, Z axes
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale

    # Draw axes
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', arrow_length_ratio=0.3, linewidth=2, alpha=alpha)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', arrow_length_ratio=0.3, linewidth=2, alpha=alpha)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', arrow_length_ratio=0.3, linewidth=2, alpha=alpha)


def plot_cartesian_trajectory(trajectory, waypoints=None, sample_step=10, figsize=(15, 5)):
    """Plot Cartesian space trajectory with positions and orientations.

    :param trajectory: Trajectory dictionary with 'positions', 'orientations', 't', etc.
    :param waypoints: Optional waypoints to plot (list of 4x4 matrices or positions)
    :param sample_step: Step size for sampling orientation axes
    :param figsize: Figure size
    :return: Figure and axes objects
    """
    if not HAS_MATPLOTLIB:
        return None, None
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory positions
    if 'positions' in trajectory:
        positions = to_numpy(trajectory['positions'])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color='blue', label='Trajectory', linewidth=2, alpha=0.8)

        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                   c='red', s=100, marker='s', label='End', zorder=5)

    # Plot waypoints
    if waypoints is not None:
        if isinstance(waypoints, list) or isinstance(waypoints, np.ndarray):
            if len(waypoints) > 0:
                # Check if waypoints are 4x4 matrices or positions
                if isinstance(waypoints[0], np.ndarray) and waypoints[0].shape == (4, 4):
                    waypoint_positions = np.array([wp[:3, 3] for wp in waypoints])
                else:
                    waypoint_positions = np.array(waypoints)
                
                ax.scatter(waypoint_positions[:, 0], waypoint_positions[:, 1], waypoint_positions[:, 2],
                           c='purple', s=150, marker='*', label='Waypoints',
                           edgecolors='black', linewidths=1, zorder=5)

    # Plot orientation axes (sampled)
    if 'orientations' in trajectory:
        orientations = to_numpy(trajectory['orientations'])
        positions = to_numpy(trajectory['positions'])

        # Convert quaternions to rotation matrices if needed
        if len(orientations.shape) == 2 and orientations.shape[1] == 4:
            R_matrices = np.zeros((len(orientations), 3, 3))
            for i, q in enumerate(orientations):
                R_matrices[i] = quaternion_to_matrix(q)
            orientations = R_matrices

        # Sample and draw axes
        for i in range(0, len(orientations), sample_step):
            pos = positions[i]
            R = orientations[i]
            draw_axis(ax, pos, R, scale=0.03, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Cartesian Space Trajectory')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_joint_trajectory(trajectory, waypoints=None, figsize=(12, 8)):
    """Plot joint space trajectory.

    :param trajectory: Trajectory dictionary with 'q', 'qd', 'qdd', 't', etc.
    :param waypoints: Optional waypoints to plot
    :param figsize: Figure size
    :return: Figure and axes objects
    """
    if not HAS_MATPLOTLIB:
        return None, None
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle('Joint Space Trajectory', fontsize=14)

    t = to_numpy(trajectory['t'])
    q = to_numpy(trajectory['q'])
    qd = to_numpy(trajectory.get('qd', np.zeros_like(q)))
    qdd = to_numpy(trajectory.get('qdd', np.zeros_like(q)))

    n_joints = q.shape[1]

    # Position
    for i in range(n_joints):
        axes[0].plot(t, q[:, i], label=f'Joint {i+1}', linewidth=2, alpha=0.8)

    # Plot waypoints if available
    if waypoints is not None:
        waypoint_times = np.linspace(t[0], t[-1], len(waypoints))
        for j in range(min(n_joints, waypoints.shape[1])):
            axes[0].scatter(waypoint_times, waypoints[:, j],
                           c='purple', s=150, marker='*',
                           edgecolors='black', linewidths=1, zorder=5,
                           label='Waypoints' if j == 0 else '')

    axes[0].set_ylabel('Position (rad)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Velocity
    for i in range(n_joints):
        axes[1].plot(t, qd[:, i], label=f'Joint {i+1}', linewidth=2, alpha=0.8)

    axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Acceleration
    for i in range(n_joints):
        axes[2].plot(t, qdd[:, i], label=f'Joint {i+1}', linewidth=2, alpha=0.8)

    axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_cartesian_with_ik(trajectory, waypoints, joint_angles, ik_results, figsize=(18, 6)):
    """Plot Cartesian trajectory with joint angles and IK success rate.

    :param trajectory: Trajectory dictionary
    :param waypoints: Waypoints (list of 4x4 matrices)
    :param joint_angles: Joint angles array (num_points, n_dof)
    :param ik_results: List of IK result dictionaries
    :param figsize: Figure size
    :return: Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig = plt.figure(figsize=figsize)

    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')

    # Plot trajectory positions
    if 'positions' in trajectory:
        positions = to_numpy(trajectory['positions'])
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                 color='blue', label='Trajectory', linewidth=2, alpha=0.8)

        # Mark start and end
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                    c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                    c='red', s=100, marker='s', label='End', zorder=5)

    # Plot waypoints
    if waypoints is not None:
        waypoint_positions = np.array([wp[:3, 3] for wp in waypoints])
        ax1.scatter(waypoint_positions[:, 0], waypoint_positions[:, 1], waypoint_positions[:, 2],
                    c='purple', s=150, marker='*', label='Waypoints',
                    edgecolors='black', linewidths=1, zorder=5)

    # Plot orientation axes (sampled)
    sample_step = 10
    if 'orientations' in trajectory:
        orientations = to_numpy(trajectory['orientations'])
        positions = to_numpy(trajectory['positions'])

        # Convert quaternions to rotation matrices if needed
        if len(orientations.shape) == 2 and orientations.shape[1] == 4:
            R_matrices = np.zeros((len(orientations), 3, 3))
            for i, q in enumerate(orientations):
                R_matrices[i] = quaternion_to_matrix(q)
            orientations = R_matrices

        # Sample and draw axes
        for i in range(0, len(orientations), sample_step):
            pos = positions[i]
            R = orientations[i]
            draw_axis(ax1, pos, R, scale=0.03, alpha=0.6)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Cartesian Space Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Joint angles plot
    ax2 = fig.add_subplot(132)
    joint_angles_np = to_numpy(joint_angles)
    t = to_numpy(trajectory['t'])

    for i in range(joint_angles_np.shape[1]):
        ax2.plot(t, joint_angles_np[:, i], label=f'Joint {i+1}', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Joint Angle (rad)')
    ax2.set_title('Joint Angle Trajectory')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # IK success rate over time
    ax3 = fig.add_subplot(133)
    success_mask = np.array([r['success'] for r in ik_results])
    ax3.plot(t, success_mask.astype(float), 'g-', linewidth=2, alpha=0.8, label='IK Success')
    ax3.fill_between(t, 0, success_mask.astype(float), alpha=0.3, color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Success (1) / Failure (0)')
    ax3.set_title('IK Success Rate Over Time')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

