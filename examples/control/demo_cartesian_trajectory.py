"""Cartesian Trajectory Tracking Controller Demo

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse

import robocore as rc
from robocore.control import CartesianTrajectoryController
from robocore.modeling import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.planning import (
    LinearPositionPlanner,
    SLERPPlanner,
    CircularArcPlanner,
)
from robocore.transform.se3 import make_transform
from robocore.transform.conversions import quaternion_to_matrix
from robocore.utils.beauty_logger import beauty_print


def demo_linear_trajectory():
    """Demo tracking linear cartesian trajectory."""
    beauty_print("=" * 70)
    beauty_print("Cartesian Trajectory Controller - Linear Trajectory")
    beauty_print("=" * 70)
    
    # Load robot model
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        beauty_print("\nNote: synriard not available, using dummy model path")
        model_path = "path/to/robot.urdf"
    
    beauty_print(f"\nLoading robot model: {model_path}")
    robot = RobotModel(model_path)
    
    # Get start and end poses
    q_start = np.zeros(robot.num_dof)
    q_end = np.array([0.5, -0.3, 0.2, 0.0, 0.5, 0.0])
    
    T_start = forward_kinematics(robot, q_start, return_end=True)
    T_end = forward_kinematics(robot, q_end, return_end=True)
    
    beauty_print("\nGenerating linear cartesian trajectory...")
    # Use LinearPositionPlanner for position and SLERPPlanner for orientation
    pos_planner = LinearPositionPlanner()
    ori_planner = SLERPPlanner()
    
    # Generate position trajectory
    pos_result = pos_planner.plan(
        start=T_start[:3, 3],
        end=T_end[:3, 3],
        duration=2.0,
        num_points=100
    )
    
    # Generate orientation trajectory
    ori_result = ori_planner.plan(
        start=T_start[:3, :3],
        end=T_end[:3, :3],
        duration=2.0,
        num_points=100
    )
    
    # Combine into full pose trajectory
    t = pos_result['t']
    num_points = len(t)
    poses = np.zeros((num_points, 4, 4))
    for i in range(num_points):
        # Convert quaternion to rotation matrix
        quat = ori_result['orientations'][i]
        R = quaternion_to_matrix(quat)
        poses[i] = make_transform(R, pos_result['positions'][i])
    
    # Generate joint trajectory using IK (simplified - use start and end)
    q_traj = np.zeros((num_points, robot.num_dof))
    q_traj[0] = q_start
    q_traj[-1] = q_end
    # Linear interpolation for intermediate points (simplified)
    for i in range(1, num_points - 1):
        alpha = i / (num_points - 1)
        q_traj[i] = q_start + alpha * (q_end - q_start)
    
    print(f"  Trajectory duration: {t[-1]:.2f} seconds")
    print(f"  Number of waypoints: {len(t)}")
    print(f"  Start position: {T_start[:3, 3]}")
    print(f"  End position: {T_end[:3, 3]}")
    
    # Create controller
    controller = CartesianTrajectoryController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5]),
        Kff=np.diag([50, 50, 50, 25, 25, 25])  # Feedforward gain
    )
    
    # Set trajectory
    controller.set_trajectory(trajectory_data={
        't': t,
        'poses': poses
    })
    
    beauty_print("\nController Configuration:")
    print(f"  Kp: {np.diag(controller.Kp)}")
    print(f"  Kd: {np.diag(controller.Kd)}")
    print(f"  Kff: {np.diag(controller.Kff)}")
    print("  Control law: τ = J^T·[Kp·e + Kd·ė + Kff·ẍd] + g(q)")
    
    # Simulate tracking at different time points
    beauty_print("\nTrajectory Tracking Simulation:")
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for t_current in time_points:
        # Current state (simulated)
        idx = int(t_current / t[-1] * (len(t) - 1))
        q_current = q_traj[idx] + 0.01 * np.random.randn(robot.num_dof)
        qd_current = np.zeros(robot.num_dof)  # Simplified
        
        # Compute control torque
        tau = controller.compute(
            q=q_current,
            qd=qd_current,
            t=t_current
        )
        
        print(f"\n  Time: {t_current:.2f}s")
        print(f"    Desired pose position: {poses[idx][:3, 3]}")
        print(f"    Control torque magnitude: {np.linalg.norm(tau)}")


def demo_circular_trajectory():
    """Demo tracking circular cartesian trajectory."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Cartesian Trajectory Controller - Circular Trajectory")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    # Get initial pose
    q_init = np.zeros(robot.num_dof)
    T_init = forward_kinematics(robot, q_init, return_end=True)
    center = T_init[:3, 3]
    
    # Generate circular trajectory
    beauty_print("\nGenerating circular trajectory...")
    # Create via point for circular arc
    via_pos = center + np.array([0.1, 0.0, 0.0])  # Point on circle
    end_pos = center + np.array([0.0, 0.1, 0.0])  # Another point on circle
    
    # Create poses
    R_init = T_init[:3, :3]
    T_via = make_transform(R_init, via_pos)
    T_end = make_transform(R_init, end_pos)
    
    planner = CircularArcPlanner()
    result = planner.plan(
        start=T_init,
        via=T_via,
        end=T_end,
        duration=3.0,
        num_points=150
    )
    
    t = result['t']
    poses = result['poses']
    
    # Generate joint trajectory (simplified - linear interpolation)
    q_traj = np.zeros((len(t), robot.num_dof))
    q_traj[0] = q_init
    # For simplicity, use linear interpolation
    for i in range(1, len(t)):
        alpha = i / (len(t) - 1)
        q_traj[i] = q_init * (1 - alpha)  # Simplified
    
    print(f"  Trajectory duration: {t[-1]:.2f} seconds")
    print(f"  Center: {center}")
    print(f"  Radius: 0.1 m")
    print(f"  Points: {len(t)}")
    
    controller = CartesianTrajectoryController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5]),
        Kff=np.diag([50, 50, 50, 25, 25, 25])
    )
    
    controller.set_trajectory(trajectory_data={
        't': t,
        'poses': poses
    })
    
    beauty_print("\nTrajectory Tracking (circular motion):")
    time_points = [0.0, 0.75, 1.5, 2.25, 3.0]
    
    for t_current in time_points:
        idx = int(t_current / t[-1] * (len(t) - 1))
        q_current = q_traj[idx]
        qd_current = np.zeros(robot.num_dof)
        
        tau = controller.compute(
            q=q_current,
            qd=qd_current,
            t=t_current
        )
        
        print(f"\n  Time: {t_current:.2f}s")
        print(f"    Position: {poses[idx][:3, 3]}")
        print(f"    Control torque magnitude: {np.linalg.norm(tau)}")


def demo_direct_mode():
    """Demo direct mode (providing desired states directly)."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Cartesian Trajectory Controller - Direct Mode")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    controller = CartesianTrajectoryController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5]),
        Kff=np.diag([50, 50, 50, 25, 25, 25])
    )
    
    # Current state
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    
    # Get current pose
    T_current = forward_kinematics(robot, q, return_end=True)
    from robocore.transform.conversions import matrix_to_quaternion
    pos_current = T_current[:3, 3]
    quat_current = matrix_to_quaternion(T_current[:3, :3])
    
    # Desired pose and velocity
    pos_desired = pos_current + np.array([0.1, 0.05, 0.0])
    quat_desired = quat_current
    xd_desired = np.concatenate([pos_desired, quat_desired])
    
    xdd_desired = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0])  # Desired velocity
    xddd_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Desired acceleration
    
    beauty_print("\nDirect Mode (no trajectory function needed):")
    print(f"  Desired pose: {xd_desired}")
    print(f"  Desired velocity: {xdd_desired}")
    print(f"  Desired acceleration: {xddd_desired}")
    
    tau = controller.compute(
        q=q,
        qd=qd,
        xd_desired=xd_desired,
        xdd_desired=xdd_desired,
        xddd_desired=xddd_desired
    )
    
    beauty_print("\nControl Output:")
    print(f"  Control torque: {tau}")


def demo_feedforward_benefit():
    """Demo showing benefit of feedforward term."""
    beauty_print("\n" + "=" * 70)
    beauty_print("Feedforward vs Feedback Only Comparison")
    beauty_print("=" * 70)
    
    try:
        from synriard import get_model_path
        model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except ImportError:
        model_path = "path/to/robot.urdf"
    
    robot = RobotModel(model_path)
    
    q_start = np.zeros(robot.num_dof)
    q_end = np.array([0.5, -0.3, 0.2, 0.0, 0.5, 0.0])
    
    T_start = forward_kinematics(robot, q_start, return_end=True)
    T_end = forward_kinematics(robot, q_end, return_end=True)
    
    # Generate trajectory using planners
    pos_planner = LinearPositionPlanner()
    ori_planner = SLERPPlanner()
    
    pos_result = pos_planner.plan(
        start=T_start[:3, 3],
        end=T_end[:3, 3],
        duration=2.0,
        num_points=50
    )
    
    ori_result = ori_planner.plan(
        start=T_start[:3, :3],
        end=T_end[:3, :3],
        duration=2.0,
        num_points=50
    )
    
    t = pos_result['t']
    num_points = len(t)
    poses = np.zeros((num_points, 4, 4))
    for i in range(num_points):
        # Convert quaternion to rotation matrix
        quat = ori_result['orientations'][i]
        R = quaternion_to_matrix(quat)
        poses[i] = make_transform(R, pos_result['positions'][i])
    
    q_traj = np.zeros((num_points, robot.num_dof))
    q_traj[0] = q_start
    q_traj[-1] = q_end
    for i in range(1, num_points - 1):
        alpha = i / (num_points - 1)
        q_traj[i] = q_start + alpha * (q_end - q_start)
    
    # Controller with feedforward
    controller_ff = CartesianTrajectoryController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5]),
        Kff=np.diag([50, 50, 50, 25, 25, 25])
    )
    
    # Controller without feedforward
    controller_no_ff = CartesianTrajectoryController(
        robot_model=robot,
        Kp=np.diag([100, 100, 100, 50, 50, 50]),
        Kd=np.diag([10, 10, 10, 5, 5, 5]),
        Kff=np.zeros(6)  # No feedforward
    )
    
    controller_ff.set_trajectory(trajectory_data={'t': t, 'poses': poses})
    controller_no_ff.set_trajectory(trajectory_data={'t': t, 'poses': poses})
    
    beauty_print("\nComparison at t=1.0s (mid-trajectory):")
    t_current = 1.0
    
    idx = int(t_current / t[-1] * (len(t) - 1))
    q_current = q_traj[idx]
    qd_current = np.zeros(robot.num_dof)
    
    tau_ff = controller_ff.compute(q=q_current, qd=qd_current, t=t_current)
    tau_no_ff = controller_no_ff.compute(q=q_current, qd=qd_current, t=t_current)
    
    print(f"  With feedforward: τ magnitude = {np.linalg.norm(tau_ff)}")
    print(f"  Without feedforward: τ magnitude = {np.linalg.norm(tau_no_ff)}")
    print(f"  Difference: {np.linalg.norm(tau_ff - tau_no_ff)}")
    print("  Note: Feedforward provides proactive compensation")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Cartesian Trajectory Controller Demo")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to robot model (URDF or MJCF)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='numpy',
        choices=['numpy', 'torch'],
        help='Backend to use (numpy or torch)'
    )
    args = parser.parse_args()
    
    rc.set_backend(args.backend)
    
    # Run demos
    demo_linear_trajectory()
    demo_circular_trajectory()
    demo_direct_mode()
    demo_feedforward_benefit()
    
    beauty_print("\n" + "=" * 70)
    beauty_print("Demo Complete!")
    beauty_print("=" * 70)


if __name__ == "__main__":
    main()

