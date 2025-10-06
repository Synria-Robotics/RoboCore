#!/usr/bin/env python3
"""MuJoCo Visualization for Dual-Arm Cooperative Operations

Copyright (c) 2025 Synria Robotics Co., Ltd.

Demonstrates hierarchical dual-arm IK with MuJoCo visualization:
- Cooperative carry with grasp maintenance
- Smooth trajectory interpolation
- Real-time simulation playback

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import os
import numpy as np
import sys

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import dual_ik_hierarchical, Task, relative_pose_error
from robocore.planning.trajectory import multi_waypoint_trajectory
from robocore.utils.path import get_robocore_path
from robocore.bridge.sim.mujoco.trajectory_visualizer import TrajectoryVisualizer


def cooperative_carry_scenario():
    """
    Scenario: Cooperative carry with grasp maintenance.
    
    Demonstrates:
    1. Start from zero configuration (home position)
    2. Reach to grasp an object (virtual box)
    3. Lift the object cooperatively
    4. Move object forward, sideways, and rotate
    5. Return to initial grasp position
    6. Maintain relative pose (grasp) throughout motion
    """
    
    print("=" * 70)
    print("  MuJoCo Dual-Arm Cooperative Carry Visualization")
    print("=" * 70)
    
    # ================================================================
    # Step 1: Load robot models
    # ================================================================
    print("\n[1/6] Loading robot models...")
    
    model_path = get_robocore_path("assets/robot/mjcf/Bessica-D_v1_0/Bessica-D_Covered.xml")
    
    left_model = RobotModel(model_path, end_link="left_arm_link7")
    left_model.print_tree()
    right_model = RobotModel(model_path, end_link="right_arm_link7")
    right_model.print_tree()
    
    print(f"✓ Left arm: {left_model.num_dof()} DOF")
    print(f"✓ Right arm: {right_model.num_dof()} DOF")
    
    # ================================================================
    # Step 2: Start from zero configuration
    # ================================================================
    print("\n[2/6] Starting from zero (home) configuration...")
    
    # Zero configuration - both arms at neutral position
    q_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    T_left_zero = left_model.fk(q_zero, return_end=True)
    T_right_zero = right_model.fk(q_zero, return_end=True)
    
    # Convert to numpy if needed
    if hasattr(T_left_zero, 'numpy'):
        T_left_zero = T_left_zero.numpy()
    if hasattr(T_right_zero, 'numpy'):
        T_right_zero = T_right_zero.numpy()
    
    p_left_zero = T_left_zero[:3, 3]
    p_right_zero = T_right_zero[:3, 3]
    
    print(f"Left gripper at zero:  [{p_left_zero[0]:.3f}, {p_left_zero[1]:.3f}, {p_left_zero[2]:.3f}] m")
    print(f"Right gripper at zero: [{p_right_zero[0]:.3f}, {p_right_zero[1]:.3f}, {p_right_zero[2]:.3f}] m")
    
    # ================================================================
    # Step 3: Define grasp configuration (reach to virtual object)
    # ================================================================
    print("\n[3/6] Planning reach-to-grasp configuration...")
    
    # # Virtual object center position (in front of robot)
    # object_center = np.array([0.0, 0.3, 0.45])  # 30cm forward, 45cm height
    # object_width = 0.25  # 25cm wide object
    
    # # Grasp points: left and right sides of object
    # grasp_left_pos = object_center + np.array([object_width/2, 0.0, 0])
    # grasp_right_pos = object_center + np.array([-object_width/2, 0.0, 0])
    
    # print(f"Object center: [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}] m")
    # print(f"Object width: {object_width*100:.0f} cm")
    # print(f"Left grasp point:  [{grasp_left_pos[0]:.3f}, {grasp_left_pos[1]:.3f}, {grasp_left_pos[2]:.3f}] m")
    # print(f"Right grasp point: [{grasp_right_pos[0]:.3f}, {grasp_right_pos[1]:.3f}, {grasp_right_pos[2]:.3f}] m")
    
    # Use manually tuned grasp configurations for reliability
    # These are pre-validated poses that work well
    print("Using pre-configured grasp joint angles...")
    q_grasp_left = np.array([0.8, 0, 0.0, -0.45, 0., 0.5, 1])
    q_grasp_right = np.array([0.8, 0, 0.0, -0.45, 0., 0.5, -1])
    
    # Compute actual grasp poses
    T_left_grasp = left_model.fk(q_grasp_left, return_end=True)
    T_right_grasp = right_model.fk(q_grasp_right, return_end=True)
    
    # Convert to numpy if needed
    if hasattr(T_left_grasp, 'numpy'):
        T_left_grasp = T_left_grasp.numpy()
    if hasattr(T_right_grasp, 'numpy'):
        T_right_grasp = T_right_grasp.numpy()
    
    # Record relative pose (this must be maintained)
    T_rel_grasp = np.linalg.inv(T_left_grasp) @ T_right_grasp
    
    p_left = T_left_grasp[:3, 3]
    p_right = T_right_grasp[:3, 3]
    grasp_dist = np.linalg.norm(p_right - p_left)
    
    print(f"Left gripper:  [{p_left[0]:.3f}, {p_left[1]:.3f}, {p_left[2]:.3f}] m")
    print(f"Right gripper: [{p_right[0]:.3f}, {p_right[1]:.3f}, {p_right[2]:.3f}] m")
    print(f"Grasp distance: {grasp_dist*1000:.1f} mm")
    
    # ================================================================
    # Step 4: Plan meaningful cooperative carry trajectory
    # ================================================================
    print("\n[4/6] Planning cooperative carry trajectory...")
    print("Scenario: Pick object from front → Lift → Move forward → Rotate → Return")
    
    # Waypoint descriptions and offsets from grasp position
    waypoint_plan = [
        ("Initial grasp", [0.00, 0.0, 0.00]),
        ("Lift up 5cm", [0.00, 0.1, 0.0]),
        ("Move forward 5cm", [0.0, 0.0, 0.1]),
        ("Move right 3cm", [0.0, 0.0, -0.1]),
        ("Move left 3cm", [0.0, -0.1, 0.0]),
    ]
    
    print(f"Planned {len(waypoint_plan)} waypoints:")
    for i, (desc, offset) in enumerate(waypoint_plan):
        print(f"  {i}. {desc:20s} Δ=[{offset[0]:+.2f}, {offset[1]:+.2f}, {offset[2]:+.2f}]m")
    
    # ================================================================
    # Step 5: Solve hierarchical IK for all waypoints
    # ================================================================
    print("\n[5/6] Solving hierarchical IK for trajectory...")
    
    waypoints_dual = []  # Store (q_left, q_right) pairs
    
    q_current_left = q_grasp_left.copy()
    q_current_right = q_grasp_right.copy()
    
    for i, (desc, offset) in enumerate(waypoint_plan):
        # Target pose for left arm
        T_left_target = T_left_grasp.copy()
        T_left_target[:3, 3] += offset
        
        # Leader-first strategy: solve left arm single-arm IK first, then
        # compute the corresponding right-arm absolute target and solve right.
        # This enforces the left arm as the leader.
        # Single-arm IK kwargs (keep consistent with hierarchical solver tolerances)
        ik_kwargs = dict(max_iters=100, pos_tol=5e-3, ori_tol=1e-3, method='dls')

        # Solve left arm single-arm IK to reach T_left_target
        tgt_left = T_left_target.tolist() if hasattr(T_left_target, 'tolist') else T_left_target
        res_left = left_model.ik(tgt_left, q_initial=q_current_left, **ik_kwargs)

        if res_left.get('success', False):
            q_sol_left = np.array(res_left['q'])
        else:
            print(f"  {i}. {desc:20s} LEFT IK FAILED ❌")
            # use best available solution if provided, else keep current
            q_sol_left = np.array(res_left.get('q', q_current_left))

        # Compute left FK from solved joints (numpy conversion if needed)
        T_L = left_model.fk(q_sol_left, return_end=True)
        if hasattr(T_L, 'numpy'):
            T_L = T_L.numpy()

        # Build right arm absolute target from leader's pose and desired relative
        T_right_target = T_L @ T_rel_grasp

        # Solve right arm single-arm IK to reach computed T_right_target
        tgt_right = T_right_target.tolist() if hasattr(T_right_target, 'tolist') else T_right_target
        res_right = right_model.ik(tgt_right, q_initial=q_current_right, **ik_kwargs)

        if res_right.get('success', False):
            q_sol_right = np.array(res_right['q'])
        else:
            print(f"  {i}. {desc:20s} RIGHT IK FAILED ❌")
            q_sol_right = np.array(res_right.get('q', q_current_right))

        # Record and validate
        q_current_left = q_sol_left
        q_current_right = q_sol_right
        waypoints_dual.append((q_current_left.copy(), q_current_right.copy()))

        # Validate grasp maintenance
        T_R = right_model.fk(q_current_right, return_end=True)
        if hasattr(T_R, 'numpy'):
            T_R = T_R.numpy()

        rel_error = relative_pose_error(T_L, T_R, T_rel_grasp)
        pos_error = np.linalg.norm(rel_error[:3]) * 1000  # mm
        status = "✓" if pos_error < 5.0 else "⚠️"
        print(f"  {i}. {desc:20s} error={pos_error:5.2f}mm {status}")
    
    print(f"\n✓ Generated {len(waypoints_dual)} waypoints")
    
    # ================================================================
    # Step 6: Generate smooth trajectories
    # ================================================================
    print("\n[6/6] Generating smooth trajectories...")
    
    # Separate left and right arm waypoints
    waypoints_left = np.array([wp[0] for wp in waypoints_dual])
    waypoints_right = np.array([wp[1] for wp in waypoints_dual])
    
    # Generate trajectories (quintic polynomial)
    segment_duration = 1.5  # seconds per segment (faster pacing)
    points_per_segment = 80  # fewer points for smoother playback
    
    t_left, q_left_traj, qd_left, qdd_left = multi_waypoint_trajectory(
        waypoints_left,
        durations=segment_duration,
        num_points_per_segment=points_per_segment,
        method='quintic'
    )
    
    t_right, q_right_traj, qd_right, qdd_right = multi_waypoint_trajectory(
        waypoints_right,
        durations=segment_duration,
        num_points_per_segment=points_per_segment,
        method='quintic'
    )
    
    print(f"✓ Left arm trajectory: {len(q_left_traj)} points, {t_left[-1]:.1f}s duration")
    print(f"✓ Right arm trajectory: {len(q_right_traj)} points, {t_right[-1]:.1f}s duration")
    
    # Combine dual-arm trajectory BEFORE adding zero config
    q_dual_traj = np.hstack([q_left_traj, q_right_traj])
    
    # Validate grasp maintenance throughout trajectory
    max_rel_error = 0.0
    for i in range(0, len(q_left_traj), 50):  # Sample every 50 points
        T_L = left_model.fk(q_left_traj[i], return_end=True)
        T_R = right_model.fk(q_right_traj[i], return_end=True)
        
        # Convert to numpy if needed
        if hasattr(T_L, 'numpy'):
            T_L = T_L.numpy()
        if hasattr(T_R, 'numpy'):
            T_R = T_R.numpy()
        
        rel_error = relative_pose_error(T_L, T_R, T_rel_grasp)
        pos_error = np.linalg.norm(rel_error[:3]) * 1000
        max_rel_error = max(max_rel_error, pos_error)
    
    print(f"✓ Max grasp error during trajectory: {max_rel_error:.2f}mm")
    
    # Add transition from zero to grasp, then trajectory, then back to zero
    print("\nAdding complete motion sequence:")
    print("  1. Home position (1s hold)")
    print("  2. Reach to grasp (2s)")
    print("  3. Cooperative carry trajectory")
    print("  4. Return to home (2s)")
    
    # Generate smooth transition from zero to grasp
    t_reach, q_reach_left, _, _ = multi_waypoint_trajectory(
        np.array([q_zero, q_grasp_left]),
        durations=2.0,
        num_points_per_segment=100,
        method='quintic'
    )
    
    t_reach_right, q_reach_right, _, _ = multi_waypoint_trajectory(
        np.array([q_zero, q_grasp_right]),
        durations=2.0,
        num_points_per_segment=100,
        method='quintic'
    )
    
    # Generate smooth return from grasp to zero
    t_return, q_return_left, _, _ = multi_waypoint_trajectory(
        np.array([q_grasp_left, q_zero]),
        durations=2.0,
        num_points_per_segment=100,
        method='quintic'
    )
    
    t_return_right, q_return_right, _, _ = multi_waypoint_trajectory(
        np.array([q_grasp_right, q_zero]),
        durations=2.0,
        num_points_per_segment=100,
        method='quintic'
    )
    
    # Combine all trajectories
    q_zero_dual = np.hstack([q_zero, q_zero])
    q_reach_dual = np.hstack([q_reach_left, q_reach_right])
    q_return_dual = np.hstack([q_return_left, q_return_right])
    
    q_dual_traj_full = np.vstack([
        np.tile(q_zero_dual, (50, 1)),   # 1. Hold at zero for 1 second
        q_reach_dual,                     # 2. Reach to grasp (2s)
        q_dual_traj,                      # 3. Carry trajectory
        q_return_dual,                    # 4. Return to zero (2s)
        np.tile(q_zero_dual, (50, 1)),   # 5. Hold at zero for 1 second
    ])
    
    # Build corresponding time vector
    t_total = np.concatenate([
        np.linspace(0, 1.0, 50),                      # Hold 1s
        t_reach + 1.0,                                # Reach 2s
        t_left + 3.0,                                 # Carry trajectory
        t_return + 3.0 + t_left[-1],                  # Return 2s
        np.linspace(5.0 + t_left[-1] + t_return[-1], 
                   6.0 + t_left[-1] + t_return[-1], 50)  # Hold 1s
    ])
    
    print(f"✓ Complete sequence: {len(q_dual_traj_full)} points, {t_total[-1]:.1f}s total duration")
    
    # ================================================================
    # Step 7: MuJoCo Visualization
    # ================================================================
    print("\n[7/7] Launching MuJoCo visualization...")
    
    # Find MJCF file
    mjcf_path = get_robocore_path('assets/robot/mjcf/Bessica-D_v1_0/Bessica-D_Covered.xml')
    print(f"✓ Using MJCF: {mjcf_path}")
    
    # Initialize visualizer
    viz = TrajectoryVisualizer(mjcf_path=str(mjcf_path))
    
    # Load complete trajectory sequence
    viz.load_trajectory(
        q_trajectory=q_dual_traj_full,
        time=t_total
    )
    
    print("\n" + "=" * 70)
    print("  Visualization Controls:")
    print("=" * 70)
    print("  Space     - Play/Pause")
    print("  R         - Reset")
    print("  [ ]       - Decrease/Increase speed")
    print("  , .       - Step backward/forward")
    print("  L         - Toggle loop")
    print("  ESC       - Close viewer")
    print("=" * 70)
    
    # Visualize
    viz.visualize(title="Dual-Arm Cooperative Carry")
    
    print("\n✓ Visualization complete!")


def main():
    cooperative_carry_scenario()



if __name__ == '__main__':
    sys.exit(main())
