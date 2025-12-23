#!/usr/bin/env python3
"""Cartesian Spline Trajectory Planning with Inverse Kinematics

This demo demonstrates:
1. Generating a smooth spline trajectory in Cartesian space through multiple waypoints
2. Solving inverse kinematics for all poses in the trajectory (batch IK)
3. Outputting joint angles for the entire trajectory

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
import argparse
import time

import robocore as rc
from robocore.modeling import RobotModel
from robocore.planning import SplineCurvePlanner, plot_cartesian_with_ik
from robocore.kinematics.ik import inverse_kinematics
from robocore.transform.se3 import make_transform
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def normalize_joint_angles(q_new, q_prev, robot_model):
    """Normalize joint angles to minimize discontinuity with previous configuration.

    For revolute joints, add/subtract 2π to keep angles close to previous configuration.
    This helps maintain continuity when joints wrap around.

    :param q_new: New joint configuration
    :param q_prev: Previous joint configuration (can be None)
    :param robot_model: RobotModel instance
    :return: Normalized joint configuration
    """
    q_new = np.asarray(q_new)
    if q_prev is None:
        return q_new

    q_prev = np.asarray(q_prev)
    q_normalized = q_new.copy()

    for js in robot_model._chain_actuated:
        if js.joint_type == 'revolute':
            # For revolute joints, try to minimize the difference
            diff = q_new[js.index] - q_prev[js.index]

            # If difference is large, try adding/subtracting 2π
            if abs(diff) > np.pi:
                # Try subtracting 2π
                q_candidate = q_new[js.index] - 2 * np.pi
                if js.limit_lower is not None and js.limit_upper is not None:
                    if js.limit_lower <= q_candidate <= js.limit_upper:
                        if abs(q_candidate - q_prev[js.index]) < abs(diff):
                            q_normalized[js.index] = q_candidate
                            continue

                # Try adding 2π
                q_candidate = q_new[js.index] + 2 * np.pi
                if js.limit_lower is not None and js.limit_upper is not None:
                    if js.limit_lower <= q_candidate <= js.limit_upper:
                        if abs(q_candidate - q_prev[js.index]) < abs(diff):
                            q_normalized[js.index] = q_candidate

    return q_normalized


def solve_ik_simple(robot_model, target_pose, q0, args):
    """Solve IK using previous solution as initial guess (simple approach like InteractiveDualArmIK).

    :param robot_model: RobotModel instance
    :param target_pose: Target pose (4x4 matrix)
    :param q0: Previous joint configuration (for continuity)
    :param args: Command line arguments
    :return: IK result dictionary
    """
    if q0 is None:
        # First pose, use normal IK
        result = inverse_kinematics(
            robot_model,
            target_pose,
            q0=None,
            method=args.method,
            max_iters=args.max_iters,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            num_initial_guesses=1,
            initial_guess_strategy='zero',
            initial_guess_scale=args.init_scale,
            random_seed=None,
        )
        return result

    # Always use previous solution as initial guess (ensures continuity)
    result = inverse_kinematics(
        robot_model,
        target_pose,
        q0=q0,
        method=args.method,
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        num_initial_guesses=1,
        initial_guess_strategy='zero',
        initial_guess_scale=args.init_scale,
        random_seed=None,
    )

    # Normalize joint angles for revolute joints to maintain continuity
    if result['success']:
        q_normalized = normalize_joint_angles(result['q'], q0, robot_model)
        result['q'] = q_normalized.tolist()

    return result


def main(args):
    # Load robot model
    beauty_print("Cartesian Spline Planning with IK Batch Solver", type="module")

    # Use numpy backend for smooth spline interpolation
    # (torch backend uses linear interpolation which is not smooth)
    rc.set_backend('numpy')
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)

    # Switch to specified backend for IK computation
    if args.backend != 'numpy':
        beauty_print(f"Switching to {args.backend} backend for IK computation...")
        rc.set_backend(args.backend, device=args.device)

    beauty_print(f"Robot Model: {args.model_path}")
    beauty_print(f"Base Link: {args.base_link}, End Link: {args.end_link}")
    beauty_print(f"DOF: {len(robot_model._chain_actuated)}")

    # [1] Generate spline trajectory in Cartesian space
    beauty_print("[1] Generating Spline Trajectory", type="module", centered=False)

    # Use numpy backend for smooth cubic spline interpolation
    # (torch backend uses linear interpolation which is not smooth)
    rc.set_backend('numpy')
    planner = SplineCurvePlanner()

    # Generate random poses within robot workspace using random_pose_batch
    beauty_print(f"Generating {args.num_waypoints} random waypoints within workspace...")
    waypoints_list = robot_model.random_pose_batch(
        batch_size=args.num_waypoints,
        seed=args.seed,
        scale=args.workspace_scale
    )

    # Convert to numpy array if needed
    waypoints = np.array([to_numpy(wp) for wp in waypoints_list])

    beauty_print(f"Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        pos = wp[:3, 3]
        print(f"  Waypoint {i+1}: position = {beauty_print_array(pos)}")

    # Generate spline trajectory
    trajectory = planner.plan(
        waypoints=waypoints,
        duration=args.duration,
        num_points=args.num_points
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Duration: {trajectory['t'][-1]:.3f} s")
    print(f"  Points: {len(trajectory['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(trajectory['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(trajectory['velocities'][:, 3:], axis=1)):.3f} rad/s")

    # Extract waypoint positions for plotting
    waypoint_positions = np.array([wp[:3, 3] for wp in waypoints])
    trajectory['waypoints'] = waypoint_positions

    # Extract poses from trajectory
    if 'poses' in trajectory:
        target_poses = trajectory['poses']  # Shape: (num_points, 4, 4)
    else:
        # Build poses from positions and orientations
        positions = trajectory['positions']
        orientations = trajectory['orientations']
        num_points = len(positions)
        target_poses = np.zeros((num_points, 4, 4))
        for i in range(num_points):
            target_poses[i] = make_transform(orientations[i], positions[i])

    # Switch to specified backend for IK computation
    if args.backend != 'numpy':
        beauty_print(f"Switching to {args.backend} backend for IK computation...")
        rc.set_backend(args.backend, device=args.device)

    # [2] Solve IK sequentially using previous solution as initial guess
    # This ensures joint angle continuity (standard approach for Cartesian trajectories)
    beauty_print("[2] Solving Inverse Kinematics (Sequential)", type="module", centered=False)

    # Display sample target pose for debugging
    beauty_print(f"Sample target pose (first point):")
    sample_pose = target_poses[0]
    print(f"  Position: {beauty_print_array(sample_pose[:3, 3])}")
    print(f"  Rotation matrix:\n{sample_pose[:3, :3]}")

    beauty_print(f"Solving IK sequentially for {len(target_poses)} poses...")
    beauty_print(f"  Using previous solution as initial guess (ensures continuity)")

    # Get initial joint configuration for first pose
    if args.use_random_init:
        rng = np.random.default_rng(args.seed)
        q0 = to_numpy(robot_model.random_q(rng, scale=args.init_scale))
    else:
        q0 = None  # Will use zero or default

    start_time = time.time()
    ik_results = []

    for i, target_pose in enumerate(target_poses):
        # Use simple IK solver (always uses previous solution as initial guess)
        result = solve_ik_simple(
            robot_model,
            target_pose,
            q0,
            args
        )

        ik_results.append(result)
        if q0 is not None:
            q_diff = np.linalg.norm(np.array(result['q']) - np.array(q0))
            print(f"Point {i+1}: Success={result['success']}, Joint angle change: {q_diff:.6f}")
        else:
            print(f"Point {i+1}: Success={result['success']}")
        print("q: ", result['q'])
        print("pos_err: ", result['pos_err'])
        print("ori_err: ", result['ori_err'])
        print("iters: ", result['iters'])
        print("--------------------------------")

        # This maintains continuity even if some poses fail
        q0 = to_numpy(result['q'])

    ik_time = time.time() - start_time

    beauty_print(f"IK sequential computation completed:")
    print(f"  Total time: {ik_time:.4f} s")
    print(f"  Average time per pose: {ik_time/len(target_poses)*1000:.4f} ms")

    # Check first few results for debugging
    beauty_print(f"First 3 IK results:")
    for i in range(min(3, len(ik_results))):
        result = ik_results[i]
        print(f"  Point {i+1}: success={result['success']}, "
              f"pos_err={result.get('pos_err', 0.0):.6e}, "
              f"ori_err={result.get('ori_err', 0.0):.6e}, "
              f"iters={result.get('iters', 0)}")

    # [3] Extract and display joint angles
    beauty_print("[3] Joint Angle Trajectory", type="module", centered=False)

    # Extract joint angles from IK results
    joint_angles = []
    success_count = 0

    for i, result in enumerate(ik_results):
        if result['success']:
            joint_angles.append(result['q'])
            success_count += 1
        else:
            # Use previous solution if available, otherwise use zeros
            if len(joint_angles) > 0:
                joint_angles.append(joint_angles[-1])
            else:
                joint_angles.append(np.zeros(len(robot_model._chain_actuated)))

    joint_angles = np.array(joint_angles)  # Shape: (num_points, n_dof)

    beauty_print(f"IK Success Rate: {success_count}/{len(ik_results)} ({success_count/len(ik_results)*100:.1f}%)")

    # Display joint angle statistics
    beauty_print(f"Joint Angle Statistics:")
    print(f"  Shape: {joint_angles.shape}")
    print(f"  Min values: {beauty_print_array(np.min(joint_angles, axis=0))}")
    print(f"  Max values: {beauty_print_array(np.max(joint_angles, axis=0))}")
    print(f"  Mean values: {beauty_print_array(np.mean(joint_angles, axis=0))}")

    # Display sample joint angles
    beauty_print(f"Sample Joint Angles (first 5 points):")
    for i in range(min(5, len(joint_angles))):
        print(f"  Point {i+1} (t={trajectory['t'][i]:.3f}s): {beauty_print_array(joint_angles[i])}")

    if len(joint_angles) > 5:
        beauty_print(f"Sample Joint Angles (last 5 points):")
        for i in range(max(0, len(joint_angles)-5), len(joint_angles)):
            print(f"  Point {i+1} (t={trajectory['t'][i]:.3f}s): {beauty_print_array(joint_angles[i])}")

    # Display error statistics for successful IK solutions
    if success_count > 0:
        pos_errors = [r['pos_err'] for r in ik_results if r['success']]
        ori_errors = [r['ori_err'] for r in ik_results if r['success']]

        beauty_print(f"Error Statistics (successful cases):")
        print(f"  Position error - Mean: {np.mean(pos_errors):.6e} m, Max: {np.max(pos_errors):.6e} m")
        print(f"  Orientation error - Mean: {np.mean(ori_errors):.6e} rad, Max: {np.max(ori_errors):.6e} rad")

    beauty_print("✓ Cartesian spline planning with IK batch solver completed!", type="success")

    # [4] Plot trajectory
    try:
        import matplotlib.pyplot as plt
        plot_cartesian_with_ik(trajectory, waypoints, joint_angles, ik_results)
        plt.show()
    except ImportError:
        beauty_print("matplotlib not installed. Skipping plots.", type="warning")

    return {
        'trajectory': trajectory,
        'joint_angles': joint_angles,
        'ik_results': ik_results,
        'success_rate': success_count / len(ik_results),
        'waypoints': waypoints
    }


if __name__ == '__main__':
    import synriard

    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description='Cartesian Spline Planning with IK Batch Solver')
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--duration', type=float, default=3.0, help='Trajectory duration in seconds')
    parser.add_argument('--num-points', type=int, default=150, help='Number of points in trajectory')
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'],
                        help='IK method (default: dls)')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum IK iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-2, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-2, help='Orientation tolerance (rad)')
    parser.add_argument('--num-initial-guesses', type=int, default=5,
                        help='Number of initial guesses for failed poses (default: 5)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Initial guess strategy for failed poses (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale factor for initial guesses')
    parser.add_argument('--use-random-init', action='store_true',
                        help='Use random initial configuration for first pose (default: use zero)')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                        help='Backend (default: torch)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--num-waypoints', type=int, default=5, help='Number of waypoints (default: 5)')
    parser.add_argument('--workspace-scale', type=float, default=0.6,
                        help='Workspace scale factor for random poses (0.0 to 1.0, default: 0.6)')
    args = parser.parse_args()

    main(args)
