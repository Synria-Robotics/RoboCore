#!/usr/bin/env python3
"""Cartesian Space Geometric Path Planning Examples with Inverse Kinematics

This demo demonstrates:
1. Linear motion (straight line) with IK solving
2. Circular motion (drawing a circle) with IK solving

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
from robocore.planning import (
    LinearPositionPlanner,
    SplineCurvePlanner,
    plot_cartesian_with_ik,
)
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.transform.se3 import make_transform
from robocore.transform.conversions import quaternion_to_matrix
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

    # Get chain joint indices
    chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
    for idx in chain_indices:
        js = robot_model.joint_list[idx]
        if js.joint_type == 'revolute':
            # For revolute joints, try to minimize the difference
            diff = q_new[idx] - q_prev[idx]

            # If difference is large, try adding/subtracting 2π
            if abs(diff) > np.pi:
                # Try subtracting 2π
                q_candidate = q_new[idx] - 2 * np.pi
                if js.limit_lower is not None and js.limit_upper is not None:
                    if js.limit_lower <= q_candidate <= js.limit_upper:
                        if abs(q_candidate - q_prev[idx]) < abs(diff):
                            q_normalized[idx] = q_candidate
                            continue

                # Try adding 2π
                q_candidate = q_new[idx] + 2 * np.pi
                if js.limit_lower is not None and js.limit_upper is not None:
                    if js.limit_lower <= q_candidate <= js.limit_upper:
                        if abs(q_candidate - q_prev[idx]) < abs(diff):
                            q_normalized[idx] = q_candidate

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


def demo_linear_motion(robot_model, args):
    """Example: Linear Motion (Straight Line) with IK solving"""
    beauty_print("[1] Linear Motion (Straight Line)", type="module", centered=False)

    # Use numpy backend for smooth spline interpolation
    rc.set_backend('numpy')
    planner = LinearPositionPlanner()

    # Define start and end poses (position + quaternion)
    # Based on joint angle analysis: positions within typical workspace range
    p_start = np.array([0.25, 0.25, +0.2])  # Start position in workspace
    p_end = np.array([0.25, 0.25, +0.5])     # End position in workspace
    q_start = np.array([1.0, 0.0, 0.0, 0.0])  # Start quaternion (identity rotation)
    q_end = np.array([1.0, 0.0, 0.0, 0.0])    # End quaternion (identity rotation)

    result = planner.plan(
        start=p_start,
        end=p_end,
        duration=args.duration,
        num_points=args.num_points
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Start position: {beauty_print_array(p_start)}")
    print(f"  Start quaternion: {beauty_print_array(q_start)}")
    print(f"  End position: {beauty_print_array(p_end)}")
    print(f"  End quaternion: {beauty_print_array(q_end)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Distance: {np.linalg.norm(p_end - p_start):.3f} m")
    print(f"  Max velocity: {np.max(np.linalg.norm(result['velocities'], axis=1)):.3f} m/s")
    beauty_print("  ✓ Straight-line motion")

    # Build poses from positions and quaternions
    positions = result['positions']
    num_points = len(positions)
    target_poses = np.zeros((num_points, 4, 4))
    for i in range(num_points):
        # Convert quaternion to rotation matrix (use start quaternion for all points)
        R = quaternion_to_matrix(q_start)
        target_poses[i] = make_transform(R, positions[i])

    # Switch to specified backend for IK computation
    if args.backend != 'numpy':
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

    # Extract joint angles from IK results
    joint_angles = []
    success_count = 0
    for result_ik in ik_results:
        if result_ik['success']:
            joint_angles.append(result_ik['q'])
            success_count += 1
        else:
            if len(joint_angles) > 0:
                joint_angles.append(joint_angles[-1])
            else:
                chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
                joint_angles.append(np.zeros(len(chain_indices)))

    joint_angles = np.array(joint_angles)
    beauty_print(f"IK Success Rate: {success_count}/{len(ik_results)} ({success_count/len(ik_results)*100:.1f}%)")

    # Store results
    result['joint_angles'] = joint_angles
    result['ik_results'] = ik_results
    result['waypoints'] = np.array([p_start, p_end])

    return result


def demo_circular_motion(robot_model, args):
    """Example: Circular Motion (Drawing a Circle) with IK solving"""
    beauty_print("[2] Circular Motion (Drawing a Circle)", type="module", centered=False)

    # Use numpy backend for smooth spline interpolation
    rc.set_backend('numpy')
    planner = SplineCurvePlanner()

    # Define circle parameters (estimated from valid joint angles in workspace)
    # Based on joint angle analysis: center and radius within typical workspace range
    center = np.array([0.40, 0.0, 0.275])  # Circle center in workspace
    radius = 0.08  # Circle radius (smaller to ensure all points in workspace)
    num_waypoints = 8  # Number of waypoints around the circle

    # Generate waypoints on a circle in XY plane
    waypoints = []
    for i in range(num_waypoints):
        angle = 2 * np.pi * i / num_waypoints
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        waypoints.append(np.array([x, y, z]))

    # Add first point at the end to close the circle
    waypoints.append(waypoints[0])

    waypoints = np.array(waypoints)

    # Define orientations (keep constant, pointing upward)
    R = np.eye(3)  # Identity rotation

    # Create transformation matrices
    transforms = np.array([make_transform(R, p) for p in waypoints])

    result = planner.plan(
        waypoints=transforms,
        duration=args.duration,
        num_points=args.num_points
    )

    beauty_print(f"Trajectory generated:")
    print(f"  Circle center: {beauty_print_array(center)}")
    print(f"  Circle radius: {radius:.3f} m")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration: {result['t'][-1]:.3f} s")
    print(f"  Points: {len(result['t'])}")
    print(f"  Max linear velocity: {np.max(np.linalg.norm(result['velocities'][:, :3], axis=1)):.3f} m/s")
    print(f"  Max angular velocity: {np.max(np.linalg.norm(result['velocities'][:, 3:], axis=1)):.3f} rad/s")
    beauty_print("  ✓ Circular path")

    # Extract poses from trajectory
    if 'poses' in result:
        target_poses = result['poses']  # Shape: (num_points, 4, 4)
    else:
        # Build poses from positions and orientations
        positions = result['positions']
        orientations = result.get('orientations', np.tile(R, (len(positions), 1, 1)))
        num_points = len(positions)
        target_poses = np.zeros((num_points, 4, 4))
        for i in range(num_points):
            if len(orientations.shape) == 3:
                target_poses[i] = make_transform(orientations[i], positions[i])
            else:
                target_poses[i] = make_transform(R, positions[i])

    # Switch to specified backend for IK computation
    if args.backend != 'numpy':
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

    # Extract joint angles from IK results
    joint_angles = []
    success_count = 0
    for result_ik in ik_results:
        if result_ik['success']:
            joint_angles.append(result_ik['q'])
            success_count += 1
        else:
            if len(joint_angles) > 0:
                joint_angles.append(joint_angles[-1])
            else:
                chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
                joint_angles.append(np.zeros(len(chain_indices)))

    joint_angles = np.array(joint_angles)
    beauty_print(f"IK Success Rate: {success_count}/{len(ik_results)} ({success_count/len(ik_results)*100:.1f}%)")

    # Store results
    result['joint_angles'] = joint_angles
    result['ik_results'] = ik_results
    result['waypoints'] = waypoints

    return result




def main(args):
    """Main function to run geometric path planning examples with IK"""
    beauty_print("Cartesian Space Geometric Path Planning Examples with IK", type="module")

    # Load robot model
    # Use numpy backend for smooth spline interpolation
    rc.set_backend('numpy')
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)

    # Switch to specified backend for IK computation
    if args.backend != 'numpy':
        beauty_print(f"Switching to {args.backend} backend for IK computation...")
        rc.set_backend(args.backend, device=args.device)

    beauty_print(f"Robot Model: {args.model_path}")
    beauty_print(f"Base Link: {args.base_link}, End Link: {args.end_link}")
    chain_indices = robot_model._get_joint_indices(args.base_link, args.end_link)
    beauty_print(f"Total DOF (unified config space): {robot_model.num_dof}")
    beauty_print(f"Chain DOF: {len(chain_indices)}")

    results = {}

    # Run all examples
    results['linear'] = demo_linear_motion(robot_model, args)
    results['circular'] = demo_circular_motion(robot_model, args)

    # Display error statistics for successful IK solutions
    for name in ['linear', 'circular']:
        if name not in results or results[name] is None:
            continue
        result = results[name]
        if 'ik_results' in result:
            ik_results = result['ik_results']
            success_count = sum(1 for r in ik_results if r['success'])
            if success_count > 0:
                pos_errors = [r['pos_err'] for r in ik_results if r['success']]
                ori_errors = [r['ori_err'] for r in ik_results if r['success']]
                beauty_print(f"{name.capitalize()} Error Statistics (successful cases):")
                print(f"  Position error - Mean: {np.mean(pos_errors):.6e} m, Max: {np.max(pos_errors):.6e} m")
                print(f"  Orientation error - Mean: {np.mean(ori_errors):.6e} rad, Max: {np.max(ori_errors):.6e} rad")

    beauty_print("✓ All examples completed!", type="success")

    # Plot trajectories
    try:
        import matplotlib.pyplot as plt

        # Plot linear motion
        if 'linear' in results and results['linear'] is not None:
            result_linear = results['linear']
            waypoints_linear = result_linear.get('waypoints', None)
            if waypoints_linear is not None and len(waypoints_linear.shape) == 2:
                # Convert position waypoints to transform matrices
                transforms_linear = np.array([make_transform(np.eye(3), p) for p in waypoints_linear])
            else:
                transforms_linear = waypoints_linear

            fig_linear = plot_cartesian_with_ik(
                result_linear,
                transforms_linear,
                result_linear['joint_angles'],
                result_linear['ik_results'],
                figsize=(18, 6)
            )
            if fig_linear is not None:
                fig_linear.suptitle('Linear Motion', fontsize=14, y=0.98)

        # Plot circular motion
        if 'circular' in results and results['circular'] is not None:
            result_circular = results['circular']
            waypoints_circular = result_circular.get('waypoints', None)
            if waypoints_circular is not None and len(waypoints_circular.shape) == 2:
                # Convert position waypoints to transform matrices
                transforms_circular = np.array([make_transform(np.eye(3), p) for p in waypoints_circular])
            else:
                transforms_circular = waypoints_circular

            fig_circular = plot_cartesian_with_ik(
                result_circular,
                transforms_circular,
                result_circular['joint_angles'],
                result_circular['ik_results'],
                figsize=(18, 6)
            )
            if fig_circular is not None:
                fig_circular.suptitle('Circular Motion', fontsize=14, y=0.98)

        plt.show()
    except ImportError:
        beauty_print("matplotlib not installed. Skipping plots.", type="warning")

    return results


if __name__ == '__main__':
    import synriard

    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description='Cartesian Space Geometric Path Planning with IK')
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='world', help='Base link name')
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
                        help='Backend (default: numpy)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    main(args)

