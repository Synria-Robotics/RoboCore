"""Forward Kinematics Parallel Demo

This demo demonstrates parallel/batch forward kinematics computation and compares
NumPy, Torch, and C++ backend batch timing (mirrors 01a_demo_fk.py for single-FK).

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import *


def compute_fk_batch(robot_model, backend, joint_angles_batch):
    """Compute batch forward kinematics for given backend.

    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param joint_angles_batch: Joint angles per row, shape (B, n_dof) or list of configs
    :return: Dictionary with batch transforms and wall-clock time for the FK call
    """
    rc.set_backend(backend)
    q_batch = np.asarray(joint_angles_batch, dtype=np.float64)
    if q_batch.ndim == 1:
        q_batch = q_batch.reshape(1, -1)

    start_time = time.time()
    T_fk = forward_kinematics(robot_model, q_batch, return_end=True)
    elapsed_time = time.time() - start_time

    return {
        'transform': T_fk,
        'time': elapsed_time,
    }


def _batch_positions(T):
    """Extract (B, 3) positions from batch transforms (B, 4, 4) or single (4, 4)."""
    Tn = to_numpy(T)
    if Tn.ndim == 2:
        return Tn[:3, 3]
    return Tn[:, :3, 3]


def _orientation_errors(T_ref, T_other, seq='XYZ'):
    """Max over batch of L2 norm of euler difference (radians)."""
    Ta = to_numpy(T_ref)
    Tb = to_numpy(T_other)
    if Ta.ndim == 2:
        Ta = Ta[np.newaxis, ...]
        Tb = Tb[np.newaxis, ...]
    max_e = 0.0
    for i in range(Ta.shape[0]):
        ea = matrix_to_euler(Ta[i, :3, :3], seq=seq)
        eb = matrix_to_euler(Tb[i, :3, :3], seq=seq)
        max_e = max(max_e, float(np.linalg.norm(to_numpy(ea) - to_numpy(eb))))
    return max_e


def _quat_errors(T_ref, T_other):
    Ta = to_numpy(T_ref)
    Tb = to_numpy(T_other)
    if Ta.ndim == 2:
        Ta = Ta[np.newaxis, ...]
        Tb = Tb[np.newaxis, ...]
    max_e = 0.0
    for i in range(Ta.shape[0]):
        qa = matrix_to_quaternion(Ta[i, :3, :3])
        qb = matrix_to_quaternion(Tb[i, :3, :3])
        max_e = max(max_e, float(np.linalg.norm(to_numpy(qa) - to_numpy(qb))))
    return max_e


def main(args):
    joint_configs = args.joint_angles
    num_batch = len(joint_configs)
    q_batch = np.asarray(joint_configs, dtype=np.float64)

    beauty_print(f"Batch forward kinematics: B={num_batch} configuration(s)")

    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    results_np = compute_fk_batch(robot_model, 'numpy', q_batch)
    results_torch = compute_fk_batch(robot_model, 'torch', q_batch)
    results_cpp = compute_fk_batch(robot_model, 'cpp', q_batch)

    pos_np = _batch_positions(results_np['transform'])
    pos_torch = _batch_positions(results_torch['transform'])
    pos_cpp = _batch_positions(results_cpp['transform'])

    if pos_np.ndim == 1:
        pos_norm_nt = np.linalg.norm(pos_np - pos_torch)
        pos_norm_nc = np.linalg.norm(pos_np - pos_cpp)
    else:
        pos_norm_nt = np.max(np.linalg.norm(pos_np - pos_torch, axis=1))
        pos_norm_nc = np.max(np.linalg.norm(pos_np - pos_cpp, axis=1))

    beauty_print("End-effector position agreement (max L2 over batch if B>1):")
    print(f"  np vs torch: {pos_norm_nt:.6e}   np vs cpp: {pos_norm_nc:.6e}")

    euler_max_nt = _orientation_errors(results_np['transform'], results_torch['transform'], seq='XYZ')
    euler_max_nc = _orientation_errors(results_np['transform'], results_cpp['transform'], seq='XYZ')
    beauty_print("Orientation (Euler XYZ, rad): max L2 over batch")
    print(f"  np vs torch: {euler_max_nt:.6e}   np vs cpp: {euler_max_nc:.6e}")

    quat_max_nt = _quat_errors(results_np['transform'], results_torch['transform'])
    quat_max_nc = _quat_errors(results_np['transform'], results_cpp['transform'])
    beauty_print("Orientation (quaternion xyzw): max L2 over batch")
    print(f"  np vs torch: {quat_max_nt:.6e}   np vs cpp: {quat_max_nc:.6e}")

    beauty_print("Batch computation time:")
    print(f"  NumPy:  {results_np['time']:.6f} seconds")
    print(f"  Torch:  {results_torch['time']:.6f} seconds")
    print(f"  C++:    {results_cpp['time']:.6f} seconds")
    tnp = max(results_np['time'], 1e-15)
    print(f"  torch/np: {results_torch['time'] / tnp:.2f}x   cpp/np: {results_cpp['time'] / tnp:.2f}x")
    if num_batch > 0:
        print(f"  NumPy avg per config: {results_np['time'] / num_batch:.6f} s")
        print(f"  Torch avg per config: {results_torch['time'] / num_batch:.6f} s")
        print(f"  C++ avg per config:   {results_cpp['time'] / num_batch:.6f} s")

    beauty_print("Results for each sample (NumPy backend)", type="module", centered=True)
    T_display = to_numpy(results_np['transform'])
    for i in range(num_batch):
        T_fk = T_display[i] if T_display.ndim == 3 else T_display
        q_config = joint_configs[i]

        beauty_print(f"Configuration {i + 1}:", type="info")
        print(f"  Joint angles: {beauty_print_array(np.array(q_config))}")

        position_fk = T_fk[:3, 3]
        rotation_fk = T_fk[:3, :3]
        euler_fk = matrix_to_euler(rotation_fk, seq='XYZ')
        quat_fk = matrix_to_quaternion(rotation_fk)

        print(f"  End-Effector Position (m):")
        print(f"     p = {beauty_print_array(position_fk)}")
        print(f"  End-Effector Orientation (Euler XYZ, degrees):")
        print(f"     rpy = {beauty_print_array(np.rad2deg(euler_fk))}")
        print(f"  End-Effector Orientation (Quaternion xyzw):")
        print(f"     quat = {beauty_print_array(quat_fk, precision=6)}")

        if args.show_matrices:
            print(f"  Homogeneous Transformation Matrix:")
            print(beauty_print_array(T_fk, precision=6))


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Forward Kinematics batch demo — NumPy / Torch / C++ timing (see also 01a_demo_fk.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 01b_demo_fk_parallel.py

        python 01b_demo_fk_parallel.py --joint-angles \\
            0.1 0.2 -0.3 0.0 0.5 -0.2 \\
            0.2 0.3 -0.4 0.1 0.6 -0.3

        python 01b_demo_fk_parallel.py --show-matrices
        """
    )
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+',
                        default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.4, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.5, -0.3, 0.0, 0.0, 0.2,
                                 0.5, 0.1, -0.9, 0.0, 0.2, -0.2,
                                 0.1, 0.2, -0.3, 0.7, 0.5, -0.2],
                        help='Joint angles in radians (flattened list, reshaped by --num-joints)')
    parser.add_argument('--num-joints', type=int, default=6,
                        help='Number of joints per configuration (default: 6)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-matrices', action='store_true',
                        help='Show full transformation matrices for each configuration')
    args = parser.parse_args()

    num_joints = args.num_joints
    joint_angles_flat = args.joint_angles
    if len(joint_angles_flat) % num_joints != 0:
        raise ValueError(
            f"Total number of joint angles ({len(joint_angles_flat)}) must be divisible by num-joints ({num_joints})"
        )

    n_batch = len(joint_angles_flat) // num_joints
    args.joint_angles = [
        joint_angles_flat[i * num_joints:(i + 1) * num_joints]
        for i in range(n_batch)
    ]

    main(args)
