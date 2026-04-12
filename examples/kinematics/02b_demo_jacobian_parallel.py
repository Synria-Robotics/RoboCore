"""Jacobian Parallel Demo

This demo demonstrates parallel/batch Jacobian computation and compares
NumPy, Torch, and C++ backend batch timing (mirrors 02a_demo_jacobian.py for single-J).

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
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy


def compute_jacobian_batch(robot_model, backend, joint_angles_batch, *, method='analytic'):
    """Compute batch Jacobian for given backend.

    Same layout as ``compute_fk_batch`` in ``01b_demo_fk_parallel.py``: normalize ``q`` to
    (B, n_dof), time one backend call, return array plus elapsed seconds.

    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param joint_angles_batch: Joint angles per row, shape (B, n_dof) or list of configs
    :param method: Jacobian method, 'analytic' or 'numeric'
    :return: Dictionary with batch Jacobian and wall-clock time for the Jacobian call
    """
    rc.set_backend(backend)
    q_batch = np.asarray(joint_angles_batch, dtype=np.float64)
    if q_batch.ndim == 1:
        q_batch = q_batch.reshape(1, -1)

    start_time = time.time()
    J = jacobian(robot_model, q_batch, method=method)
    elapsed_time = time.time() - start_time

    return {
        'jacobian': J,
        'time': elapsed_time,
    }


def _jacobian_agreement_frobenius(J_ref, J_other):
    """Max Frobenius norm of (J_ref - J_other) over batch, or single-matrix Frobenius norm."""
    A = to_numpy(J_ref)
    B = to_numpy(J_other)
    D = A - B
    if D.ndim == 2:
        return float(np.linalg.norm(D, 'fro'))
    return float(np.max(np.linalg.norm(D.reshape(D.shape[0], -1), axis=1)))


def main(args):
    joint_configs = args.joint_angles
    num_batch = len(joint_configs)
    q_batch = np.asarray(joint_configs, dtype=np.float64)

    beauty_print(f"Batch Jacobian: B={num_batch} configuration(s), method={args.method}")

    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    results_np = compute_jacobian_batch(robot_model, 'numpy', q_batch, method=args.method)
    results_torch = compute_jacobian_batch(robot_model, 'torch', q_batch, method=args.method)
    results_cpp = compute_jacobian_batch(robot_model, 'cpp', q_batch, method=args.method)

    J_np = to_numpy(results_np['jacobian'])
    fro_nt = _jacobian_agreement_frobenius(results_np['jacobian'], results_torch['jacobian'])
    fro_nc = _jacobian_agreement_frobenius(results_np['jacobian'], results_cpp['jacobian'])

    beauty_print("Jacobian agreement (max Frobenius over batch if B>1):")
    print(f"  np vs torch: {fro_nt:.6e}   np vs cpp: {fro_nc:.6e}")

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
    for i in range(num_batch):
        Ji = J_np[i] if J_np.ndim == 3 else J_np
        q_config = joint_configs[i]

        beauty_print(f"Configuration {i + 1}:", type="info")
        print(f"  Joint angles: {beauty_print_array(np.array(q_config))}")
        print(f"  Jacobian shape: {Ji.shape}")
        print(f"  Condition number: {np.linalg.cond(Ji):.2e}")

        if args.show_matrices:
            print(f"  Jacobian matrix:")
            print(beauty_print_array(Ji, precision=6))


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Jacobian batch demo — NumPy / Torch / C++ timing (see also 02a_demo_jacobian.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 02b_demo_jacobian_parallel.py

        python 02b_demo_jacobian_parallel.py --joint-angles \\
            0.1 0.2 -0.3 0.0 0.5 -0.2 \\
            0.2 0.3 -0.4 0.1 0.6 -0.3

        python 02b_demo_jacobian_parallel.py --method numeric --show-matrices
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
    parser.add_argument('--method', type=str, default='analytic',
                        choices=['analytic', 'numeric'],
                        help='Jacobian method (default: analytic; autograd not used in batch backend compare)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-matrices', action='store_true',
                        help='Show full Jacobian matrix for each configuration')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch', 'cpp'],
                        help='Legacy option (ignored — NumPy, Torch, and C++ are all tested)')
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
