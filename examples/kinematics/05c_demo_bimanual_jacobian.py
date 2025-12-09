"""Bimanual Jacobian Demo

This demo demonstrates bimanual Jacobian computation for dual-arm systems.
It shows how to compute Jacobian matrices for both arms independently and with relative constraints.

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
from robocore.kinematics.bimanual import bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy


def compute_bimanual_jacobian(left_model, right_model, backend, q_left, q_right, mode='indep'):
    """Compute bimanual Jacobian for given backend.
    
    :param left_model: Left arm RobotModel
    :param right_model: Right arm RobotModel
    :param backend: Backend name ('numpy' or 'torch')
    :param q_left: Left joint angles in radians
    :param q_right: Right joint angles in radians
    :param mode: Jacobian mode ('indep', 'relative')
    :return: Dictionary with results and computation time
    """
    rc.set_backend(backend)
    start_time = time.time()

    J = bimanual_jacobian(left_model, right_model, q_left, q_right, mode=mode)

    elapsed_time = time.time() - start_time

    J_np = to_numpy(J)

    return {
        'jacobian': J_np,
        'shape': J_np.shape,
        'time': elapsed_time
    }


def main(args):
    # Load robot model (Bessica is a dual-arm robot, use same model with different base/end links)
    left_model = RobotModel(str(args.model_path), base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(str(args.model_path), base_link=args.right_base_link, end_link=args.right_end_link)

    if args.verbose:
        beauty_print("Left Arm Model:", type="module")
        left_model.summary(show_chain=True)
        beauty_print("Right Arm Model:", type="module")
        right_model.summary(show_chain=True)

    # Compute with both backends
    results_np = compute_bimanual_jacobian(left_model, right_model, 'numpy', 
                                          args.q_left, args.q_right, mode=args.mode)
    results_torch = compute_bimanual_jacobian(left_model, right_model, 'torch', 
                                             args.q_left, args.q_right, mode=args.mode)

    # Display results
    beauty_print(f"Jacobian Matrix Shape:")
    print(f"  NumPy:  {results_np['shape']}")
    print(f"  Torch:  {results_torch['shape']}")

    J_np = results_np['jacobian']
    J_torch = results_torch['jacobian']

    # Compare matrices
    J_diff = np.linalg.norm(J_np - J_torch)
    beauty_print(f"Matrix Difference (Frobenius norm):")
    print(f"  Diff:   {J_diff:.6e}")

    # Show sub-matrices for independent mode
    if args.mode == 'indep':
        nL = left_model.num_chain_dof
        nR = right_model.num_chain_dof
        J_L = J_np[:6, :nL]
        J_R = J_np[6:, nL:]

        beauty_print(f"Left Arm Jacobian (6 x {nL}):")
        print(beauty_print_array(J_L, precision=4))

        beauty_print(f"Right Arm Jacobian (6 x {nR}):")
        print(beauty_print_array(J_R, precision=4))

    # Show condition number
    cond_np = np.linalg.cond(J_np)
    cond_torch = np.linalg.cond(J_torch)
    beauty_print(f"Condition Number:")
    print(f"  NumPy:  {cond_np:.6e}")
    print(f"  Torch:  {cond_torch:.6e}")

    beauty_print(f"Computation Time:")
    print(f"  NumPy:  {results_np['time']*1000:.4f} ms")
    print(f"  Torch:  {results_torch['time']*1000:.4f} ms")
    print(f"  Ratio:  {results_torch['time'] / results_np['time']:.2f}x")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")

    parser = argparse.ArgumentParser(description="Bimanual Jacobian Demo")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--q-left', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.1],
                        help='Left joint angles in radians')
    parser.add_argument('--q-right', type=float, nargs='+', default=[-0.1, -0.2, 0.3, 0.0, -0.5, 0.2, -0.1],
                        help='Right joint angles in radians')
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative'],
                        help='Jacobian mode: indep (independent), relative (relative constraint)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed model information')
    args = parser.parse_args()
    main(args)

