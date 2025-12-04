"""Module

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

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import *

def random_q(model, rng, scale=0.5):
    """Generate random joint configuration within limits."""
    q = [0.0] * model.num_dof()
    for js in model._actuated:  # type: ignore[attr-defined]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        q[js.index] = float(rng.uniform(mid - span, mid + span))
    return q


def main(args):
    urdf_path = args.urdf
    end_link = args.end_link

    robot_model = RobotModel(str(urdf_path), end_link=end_link)
    T_fk = np.zeros((4, 4))
    
    T_fk[:3, 3] = args.end_pose[:3]
    T_fk[3, 3] = 1.0
    T_fk[:3, :3] = quaternion_to_matrix(args.end_pose[3:])
    
    # Use a random initial guess
    rng = np.random.default_rng(42)
    q_init = random_q(robot_model, rng)

    beauty_print(f"Initial Guess (radians):")
    print(f"  q_init = {beauty_print_array(q_init)}")

    # Solve IK using DLS method
    ik_result = inverse_kinematics(
        robot_model,
        T_fk,
        q_init,
        backend='numpy',
        method='dls',
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
        use_analytic_jacobian=True,
    )

    beauty_print(f"IK Solution:")
    print(f"  Success: {ik_result['success']}")
    print(f"  Iterations: {ik_result['iters']}")
    print(f"  Position Error: {ik_result['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
    beauty_print(f"Solved Joint Angles (degrees):")
    print(f"  q_ik = {beauty_print_array(np.rad2deg(ik_result['q']))}")


if __name__ == "__main__":
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm")
    
    parser = argparse.ArgumentParser(description="Forward Kinematics Demo")
    parser.add_argument('--urdf', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='tool0',
                        help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+', default=[0.29088, 0.02910, 0.15621, 0.041452, 0.828401, 0.083479, 0.552327], 
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    # Target joint angles: [0.1, 0.2, -0.3, 0.0, 0.5, -0.2]
    args = parser.parse_args()
    main(args)
