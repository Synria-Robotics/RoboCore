"""Gravity, Nonlinear Effects and Coriolis Matrix Demo

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import argparse
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore import dynamics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def main(args):
    model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    model.summary(show_chain=True)

    nq = model.num_chain_dof
    q = np.array(args.joint_angles, dtype=np.float64)
    if len(q) != nq:
        q = np.resize(q, nq) if len(q) < nq else q[:nq]
    v = np.array(args.velocity, dtype=np.float64) if args.velocity else np.zeros(nq)
    if len(v) != nq:
        v = np.resize(v, nq) if len(v) < nq else v[:nq]

    beauty_print("Gravity g(q), Nonlinear Effects nle(q,v), Coriolis C(q,v)", type="module")
    beauty_print("Joint configuration q (rad):")
    print(f"  {beauty_print_array(q)}")
    beauty_print("Joint velocity v (rad/s):")
    print(f"  {beauty_print_array(v)}")

    g = dynamics.gravity(model, q)
    beauty_print("Gravity g(q):")
    print(f"  {beauty_print_array(g)}")

    nle = dynamics.nonlinear_effects(model, q, v)
    beauty_print("Nonlinear effects nle(q,v):")
    print(f"  {beauty_print_array(nle)}")

    C = dynamics.coriolis_matrix(model, q, v, epsilon=args.epsilon)
    Cv = C @ v
    diff = nle - g - Cv
    beauty_print("Coriolis: C(q,v) such that nle = g + C@v")
    beauty_print("nle - g - C@v (should be near zero):")
    print(f"  {beauty_print_array(diff)}")
    beauty_print(f"||nle - g - C@v||: {np.linalg.norm(diff):.6e}", type="success" if np.linalg.norm(diff) < 1e-6 else "info")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        default_path = "robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"

    parser = argparse.ArgumentParser(description="Gravity and Nonlinear Effects Demo")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help="Joint positions (rad)")
    parser.add_argument("--velocity", type=float, nargs="+", default=None, help="Joint velocities (rad/s)")
    parser.add_argument("--epsilon", type=float, default=1e-7, help="Finite difference step for C(q,v)")
    args = parser.parse_args()
    main(args)
