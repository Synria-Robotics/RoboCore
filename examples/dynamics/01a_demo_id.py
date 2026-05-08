"""Inverse Dynamics Demo

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
    v = np.zeros(nq) if args.velocity is None else np.array(args.velocity, dtype=np.float64)
    if len(v) != nq:
        v = np.resize(v, nq) if len(v) < nq else v[:nq]
    a = np.zeros(nq) if args.acceleration is None else np.array(args.acceleration, dtype=np.float64)
    if len(a) != nq:
        a = np.resize(a, nq) if len(a) < nq else a[:nq]

    beauty_print("Inverse Dynamics: tau = RNEA(q, v, a)", type="module")
    beauty_print("Joint configuration q (rad):")
    print(f"  {beauty_print_array(q)}")
    beauty_print("Joint velocity v (rad/s):")
    print(f"  {beauty_print_array(v)}")
    beauty_print("Joint acceleration a (rad/s^2):")
    print(f"  {beauty_print_array(a)}")

    tau = dynamics.inverse_dynamics(model, q, v, a)
    beauty_print("Joint torques tau (N or N·m):")
    print(f"  {beauty_print_array(tau)}")

    g = dynamics.gravity(model, q)
    beauty_print("Gravity vector g(q):")
    print(f"  {beauty_print_array(g)}")

    nle = dynamics.nonlinear_effects(model, q, v)
    beauty_print("Nonlinear effects nle(q,v) = RNEA(q,v,0):")
    print(f"  {beauty_print_array(nle)}")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        default_path = "robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"

    parser = argparse.ArgumentParser(description="Inverse Dynamics Demo")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help="Joint positions (rad)")
    parser.add_argument("--velocity", type=float, nargs="+", default=None, help="Joint velocities (rad/s)")
    parser.add_argument("--acceleration", type=float, nargs="+", default=None, help="Joint accelerations (rad/s^2)")
    args = parser.parse_args()
    main(args)
