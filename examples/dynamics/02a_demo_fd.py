"""Forward Dynamics Demo

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
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        q = rng.uniform(-0.5, 0.5, nq)
        v = rng.uniform(-0.1, 0.1, nq)
        tau = rng.uniform(-1.0, 1.0, nq)
    else:
        q = np.array(args.joint_angles, dtype=np.float64)
        if len(q) != nq:
            q = np.resize(q, nq) if len(q) < nq else q[:nq]
        v = np.array(args.velocity, dtype=np.float64) if args.velocity else np.zeros(nq)
        if len(v) != nq:
            v = np.resize(v, nq) if len(v) < nq else v[:nq]
        tau = np.array(args.torques, dtype=np.float64) if args.torques else np.zeros(nq)
        if len(tau) != nq:
            tau = np.resize(tau, nq) if len(tau) < nq else tau[:nq]

    beauty_print("Forward Dynamics: ddq = ABA(q, v, tau)", type="module")
    beauty_print("Joint configuration q (rad):")
    print(f"  {beauty_print_array(q)}")
    beauty_print("Joint velocity v (rad/s):")
    print(f"  {beauty_print_array(v)}")
    beauty_print("Joint torques tau:")
    print(f"  {beauty_print_array(tau)}")

    ddq = dynamics.forward_dynamics(model, q, v, tau)
    beauty_print("Joint accelerations ddq (rad/s^2):")
    print(f"  {beauty_print_array(ddq)}")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        from robocore.configs import resolve_description_path
        default_path = resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")

    parser = argparse.ArgumentParser(description="Forward Dynamics Demo")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for q,v,tau (overrides manual values)")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help="Joint positions (rad)")
    parser.add_argument("--velocity", type=float, nargs="+", default=None, help="Joint velocities (rad/s)")
    parser.add_argument("--torques", type=float, nargs="+", default=None, help="Joint torques")
    args = parser.parse_args()
    main(args)
