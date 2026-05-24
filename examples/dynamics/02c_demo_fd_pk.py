"""Forward dynamics comparison with Pinocchio (optional).

Same URDF and (base_link, end_link): RC and Pin use the same chain (same DOF).
Same (q,v,tau); compare robocore.dynamics.forward_dynamics with pinocchio.aba.

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

try:
    import pinocchio
    _HAS_PINOCCHIO = True
except ImportError:
    _HAS_PINOCCHIO = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pin_utils import build_pinocchio_reduced_to_chain


def main(args):
    model_path = str(args.model_path)
    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=args.end_link)
    rc_model.summary(show_chain=True)

    nq = rc_model.num_chain_dof
    rng = np.random.default_rng(args.seed)
    q = rng.uniform(-0.5, 0.5, nq) if args.seed is not None else np.array(args.joint_angles, dtype=np.float64)
    if len(q) != nq:
        q = np.resize(q, nq) if len(q) < nq else q[:nq]
    v = rng.uniform(-0.1, 0.1, nq) if args.seed is not None else (np.array(args.velocity, dtype=np.float64) if args.velocity else np.zeros(nq))
    if len(v) != nq:
        v = np.resize(v, nq) if len(v) < nq else v[:nq]
    tau = rng.uniform(-1.0, 1.0, nq) if args.seed is not None else (np.array(args.torques, dtype=np.float64) if args.torques else np.zeros(nq))
    if len(tau) != nq:
        tau = np.resize(tau, nq) if len(tau) < nq else tau[:nq]

    beauty_print("Forward Dynamics: RoboCore vs Pinocchio", type="module")
    beauty_print("q, v, tau:")
    print(f"  q   = {beauty_print_array(q)}")
    print(f"  v   = {beauty_print_array(v)}")
    print(f"  tau = {beauty_print_array(tau)}")

    ddq_rc = dynamics.forward_dynamics(rc_model, q, v, tau)
    beauty_print("RoboCore ddq:")
    print(f"  {beauty_print_array(ddq_rc)}")

    if _HAS_PINOCCHIO and not args.no_pinocchio:
        pin_model, pin_data = build_pinocchio_reduced_to_chain(model_path, rc_model)
        if pin_model is None:
            beauty_print("Pinocchio not available or reduction failed; skipping.", type="warning")
        elif pin_model.nv != nq:
            beauty_print("Pinocchio reduced model nv != chain DOF; skipping.", type="warning")
        else:
            beauty_print("Pinocchio model reduced to same chain (base_link -> end_link), DOF = %d" % nq, type="info")
            ddq_pin = pinocchio.aba(pin_model, pin_data, q, v, tau)
            beauty_print("Pinocchio ddq:")
            print(f"  {beauty_print_array(ddq_pin)}")
            diff = ddq_rc - ddq_pin
            beauty_print("RC vs Pin:")
            print(f"  max |ddq_rc - ddq_pin|: {np.max(np.abs(diff)):.6e}")
            print(f"  ||ddq_rc - ddq_pin||:   {np.linalg.norm(diff):.6e}")
    else:
        if not _HAS_PINOCCHIO:
            beauty_print("Pinocchio not installed; only RoboCore result shown.", type="info")
        else:
            beauty_print("Pinocchio comparison disabled (--no-pinocchio).", type="info")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        from robocore.configs import resolve_description_path
        default_path = resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")

    parser = argparse.ArgumentParser(description="Forward Dynamics vs Pinocchio")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for q,v,tau")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2], help="q (rad)")
    parser.add_argument("--velocity", type=float, nargs="+", default=None, help="v (rad/s)")
    parser.add_argument("--torques", type=float, nargs="+", default=None, help="tau")
    parser.add_argument("--no-pinocchio", action="store_true", help="Disable Pinocchio comparison")
    args = parser.parse_args()
    main(args)
