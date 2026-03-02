"""Mass matrix and gravity/nle comparison with Pinocchio (optional).

Same URDF and (base_link, end_link): RC and Pin use the same chain (same DOF).
Same q; compare mass_matrix, gravity, nonlinear_effects with pinocchio.crba,
pinocchio.computeGeneralizedGravity, pinocchio.nonLinearEffects.

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
    q = np.array(args.joint_angles, dtype=np.float64)
    if len(q) != nq:
        q = np.resize(q, nq) if len(q) < nq else q[:nq]
    v = np.zeros(nq) if args.velocity is None else np.array(args.velocity, dtype=np.float64)
    if len(v) != nq:
        v = np.resize(v, nq) if len(v) < nq else v[:nq]

    beauty_print("Mass matrix / gravity / nle: RoboCore vs Pinocchio", type="module")
    beauty_print("q (rad):")
    print(f"  {beauty_print_array(q)}")
    beauty_print("v (rad/s):")
    print(f"  {beauty_print_array(v)}")

    M_rc = dynamics.mass_matrix(rc_model, q)
    g_rc = dynamics.gravity(rc_model, q)
    nle_rc = dynamics.nonlinear_effects(rc_model, q, v)

    beauty_print("RoboCore M (diag):")
    print(f"  {beauty_print_array(np.diag(M_rc))}")
    beauty_print("RoboCore g:")
    print(f"  {beauty_print_array(g_rc)}")
    beauty_print("RoboCore nle:")
    print(f"  {beauty_print_array(nle_rc)}")

    if _HAS_PINOCCHIO and not args.no_pinocchio:
        pin_model, pin_data = build_pinocchio_reduced_to_chain(model_path, rc_model)
        if pin_model is None:
            beauty_print("Pinocchio not available or reduction failed; skipping.", type="warning")
        elif pin_model.nv != nq:
            beauty_print("Pinocchio reduced model nv != chain DOF; skipping.", type="warning")
        else:
            beauty_print("Pinocchio model reduced to same chain (base_link -> end_link), DOF = %d" % nq, type="info")
            pinocchio.crba(pin_model, pin_data, q)
            M_pin = np.array(pin_data.M)
            g_pin = pinocchio.computeGeneralizedGravity(pin_model, pin_data, q)
            nle_pin = pinocchio.nonLinearEffects(pin_model, pin_data, q, v)

            beauty_print("Pinocchio M (diag), g, nle:")
            print(f"  M diag = {beauty_print_array(np.diag(M_pin))}")
            print(f"  g     = {beauty_print_array(g_pin)}")
            print(f"  nle   = {beauty_print_array(nle_pin)}")

            err_M = np.linalg.norm(M_rc - M_pin, "fro")
            err_g = np.linalg.norm(g_rc - g_pin)
            err_nle = np.linalg.norm(nle_rc - nle_pin)
            beauty_print("RC vs Pin:")
            print(f"  ||M_rc - M_pin||_F:   {err_M:.6e}")
            print(f"  ||g_rc - g_pin||:     {err_g:.6e}")
            print(f"  ||nle_rc - nle_pin||: {err_nle:.6e}")
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
        default_path = "robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"

    parser = argparse.ArgumentParser(description="Mass Matrix / Gravity / NLE vs Pinocchio")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2], help="q (rad)")
    parser.add_argument("--velocity", type=float, nargs="+", default=None, help="v (rad/s)")
    parser.add_argument("--no-pinocchio", action="store_true", help="Disable Pinocchio comparison")
    args = parser.parse_args()
    main(args)
