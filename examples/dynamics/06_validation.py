"""Dynamics self-consistency validation (no Pinocchio).

Checks: tau = M*ddq + nle; ID-FD roundtrip tau' = ID(q,v,FD(q,v,tau)) == tau.

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


def main(args):
    model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    model.summary(show_chain=True)

    nq = model.num_chain_dof
    rng = np.random.default_rng(args.seed)
    n_trials = args.trials

    beauty_print("Dynamics Self-Consistency Validation", type="module", centered=True)
    beauty_print(f"Trials: {n_trials}")

    err_tau_eq = []
    err_roundtrip = []
    for _ in range(n_trials):
        q = rng.uniform(-0.5, 0.5, nq)
        v = rng.uniform(-0.1, 0.1, nq)
        a = rng.uniform(-0.1, 0.1, nq)
        tau = dynamics.inverse_dynamics(model, q, v, a)
        M = dynamics.mass_matrix(model, q)
        nle = dynamics.nonlinear_effects(model, q, v)
        tau_reconstructed = M @ a + nle
        err_tau_eq.append(np.linalg.norm(tau - tau_reconstructed))

        tau_in = rng.uniform(-1.0, 1.0, nq)
        ddq = dynamics.forward_dynamics(model, q, v, tau_in)
        tau_back = dynamics.inverse_dynamics(model, q, v, ddq)
        err_roundtrip.append(np.linalg.norm(tau_in - tau_back))

    err_tau_eq = np.array(err_tau_eq)
    err_roundtrip = np.array(err_roundtrip)
    beauty_print("tau = M@ddq + nle:")
    print(f"  max error: {np.max(err_tau_eq):.6e}, mean: {np.mean(err_tau_eq):.6e}")
    beauty_print("ID-FD roundtrip (tau -> ddq -> tau'):")
    print(f"  max error: {np.max(err_roundtrip):.6e}, mean: {np.mean(err_roundtrip):.6e}")

    atol_tau = 1e-10
    atol_roundtrip = 1e-9
    ok1 = np.all(err_tau_eq < atol_tau)
    ok2 = np.all(err_roundtrip < atol_roundtrip)
    if ok1 and ok2:
        beauty_print("All checks passed.", type="success")
    else:
        beauty_print("Some checks failed (see errors above).", type="warning")
        beauty_print(f"Tolerances: tau=M@a+nle < {atol_tau}, roundtrip < {atol_roundtrip}", type="info")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        default_path = "robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"

    parser = argparse.ArgumentParser(description="Dynamics Validation")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
