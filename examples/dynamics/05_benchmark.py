"""Dynamics Benchmark: ID, FD, mass_matrix timing.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import argparse
import time
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore import dynamics
from robocore.utils.beauty_logger import beauty_print


def main(args):
    model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    model.summary(show_chain=True)

    nq = model.num_chain_dof
    rng = np.random.default_rng(args.seed)
    q = rng.uniform(-0.5, 0.5, nq)
    v = rng.uniform(-0.1, 0.1, nq)
    a = rng.uniform(-0.1, 0.1, nq)
    tau = rng.uniform(-1.0, 1.0, nq)

    n_runs = args.runs
    beauty_print("Dynamics Benchmark", type="module", centered=True)
    beauty_print(f"DOF: {nq}, runs: {n_runs}")

    # Inverse dynamics
    dynamics.inverse_dynamics(model, q, v, a)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dynamics.inverse_dynamics(model, q, v, a)
    t_id = (time.perf_counter() - t0) / n_runs * 1000
    beauty_print(f"Inverse dynamics (RNEA): {t_id:.4f} ms")

    # Forward dynamics
    dynamics.forward_dynamics(model, q, v, tau)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dynamics.forward_dynamics(model, q, v, tau)
    t_fd = (time.perf_counter() - t0) / n_runs * 1000
    beauty_print(f"Forward dynamics (mass-matrix solve): {t_fd:.4f} ms")

    # Mass matrix
    dynamics.mass_matrix(model, q)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dynamics.mass_matrix(model, q)
    t_m = (time.perf_counter() - t0) / n_runs * 1000
    beauty_print(f"Mass matrix (CRBA): {t_m:.4f} ms")

    # Gravity
    dynamics.gravity(model, q)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dynamics.gravity(model, q)
    t_g = (time.perf_counter() - t0) / n_runs * 1000
    beauty_print(f"Gravity: {t_g:.4f} ms")

    # Nonlinear effects
    dynamics.nonlinear_effects(model, q, v)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dynamics.nonlinear_effects(model, q, v)
    t_nle = (time.perf_counter() - t0) / n_runs * 1000
    beauty_print(f"Nonlinear effects: {t_nle:.4f} ms")

    beauty_print("Benchmark complete", type="success")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        from robocore.configs import resolve_description_path
        default_path = resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")

    parser = argparse.ArgumentParser(description="Dynamics Benchmark")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--runs", type=int, default=1000, help="Number of runs per function")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
