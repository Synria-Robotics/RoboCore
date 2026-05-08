"""Mass Matrix Demo

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

    beauty_print("Mass Matrix M(q) and M^{-1}(q)", type="module")
    beauty_print("Joint configuration q (rad):")
    print(f"  {beauty_print_array(q)}")

    M = dynamics.mass_matrix(model, q)
    beauty_print("Mass matrix M (nq x nq):")
    print(beauty_print_array(M, precision=6))

    cond = np.linalg.cond(M)
    norm_M = np.linalg.norm(M, "fro")
    beauty_print(f"M norm (Frobenius): {norm_M:.6f}")
    beauty_print(f"M condition number: {cond:.6e}")

    Minv = dynamics.mass_matrix_inverse(model, q)
    beauty_print("Inverse mass matrix M^{-1} (first 3x3 block):")
    k = min(3, nq)
    print(beauty_print_array(Minv[:k, :k], precision=6))

    I_reconstructed = M @ Minv
    err = np.linalg.norm(I_reconstructed - np.eye(nq), "fro")
    beauty_print(f"||M @ M^{-1} - I||_F: {err:.6e}", type="success" if err < 1e-8 else "info")


if __name__ == "__main__":
    try:
        from synriard import get_model_path
        default_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    except Exception:
        default_path = "robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf"

    parser = argparse.ArgumentParser(description="Mass Matrix Demo")
    parser.add_argument("--model-path", type=str, default=default_path, help="Path to URDF")
    parser.add_argument("--base-link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--end-link", type=str, default="tool0", help="End-effector link name")
    parser.add_argument("--joint-angles", type=float, nargs="+", default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help="Joint positions (rad)")
    args = parser.parse_args()
    main(args)
