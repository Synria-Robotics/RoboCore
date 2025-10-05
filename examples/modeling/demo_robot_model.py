#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RobotModel loading demo (URDF & MJCF)
=======================================

Showcases:
1. Load the same robot in URDF and MJCF formats.
2. Print DOF, joint names, end link via model.summary().
3. Visualize kinematic tree structure with model.print_tree().
4. Generate a random joint configuration via unified API.
5. Compute and compare FK end-effector poses (expected to be close).

Notes:
- MJCF parser is a minimal subset (single longest serial chain, first hinge/slide joint per body).
- Branches or multiple stacked joints per body are not fully represented.

Example::

    # Basic usage
    python examples/modeling/demo_robot_model.py

    # Show detailed chain info
    python examples/modeling/demo_robot_model.py --show-chain

    # Show kinematic tree structure
    python examples/modeling/demo_robot_model.py --show-tree

    # Show tree with fixed joints
    python examples/modeling/demo_robot_model.py --show-tree --show-fixed

    # Random configuration comparison
    python examples/modeling/demo_robot_model.py --random --seed 123

    # Full example
    python examples/modeling/demo_robot_model.py \\
        --urdf robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf \\
        --mjcf robocore/assets/robot/mjcf/Alicia-D_v5_4/alicia_duo_with_gripper.xml \\
        --show-tree --show-fixed --random --seed 123
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.path import get_robocore_path


def compare_fk(model_a: RobotModel, model_b: RobotModel, q):
	"""Compute FK for both models and report pose differences.

	:param model_a: First model (URDF)
	:param model_b: Second model (MJCF)
	:param q: Joint configuration (min DOF length)
	"""
	beauty_print("Compute and compare FK", type="module", centered=True)
	T_a = forward_kinematics(model_a, q, backend='numpy', return_end=True)
	T_b = forward_kinematics(model_b, q, backend='numpy', return_end=True)
	pa, pb = T_a[:3, 3], T_b[:3, 3]
	Ra, Rb = T_a[:3, :3], T_b[:3, :3]
	pos_err = np.linalg.norm(pa - pb)
	# orientation difference angle
	dR = Ra @ Rb.T
	angle_err = np.arccos(np.clip((np.trace(dR) - 1) / 2.0, -1, 1))
	beauty_print("URDF end pose (T)")
	print(beauty_print_array(T_a, precision=6))
	beauty_print("MJCF end pose (T)")
	print(beauty_print_array(T_b, precision=6))
	beauty_print("Position difference |p_urdf - p_mjcf|:")
	print(f"  {pos_err:.6e} m")
	beauty_print("Orientation difference angle:")
	print(f"  {angle_err:.6e} rad  ({np.rad2deg(angle_err):.6e} deg)")


def main(args):
    """Entry point for CLI.

    :param args: Parsed CLI arguments
    :return: None
    """
    urdf_path = Path(args.urdf)
    mjcf_path = Path(args.mjcf)
    if not urdf_path.exists():
        beauty_print(f"URDF not found: {urdf_path}", type="error")
        return
    if not mjcf_path.exists():
        beauty_print(f"MJCF not found: {mjcf_path}", type="error")
        return

    # Load models - use auto-detect for end_link if not explicitly set to ensure compatibility
    end_link = args.end_link if args.end_link != 'tool0' else None
    if end_link is None:
        beauty_print("Auto-detecting end-effector link (longest chain)", type="info")
    
    urdf_model = RobotModel(str(urdf_path), end_link=end_link)
    mjcf_model = RobotModel(str(mjcf_path), end_link=end_link)

    # Display model summaries
    urdf_model.summary(show_chain=args.show_chain, title="[URDF] Model Summary")
    mjcf_model.summary(show_chain=args.show_chain, title="[MJCF] Model Summary")
    
    # Optionally show tree structure
    if args.show_tree:
        urdf_model.print_tree(show_fixed=args.show_fixed)
        mjcf_model.print_tree(show_fixed=args.show_fixed)

    if urdf_model.num_dof() != mjcf_model.num_dof():
        beauty_print("DOF mismatch: configuration will be truncated to the smaller DOF", type="warning")

    if args.random:
        rng = np.random.default_rng(args.seed)
        q = np.array(urdf_model.random_q(rng=rng, scale=args.scale))
        beauty_print("Using random joint angles (middle range)")
    else:
        q = np.zeros(urdf_model.num_dof())
        beauty_print("Using zero joint configuration")

    min_dof = min(urdf_model.num_dof(), mjcf_model.num_dof())
    if len(q) != min_dof:
        q = q[:min_dof]

    beauty_print("Joint angles (rad):")
    print(f"  q = {beauty_print_array(q)}")

    compare_fk(urdf_model, mjcf_model, q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RobotModel loading demo (URDF & MJCF)")
    parser.add_argument(
        '--urdf',
        type=str,
        default=get_robocore_path('assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf'),
        help='URDF file path'
    )
    parser.add_argument(
        '--mjcf',
        type=str,
        default=get_robocore_path('assets/robot/mjcf/Alicia-D_v5_4/alicia_duo_with_gripper.xml'),
        help='MJCF (MuJoCo XML) file path'
    )
    parser.add_argument(
        '--end-link',
        type=str,
        default='tool0',
        help='End-effector link name (default: tool0)'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random joint configuration'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Random sampling scale in joint range (0~1, default 0.5)'
    )
    parser.add_argument(
        '--show-chain',
        action='store_true',
        help='Show internal actuated chain details'
    )
    parser.add_argument(
        '--show-tree',
        action='store_true',
        help='Show kinematic tree structure'
    )
    parser.add_argument(
        '--show-fixed',
        action='store_true',
        help='Include fixed joints in tree (requires --show-tree)'
    )
    args = parser.parse_args()
    main(args)


