#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

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

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.path import get_robocore_path


def compare_fk(model_a: RobotModel, model_b: RobotModel, q):
	"""Compute FK for both models and report pose differences.

	:param model_a: First model (URDF)
	:param model_b: Second model (MJCF)
	:param q: Joint configuration (may be shorter than either model's chain DOF)
	"""
	beauty_print("Compute and compare FK", type="module", centered=True)
	# Prepare q for each model: pad with zeros if needed, truncate if too long
	q_a = np.zeros(model_a.num_chain_dof)
	q_a[:min(len(q), model_a.num_chain_dof)] = q[:min(len(q), model_a.num_chain_dof)]

	q_b = np.zeros(model_b.num_chain_dof)
	q_b[:min(len(q), model_b.num_chain_dof)] = q[:min(len(q), model_b.num_chain_dof)]

	T_a = forward_kinematics(model_a, q_a, return_end=True)
	T_b = forward_kinematics(model_b, q_b, return_end=True)
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


def validate_kinematics_methods(model: RobotModel, q: np.ndarray):
	"""Validate that RobotModel methods match standalone kinematics functions.

	:param model: RobotModel instance
	:param q: Joint configuration to test
	"""
	beauty_print("Kinematics Methods Validation", type="module", centered=True)
	
	# ===== FK Validation =====
	beauty_print("1️⃣  Forward Kinematics (FK) Validation", type="info")
	
	# Call via RobotModel.fk()
	fk_model_full = model.fk(q, return_end=False)
	fk_model_end = model.fk(q, return_end=True)
	
	# Call via standalone function
	fk_standalone_full = forward_kinematics(model, q, return_end=False)
	fk_standalone_end = forward_kinematics(model, q, return_end=True)
	
	# Compare results
	fk_full_match = np.allclose(fk_model_full['end'], fk_standalone_full['end'], atol=1e-10)
	fk_end_match = np.allclose(fk_model_end, fk_standalone_end, atol=1e-10)
	
	if fk_full_match and fk_end_match:
		beauty_print("   ✓ FK: model.fk() == forward_kinematics()", type="success")
		print(f"     Max diff (full): {np.max(np.abs(fk_model_full['end'] - fk_standalone_full['end'])):.2e}")
		print(f"     Max diff (end):  {np.max(np.abs(fk_model_end - fk_standalone_end)):.2e}")
	else:
		beauty_print("   ✗ FK: Mismatch detected!", type="error")
		print(f"     Full match: {fk_full_match}, End match: {fk_end_match}")
	
	# ===== Jacobian Validation =====
	beauty_print("2️⃣  Jacobian Validation", type="info")
	
	# Call via RobotModel.jacobian()
	J_model = model.jacobian(q, method='analytic')
	
	# Call via standalone function
	J_standalone = jacobian(model, q, method='analytic')
	
	# Compare results
	J_match = np.allclose(J_model, J_standalone, atol=1e-10)
	
	if J_match:
		beauty_print("   ✓ Jacobian: model.jacobian() == jacobian()", type="success")
		print(f"     Shape: {J_model.shape}, Max diff: {np.max(np.abs(J_model - J_standalone)):.2e}")
	else:
		beauty_print("   ✗ Jacobian: Mismatch detected!", type="error")
		print(f"     Model shape: {J_model.shape}, Standalone shape: {J_standalone.shape}")
		print(f"     Max diff: {np.max(np.abs(J_model - J_standalone)):.2e}")
	
	# ===== IK Validation =====
	beauty_print("3️⃣  Inverse Kinematics (IK) Validation", type="info")
	
	# Get target pose from FK
	target_pose = fk_model_end
	q_initial = np.zeros(model.num_dof)
	
	# Call via RobotModel.ik()
	ik_result_model = model.ik(
		target_pose.tolist(),
		q_initial=q_initial,
		method='pinv',
		max_iters=100,
		pos_tol=1e-4,
		ori_tol=1e-4
	)
	
	# Call via standalone function
	ik_result_standalone = inverse_kinematics(
		model,
		target_pose.tolist(),
		q_initial,  # q0 is a positional argument
		method='pinv',
		max_iters=100,
		pos_tol=1e-4,
		ori_tol=1e-4
	)
	
	# Compare results
	q_model = np.array(ik_result_model['q'])
	q_standalone = np.array(ik_result_standalone['q'])
	
	# IK may converge to slightly different solutions, so check if both are valid
	model_valid = ik_result_model['success']
	standalone_valid = ik_result_standalone['success']
	q_diff = np.max(np.abs(q_model - q_standalone))
	
	if model_valid and standalone_valid:
		beauty_print("   ✓ IK: Both methods converged successfully", type="success")
		print(f"     Model:      success={model_valid}, pos_err={ik_result_model['pos_err']:.2e}, ori_err={ik_result_model['ori_err']:.2e}")
		print(f"     Standalone: success={standalone_valid}, pos_err={ik_result_standalone['pos_err']:.2e}, ori_err={ik_result_standalone['ori_err']:.2e}")
		print(f"     Solution diff: {q_diff:.2e} rad")
		
		# Verify both solutions reach the target
		fk_check_model = model.fk(q_model, return_end=True)
		fk_check_standalone = forward_kinematics(model, q_standalone, return_end=True)
		pose_err_model = np.linalg.norm(fk_check_model[:3, 3] - target_pose[:3, 3])
		pose_err_standalone = np.linalg.norm(fk_check_standalone[:3, 3] - target_pose[:3, 3])
		
		if pose_err_model < 1e-3 and pose_err_standalone < 1e-3:
			beauty_print("   ✓ IK: Both solutions reach target pose", type="success")
			print(f"     Model pose error:      {pose_err_model:.2e} m")
			print(f"     Standalone pose error: {pose_err_standalone:.2e} m")
		else:
			beauty_print("   ⚠ IK: Solutions may not reach target accurately", type="warning")
	elif model_valid or standalone_valid:
		beauty_print("   ⚠ IK: One method converged, the other didn't (may be difficult target)", type="warning")
		print(f"     Model success: {model_valid}, Standalone success: {standalone_valid}")
		if model_valid:
			print(f"     Model: pos_err={ik_result_model['pos_err']:.2e}, ori_err={ik_result_model['ori_err']:.2e}")
		if standalone_valid:
			print(f"     Standalone: pos_err={ik_result_standalone['pos_err']:.2e}, ori_err={ik_result_standalone['ori_err']:.2e}")
	else:
		beauty_print("   ⚠ IK: Both methods failed to converge (challenging target from zero config)", type="warning")
		print(f"     Model: pos_err={ik_result_model['pos_err']:.2e}, ori_err={ik_result_model['ori_err']:.2e}")
		print(f"     Standalone: pos_err={ik_result_standalone['pos_err']:.2e}, ori_err={ik_result_standalone['ori_err']:.2e}")
		print(f"     This is expected when target is far from initial guess")


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

    if urdf_model.num_chain_dof != mjcf_model.num_chain_dof:
        beauty_print("Chain DOF mismatch: using max chain DOF, shorter model will pad with zeros", type="warning")

    if args.random:
        q = np.array(urdf_model.random_q(seed=args.seed, scale=args.scale))
        beauty_print("Using random joint angles (middle range)")
        # Use max chain DOF to accommodate both models
        max_chain_dof = max(urdf_model.num_chain_dof, mjcf_model.num_chain_dof)
        if len(q) < max_chain_dof:
            q_padded = np.zeros(max_chain_dof)
            q_padded[:len(q)] = q
            q = q_padded
        elif len(q) > max_chain_dof:
            q = q[:max_chain_dof]
    else:
        # Use max chain DOF to accommodate both models
        max_chain_dof = max(urdf_model.num_chain_dof, mjcf_model.num_chain_dof)
        q = np.zeros(max_chain_dof)
        beauty_print("Using zero joint configuration")

    beauty_print("Joint angles (rad):")
    print(f"  q = {beauty_print_array(q)}")

    # Validate kinematics methods if requested
    if args.validate:
        validate_kinematics_methods(urdf_model, q)

    compare_fk(urdf_model, mjcf_model, q)


if __name__ == "__main__":
    import synriard

    parser = argparse.ArgumentParser(description="RobotModel loading demo (URDF & MJCF)")
    parser.add_argument(
        '--urdf',
        type=str,
        default=synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf"),
        help='URDF file path'
    )
    parser.add_argument(
        '--mjcf',
        type=str,
        default=synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="mjcf"),
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
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate that RobotModel.fk/ik/jacobian match standalone functions'
    )
    args = parser.parse_args()
    main(args)


