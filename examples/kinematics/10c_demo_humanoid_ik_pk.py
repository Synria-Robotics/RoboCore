"""Humanoid Inverse Kinematics validation and comparison with Pytorch Kinematics and Pinocchio

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
import time
import numpy as np
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.ik import PseudoInverseIK
from pytorch_kinematics.transforms import Transform3d
import pinocchio

import robocore as rc
from robocore.modeling import RobotModel
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_quaternion


def main(args):
    # Load models
    model_path = str(args.model_path)
    
    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]
    end_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']
    
    # RoboCore - create first to get joint mapping
    robot_model = RobotModel(model_path, base_link=args.base_link)
    if args.rc_backend == 'torch':
        rc.set_backend('torch', device=args.device)
    else:
        rc.set_backend('cpp')
    
    # PyTorch Kinematics - build chains for all four end-effectors
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chains_pk = {}
    chain_joint_indices = {}  # Map chain joint names to indices in q_full
    for name, end_link in zip(end_names, end_links):
        try:
            chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name=args.base_link)
            chains_pk[name] = chain
            # Get joint indices for this chain in RobotModel's q_full
            indices = robot_model._get_joint_indices(args.base_link, end_link)
            chain_joint_indices[name] = indices
        except Exception as e:
            beauty_print(f"Warning: Could not build PyTorch Kinematics chain for {end_link}: {e}", type="warning")
            chains_pk[name] = None
            chain_joint_indices[name] = []
    
    # Pinocchio - build model
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()
    
    beauty_print(f"Humanoid Inverse Kinematics Comparison: PyTorch Kinematics vs RoboCore ({args.rc_backend})", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    beauty_print("Note: PyTorch Kinematics solves each end-effector independently. RoboCore solves all four simultaneously.", type="info")
    beauty_print("Note: Pinocchio IK for multi-chain is complex and skipped in this demo.", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    
    # Set up PyTorch Kinematics chains
    for name, chain in chains_pk.items():
        if chain is not None:
            chains_pk[name] = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Inverse Kinematics Computation", type="module", centered=False)
    
    # Use default target poses from 10a_demo_humanoid_ik.py (not generated from random_q)
    targets = {}
    target_list = [
        [+0.26426, +0.1655, +0.6523, 0.0, 0.0, 0.0, 1.0],  # left_thumb
        [+0.26426, -0.1654, +0.1523, 0.0, 0.0, 0.0, 1.0],  # right_thumb
        [-0.00000, +0.11851, -0.75686, 0.0, 0.0, 0.0, 1.0],  # left_toe
        [-0.00000, -0.11851, -0.75686, 0.0, 0.0, 0.0, 1.0],  # right_toe
    ]
    
    # Convert target poses to 4x4 matrices
    for end_link, target in zip(end_links, target_list):
        T = np.zeros((4, 4))
        T[:3, 3] = target[:3]
        T[3, 3] = 1.0
        T[:3, :3] = quaternion_to_matrix(target[3:])
        targets[end_link] = T
    
    beauty_print(f"Using default target poses from 10a_demo_humanoid_ik.py")
    
    # Solve IK with PyTorch Kinematics (independent for each end-effector)
    results_pk = {}
    for name, end_link, display_name in zip(end_names, end_links, display_names):
        if chains_pk[name] is not None and len(chain_joint_indices[name]) > 0:
            try:
                indices = chain_joint_indices[name]
                
                # Build joint limits array from dof_list (not joint_list)
                limits_list = []
                for idx in indices:
                    js = robot_model.dof_list[idx]  # Use dof_list for DOF joints
                    lo = js.limit_lower if js.limit_lower is not None else -np.pi
                    hi = js.limit_upper if js.limit_upper is not None else np.pi
                    limits_list.append([lo, hi])
                joint_limits = np.array(limits_list)
                joint_limits_torch = torch.tensor(joint_limits, dtype=dtype, device=device)
                
                ik_solver_pk = PseudoInverseIK(
                    chains_pk[name],
                    pos_tolerance=args.pos_tol,
                    rot_tolerance=args.ori_tol,
                    max_iterations=args.max_iters,
                    lr=args.step_size,
                    regularlization=args.damping,
                    num_retries=args.num_retries,
                    joint_limits=joint_limits_torch,
                )
                
                target_transform = Transform3d(matrix=torch.tensor(targets[end_link], dtype=dtype, device=device))
                # PseudoInverseIK.solve() doesn't support initial_guess parameter directly
                # It uses num_retries to try different random initial guesses
                sol_pk = ik_solver_pk.solve(target_transform)
                
                # Select best retry (prefer converged ones)
                converged_mask = sol_pk.converged[0, :].cpu().numpy()
                if np.any(converged_mask):
                    # Use the converged solution with smallest error
                    pos_errors = sol_pk.err_pos[0, :].cpu().numpy()
                    ori_errors = sol_pk.err_rot[0, :].cpu().numpy()
                    total_errors = pos_errors + ori_errors
                    converged_errors = np.where(converged_mask, total_errors, np.inf)
                    best_retry_idx = np.argmin(converged_errors)
                else:
                    # If none converged, use the one with smallest error
                    pos_errors = sol_pk.err_pos[0, :].cpu().numpy()
                    ori_errors = sol_pk.err_rot[0, :].cpu().numpy()
                    total_errors = pos_errors + ori_errors
                    best_retry_idx = np.argmin(total_errors)
                
                q_pk_chain = sol_pk.solutions[0, best_retry_idx, :].cpu().numpy()
                converged_pk = sol_pk.converged[0, best_retry_idx].item()
                
                results_pk[name] = {
                    'success': converged_pk,
                    'q_chain': q_pk_chain,  # Chain joint angles
                    'indices': indices,  # Store indices for reference
                    'iters': sol_pk.iterations,
                    'pos_err': sol_pk.err_pos[0, best_retry_idx].item(),
                    'ori_err': sol_pk.err_rot[0, best_retry_idx].item(),
                }
            except Exception as e:
                beauty_print(f"Warning: PyTorch Kinematics IK failed for {display_name}: {e}", type="warning")
                import traceback
                traceback.print_exc()
                results_pk[name] = None
        else:
            results_pk[name] = None
    
    # Solve IK with Pinocchio (independent for each end-effector)
    # Note: Pinocchio IK is complex for multi-chain, so we'll skip it for now
    # and just show PyTorch Kinematics vs RoboCore comparison
    results_pin = {}
    for name, end_link, display_name in zip(end_names, end_links, display_names):
        results_pin[name] = None  # Skip Pinocchio IK for multi-chain case
    
    # Solve IK with RoboCore (simultaneous for all four end-effectors)
    # Use random initial guesses (not the configuration that generated targets)
    # Use same parameters as 10a_demo_humanoid_ik.py
    ik_result_rc = robot_model.ik(
        targets=targets,
        end_links=end_links,
        q_initial=None,  # Use random initial guesses
        method='dls',
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
        base_link=args.base_link,
    )
    
    q_rc = to_numpy(ik_result_rc['q'])
    success_rc = ik_result_rc.get('success', False)
    pos_err_rc = ik_result_rc.get('pos_err', 0.0)
    ori_err_rc = ik_result_rc.get('ori_err', 0.0)
    iters_rc = ik_result_rc.get('iters', 0)
    
    # Display results
    beauty_print("[2] IK Solution Comparison", type="module", centered=False)
    beauty_print(f"RoboCore (simultaneous):")
    beauty_print(f"  Success: {success_rc}")
    beauty_print(f"  Iterations: {iters_rc}")
    beauty_print(f"  Position error: {pos_err_rc:.6e} m")
    beauty_print(f"  Orientation error: {ori_err_rc:.6e} rad")
    beauty_print(f"  Joint angles (first 10): {beauty_print_array(q_rc[:10])}...")
    
    for name, display_name in zip(end_names, display_names):
        beauty_print(f"{display_name}:")
        if results_pk[name] is not None:
            beauty_print(f"  PyTorch Kinematics:")
            beauty_print(f"    Success: {results_pk[name]['success']}")
            beauty_print(f"    Position error: {results_pk[name]['pos_err']:.6e} m")
            beauty_print(f"    Orientation error: {results_pk[name]['ori_err']:.6e} rad")
        if results_pin[name] is not None:
            beauty_print(f"  Pinocchio:")
            beauty_print(f"    Success: {results_pin[name]['success']}")
            beauty_print(f"    Position error: {results_pin[name]['pos_err']:.6e} m")
            beauty_print(f"    Orientation error: {results_pin[name]['ori_err']:.6e} rad")
    
    beauty_print("✓ Humanoid IK validation complete", type="success")


if __name__ == '__main__':
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")
    
    parser = argparse.ArgumentParser(description="Humanoid Inverse Kinematics validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Unitree G1)')
    parser.add_argument('--base-link', type=str, default='pelvis', help='Base link name')
    parser.add_argument('--left-thumb-end', type=str, default='left_hand_thumb_2_link', 
                        help='Left thumb end-effector link name')
    parser.add_argument('--right-thumb-end', type=str, default='right_hand_thumb_2_link', 
                        help='Right thumb end-effector link name')
    parser.add_argument('--left-toe-end', type=str, default='left_ankle_roll_link', 
                        help='Left toe end-effector link name')
    parser.add_argument('--right-toe-end', type=str, default='right_ankle_roll_link', 
                        help='Right toe end-effector link name')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    parser.add_argument('--scale', type=float, default=0.8, help='Scale factor for joint limits when generating targets')
    parser.add_argument('--pos-tol', type=float, default=1e-3, help='Position tolerance in meters (default: 1e-3)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance in radians (default: 1e-3)')
    parser.add_argument('--max-iters', type=int, default=200, help='Maximum IK iterations (default: 200)')
    parser.add_argument('--step-size', type=float, default=1.0, help='Step size for PyTorch Kinematics IK')
    parser.add_argument('--damping', type=float, default=1e-6, help='Damping factor for PyTorch Kinematics IK')
    parser.add_argument('--num-inits', type=int, default=1, help='Number of initial guesses to try per target (default: 1)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Strategy for generating initial guesses (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale factor for joint limits when generating guesses (0.0 to 1.0, default: 1.0)')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of retries for PyTorch Kinematics IK (default: 1)')
    parser.add_argument('--device', default='cpu', help='PyTorch device')
    parser.add_argument('--rc-backend', type=str, default='torch', choices=['torch', 'cpp'],
                        help='RoboCore IK backend for PK comparison')
    args = parser.parse_args()
    
    main(args)
