"""Humanoid Jacobian validation and comparison with Pytorch Kinematics and Pinocchio

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
import pinocchio

import robocore as rc
from robocore.modeling import RobotModel
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def main(args):
    # Load models
    model_path = str(args.model_path)
    
    end_links = [args.left_thumb_end, args.right_thumb_end, args.left_toe_end, args.right_toe_end]
    end_names = ['left_thumb', 'right_thumb', 'left_toe', 'right_toe']
    display_names = ['Left Thumb', 'Right Thumb', 'Left Toe', 'Right Toe']
    
    # RoboCore - create first to get joint mapping
    robot_model = RobotModel(model_path, base_link=args.base_link)
    rc.set_backend('torch', device=args.device)
    
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
    
    beauty_print(f"Humanoid Jacobian Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    
    # Set up PyTorch Kinematics chains
    for name, chain in chains_pk.items():
        if chain is not None:
            chains_pk[name] = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Jacobian Computation", type="module", centered=False)
    
    # Build unified configuration vector
    q_full = np.zeros(robot_model.num_dof)
    if args.joint_angles and len(args.joint_angles) <= robot_model.num_dof:
        q_full[:len(args.joint_angles)] = args.joint_angles
    
    q_full_torch = torch.tensor(q_full, dtype=dtype, device=device)
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q_full[:min(10, len(q_full))])}...")
    
    # Compute Jacobians
    J_pk_list = []
    J_pin_list = []
    J_rc_list = []
    
    # Pinocchio forward kinematics and Jacobian computation
    pinocchio.forwardKinematics(pin_model, pin_data, q_full)
    pinocchio.computeJointJacobians(pin_model, pin_data, q_full)
    pinocchio.updateFramePlacements(pin_model, pin_data)
    
    for name, end_link, display_name in zip(end_names, end_links, display_names):
        # PyTorch Kinematics
        if chains_pk[name] is not None and len(chain_joint_indices[name]) > 0:
            try:
                # Extract joint values for this chain from q_full using correct indices
                indices = chain_joint_indices[name]
                q_chain = q_full_torch[indices]
                J_pk_chain = chains_pk[name].jacobian(q_chain)
                if J_pk_chain.ndim == 3:
                    J_pk_chain = J_pk_chain.squeeze(0)
                J_pk_chain_np = J_pk_chain.cpu().numpy()
                
                # Expand to full configuration space (same as RoboCore)
                # PyTorch Kinematics Jacobian is 6 x n_chain, need to expand to 6 x n_full
                J_pk_full = np.zeros((6, robot_model.num_dof))
                J_pk_full[:, indices] = J_pk_chain_np
                J_pk_list.append(J_pk_full)
            except Exception as e:
                beauty_print(f"Warning: PyTorch Kinematics Jacobian failed for {display_name}: {e}", type="warning")
                J_pk_list.append(None)
        else:
            J_pk_list.append(None)
        
        # Pinocchio
        try:
            frame_id = pin_model.getFrameId(end_link)
            J_pin = pinocchio.getFrameJacobian(pin_model, pin_data, frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
            J_pin_list.append(J_pin)
        except:
            try:
                joint_id = pin_model.getJointId(end_link)
                J_pin = pinocchio.getJointJacobian(pin_model, pin_data, joint_id, pinocchio.LOCAL_WORLD_ALIGNED)
                J_pin_list.append(J_pin)
            except:
                beauty_print(f"Warning: Pinocchio Jacobian failed for {display_name}", type="warning")
                J_pin_list.append(None)
        
        # RoboCore
        J_rc = robot_model.jacobian(q_full, base_link=args.base_link, end_link=end_link)
        J_rc_list.append(to_numpy(J_rc))
    
    # Stack Jacobians (all should have same number of columns now: num_dof)
    J_pk_combined = np.vstack([j for j in J_pk_list if j is not None]) if any(j is not None for j in J_pk_list) else None
    J_pin_combined = np.vstack([j for j in J_pin_list if j is not None]) if any(j is not None for j in J_pin_list) else None
    J_rc_combined = np.vstack(J_rc_list)
    
    beauty_print(f"Combined Jacobian shape (PyTorch Kinematics): {J_pk_combined.shape if J_pk_combined is not None else 'N/A'}")
    beauty_print(f"Combined Jacobian shape (Pinocchio): {J_pin_combined.shape if J_pin_combined is not None else 'N/A'}")
    beauty_print(f"Combined Jacobian shape (RoboCore): {J_rc_combined.shape}")
    
    # Display and compare
    for i, (name, display_name) in enumerate(zip(end_names, display_names)):
        beauty_print(f"{display_name} Jacobian:", type="module", centered=False)
        if J_pk_list[i] is not None:
            print(f"  PyTorch Kinematics (first 3 rows):")
            print(beauty_print_array(J_pk_list[i][:3, :], precision=6))
        if J_pin_list[i] is not None:
            print(f"  Pinocchio (first 3 rows):")
            print(beauty_print_array(J_pin_list[i][:3, :], precision=6))
        print(f"  RoboCore (first 3 rows):")
        print(beauty_print_array(J_rc_list[i][:3, :], precision=6))
        
        # Comparison (all Jacobians should now have same shape: 6 x num_dof)
        if J_pk_list[i] is not None and J_rc_list[i] is not None:
            if J_pk_list[i].shape == J_rc_list[i].shape:
                diff_pk_rc = J_pk_list[i] - J_rc_list[i]
                beauty_print(f"{display_name} PyTorch Kinematics vs RoboCore:")
                beauty_print(f"    Max difference:        {np.max(np.abs(diff_pk_rc)):.3e}")
                beauty_print(f"    Frobenius norm:        {np.linalg.norm(diff_pk_rc, 'fro'):.3e}")
            else:
                beauty_print(f"{display_name} PyTorch Kinematics vs RoboCore: Shape mismatch (PK: {J_pk_list[i].shape}, RC: {J_rc_list[i].shape})", type="warning")
        
        if J_pin_list[i] is not None and J_rc_list[i] is not None:
            if J_pin_list[i].shape == J_rc_list[i].shape:
                diff_pin_rc = J_pin_list[i] - J_rc_list[i]
                beauty_print(f"{display_name} Pinocchio vs RoboCore:")
                beauty_print(f"    Max difference:        {np.max(np.abs(diff_pin_rc)):.3e}")
                beauty_print(f"    Frobenius norm:        {np.linalg.norm(diff_pin_rc, 'fro'):.3e}")
            else:
                beauty_print(f"{display_name} Pinocchio vs RoboCore: Shape mismatch (Pin: {J_pin_list[i].shape}, RC: {J_rc_list[i].shape})", type="warning")
    
    beauty_print("✓ Humanoid Jacobian validation complete", type="success")


if __name__ == '__main__':
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="urdf")
    
    parser = argparse.ArgumentParser(description="Humanoid Jacobian validation")
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
    parser.add_argument('--joint-angles', type=float, nargs='+', default=None,
                        help='Joint angles in radians (optional)')
    parser.add_argument('--device', default='cpu', help='PyTorch device')
    args = parser.parse_args()
    
    main(args)
