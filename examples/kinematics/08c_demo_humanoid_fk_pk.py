"""Humanoid Forward Kinematics validation and comparison with Pytorch Kinematics and Pinocchio

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
from robocore.transform.conversions import matrix_to_quaternion


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
            # Get joint names from PyTorch Kinematics chain
            pk_joint_names = chain.get_joint_parameter_names()
            # Map to indices in RobotModel's q_full using the chain's actual joint indices
            # Use RobotModel's method to get correct joint indices for this chain
            indices = robot_model._get_joint_indices(args.base_link, end_link)
            # Verify that the number of joints matches
            if len(indices) != len(pk_joint_names):
                beauty_print(f"Warning: Joint count mismatch for {end_link}: PK={len(pk_joint_names)}, RC={len(indices)}", type="warning")
                # Try to match by name as fallback
                indices = []
                for pk_joint_name in pk_joint_names:
                    if pk_joint_name in robot_model._dof_name_to_index:
                        indices.append(robot_model._dof_name_to_index[pk_joint_name])
                    else:
                        beauty_print(f"Warning: Could not map PK joint '{pk_joint_name}' to RobotModel", type="warning")
            chain_joint_indices[name] = indices
        except Exception as e:
            beauty_print(f"Warning: Could not build PyTorch Kinematics chain for {end_link}: {e}", type="warning")
            chains_pk[name] = None
            chain_joint_indices[name] = []
    
    # Pinocchio - build model
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()
    frame_ids = {}
    for name, end_link in zip(end_names, end_links):
        try:
            frame_id = pin_model.getFrameId(end_link)
            frame_ids[name] = frame_id
        except:
            try:
                joint_id = pin_model.getJointId(end_link)
                frame_ids[name] = None
            except:
                beauty_print(f"Warning: Could not find {end_link} in pinocchio model.", type="warning")
                frame_ids[name] = None
    
    beauty_print(f"Humanoid Forward Kinematics Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    
    # Set up PyTorch Kinematics chains
    for name, chain in chains_pk.items():
        if chain is not None:
            chains_pk[name] = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Forward Kinematics Computation", type="module", centered=False)
    
    # Build unified configuration vector
    # q_full = np.zeros(robot_model.num_dof)
    q_full = robot_model.random_q_full()
    if args.joint_angles and len(args.joint_angles) <= robot_model.num_dof:
        q_full[:len(args.joint_angles)] = args.joint_angles
    
    q_full_torch = torch.tensor(q_full, dtype=dtype, device=device)
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q_full)}")
    # Compute FK with PyTorch Kinematics, Pinocchio, and RoboCore
    results_pk = {}
    results_pin = {}
    results_rc = {}
    
    # Pinocchio forward kinematics (once for all frames)
    pinocchio.forwardKinematics(pin_model, pin_data, q_full)
    pinocchio.updateFramePlacements(pin_model, pin_data)
    
    for name, end_link, display_name in zip(end_names, end_links, display_names):
        # PyTorch Kinematics
        if chains_pk[name] is not None and len(chain_joint_indices[name]) > 0:
            try:
                # Extract joint values for this chain from q_full using correct indices
                indices = chain_joint_indices[name]
                q_chain = q_full_torch[indices]
                q_chain_tensor = q_chain.unsqueeze(0)
                ret_pk = chains_pk[name].forward_kinematics(q_chain_tensor, end_only=False)
                tg_pk = ret_pk[end_link]
                m_pk = tg_pk.get_matrix()[0]
                T_pk = m_pk.cpu().numpy()
                results_pk[name] = T_pk
            except Exception as e:
                beauty_print(f"Warning: PyTorch Kinematics failed for {display_name}: {e}", type="warning")
                results_pk[name] = None
        else:
            results_pk[name] = None
        
        # Pinocchio
        if frame_ids[name] is not None:
            T_pin = pin_data.oMf[frame_ids[name]].homogeneous
        else:
            T_pin = None
        results_pin[name] = T_pin
        
        # RoboCore
        T_rc = robot_model.fk(q_full, base_link=args.base_link, end_link=end_link, return_end=True)
        results_rc[name] = to_numpy(T_rc)
    
    # Display results
    for name, display_name in zip(end_names, display_names):
        beauty_print(f"{display_name} End-Effector Position:", type="module", centered=False)
        if results_pk[name] is not None:
            pos_pk = results_pk[name][:3, 3]
            print(f"  PyTorch Kinematics:  {beauty_print_array(pos_pk)}")
        if results_pin[name] is not None:
            pos_pin = results_pin[name][:3, 3]
            print(f"  Pinocchio:          {beauty_print_array(pos_pin)}")
        pos_rc = results_rc[name][:3, 3]
        print(f"  RoboCore:            {beauty_print_array(pos_rc)}")
        
        # Comparison
        if results_pk[name] is not None and results_pin[name] is not None:
            pos_diff_pk_rc = results_pk[name][:3, 3] - pos_rc
            pos_diff_pin_rc = results_pin[name][:3, 3] - pos_rc
            pos_diff_pk_pin = results_pk[name][:3, 3] - results_pin[name][:3, 3]
            
            beauty_print(f"{display_name} Position Comparison:")
            if results_pk[name] is not None:
                beauty_print("  PyTorch Kinematics vs RoboCore:")
                beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_rc)):.3e} m")
                beauty_print(f"    Mean difference:       {np.mean(np.abs(pos_diff_pk_rc)):.3e} m")
            if results_pin[name] is not None:
                beauty_print("  Pinocchio vs RoboCore:")
                beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pin_rc)):.3e} m")
                beauty_print(f"    Mean difference:       {np.mean(np.abs(pos_diff_pin_rc)):.3e} m")
            if results_pk[name] is not None and results_pin[name] is not None:
                beauty_print("  PyTorch Kinematics vs Pinocchio:")
                beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_pin)):.3e} m")
                beauty_print(f"    Mean difference:       {np.mean(np.abs(pos_diff_pk_pin)):.3e} m")
        
        # Quaternion comparison
        if results_pk[name] is not None:
            quat_pk = matrix_to_quaternion(results_pk[name][:3, :3])
            print(f"  PyTorch Kinematics quat:  {beauty_print_array(quat_pk, precision=6)}")
        if results_pin[name] is not None:
            quat_pin = matrix_to_quaternion(results_pin[name][:3, :3])
            print(f"  Pinocchio quat:          {beauty_print_array(quat_pin, precision=6)}")
        quat_rc = matrix_to_quaternion(results_rc[name][:3, :3])
        print(f"  RoboCore quat:            {beauty_print_array(quat_rc, precision=6)}")
    
    beauty_print("✓ Humanoid FK validation complete", type="success")


if __name__ == '__main__':
    from openrd import get_model_path
    
    model_path = get_model_path("unitree_g1", variant="g1_body29_hand14", model_format="mjcf")
    
    parser = argparse.ArgumentParser(description="Humanoid Forward Kinematics validation")
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
