"""Bimanual Forward Kinematics validation and comparison with Pytorch Kinematics and Pinocchio

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
from robocore.kinematics.bimanual import bimanual_forward_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import matrix_to_quaternion


def main(args):
    # Load models
    model_path = str(args.model_path)
    
    # PyTorch Kinematics - build chains for both arms
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain_left = pk.build_serial_chain_from_urdf(urdf_bytes, args.left_end_link, root_link_name=args.left_base_link)
    chain_right = pk.build_serial_chain_from_urdf(urdf_bytes, args.right_end_link, root_link_name=args.right_base_link)
    n_dof_left = len(chain_left.get_joint_parameter_names())
    n_dof_right = len(chain_right.get_joint_parameter_names())
    
    # Pinocchio - build models for both arms
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()
    try:
        left_frame_id = pin_model.getFrameId(args.left_end_link)
    except:
        try:
            left_joint_id = pin_model.getJointId(args.left_end_link)
            left_frame_id = None
        except:
            beauty_print(f"Warning: Could not find {args.left_end_link} in pinocchio model.", type="warning")
            left_joint_id = len(pin_model.joints) - 1
            left_frame_id = None
    try:
        right_frame_id = pin_model.getFrameId(args.right_end_link)
    except:
        try:
            right_joint_id = pin_model.getJointId(args.right_end_link)
            right_frame_id = None
        except:
            beauty_print(f"Warning: Could not find {args.right_end_link} in pinocchio model.", type="warning")
            right_joint_id = len(pin_model.joints) - 1
            right_frame_id = None
    
    # RoboCore
    left_model = RobotModel(model_path, base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(model_path, base_link=args.right_base_link, end_link=args.right_end_link)
    rc.set_backend('torch', device=args.device)
    
    beauty_print(f"Bimanual Forward Kinematics Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore", type="module")
    beauty_print(f"Left arm: {n_dof_left} DOF, Right arm: {n_dof_right} DOF", type="info")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    chain_left = chain_left.to(dtype=dtype, device=device)
    chain_right = chain_right.to(dtype=dtype, device=device)
    
    beauty_print("[1] Forward Kinematics Computation", type="module", centered=False)
    q_left = torch.tensor(args.q_left, dtype=dtype, device=device)
    q_right = torch.tensor(args.q_right, dtype=dtype, device=device)
    q_left_np = q_left.cpu().numpy()
    q_right_np = q_right.cpu().numpy()
    
    beauty_print(f"Left arm joint configuration (rad):")
    print(f"  q_left = {beauty_print_array(q_left_np)}")
    beauty_print(f"Right arm joint configuration (rad):")
    print(f"  q_right = {beauty_print_array(q_right_np)}")
    
    # Compute FK with PyTorch Kinematics
    q_left_tensor = q_left.unsqueeze(0)
    q_right_tensor = q_right.unsqueeze(0)
    ret_pk_left = chain_left.forward_kinematics(q_left_tensor, end_only=False)
    ret_pk_right = chain_right.forward_kinematics(q_right_tensor, end_only=False)
    tg_pk_left = ret_pk_left[args.left_end_link]
    tg_pk_right = ret_pk_right[args.right_end_link]
    m_pk_left = tg_pk_left.get_matrix()[0]  # (4, 4)
    m_pk_right = tg_pk_right.get_matrix()[0]  # (4, 4)
    T_pk_left = m_pk_left.cpu().numpy()
    T_pk_right = m_pk_right.cpu().numpy()
    pos_pk_left = T_pk_left[:3, 3]
    pos_pk_right = T_pk_right[:3, 3]
    rot_pk_left = T_pk_left[:3, :3]
    rot_pk_right = T_pk_right[:3, :3]
    
    # Compute FK with Pinocchio (Note: Pinocchio processes full model, so we need to handle joint mapping)
    # For bimanual, we assume the model has both arms and we need to set all joints
    # This is a simplified version - in practice, you'd need to map left/right joints correctly
    q_full = np.zeros(pin_model.nq)
    # Map left and right joint angles to full model (simplified - assumes sequential joints)
    if n_dof_left <= pin_model.nq:
        q_full[:n_dof_left] = q_left_np
    if n_dof_right <= pin_model.nq - n_dof_left:
        q_full[n_dof_left:n_dof_left+n_dof_right] = q_right_np
    
    pinocchio.forwardKinematics(pin_model, pin_data, q_full)
    pinocchio.updateFramePlacements(pin_model, pin_data)
    
    if left_frame_id is not None:
        T_pin_left = pin_data.oMf[left_frame_id].homogeneous
    else:
        T_pin_left = pin_data.oMi[left_joint_id].homogeneous
    if right_frame_id is not None:
        T_pin_right = pin_data.oMf[right_frame_id].homogeneous
    else:
        T_pin_right = pin_data.oMi[right_joint_id].homogeneous
    
    pos_pin_left = T_pin_left[:3, 3]
    pos_pin_right = T_pin_right[:3, 3]
    rot_pin_left = T_pin_left[:3, :3]
    rot_pin_right = T_pin_right[:3, :3]
    
    # Compute FK with RoboCore
    result_rc = bimanual_forward_kinematics(
        left_model, right_model, q_left, q_right,
        return_end=True, mode=args.mode, device=device, dtype=dtype
    )
    T_rc_left = to_numpy(result_rc['left'])
    T_rc_right = to_numpy(result_rc['right'])
    pos_rc_left = T_rc_left[:3, 3]
    pos_rc_right = T_rc_right[:3, 3]
    rot_rc_left = T_rc_left[:3, :3]
    rot_rc_right = T_rc_right[:3, :3]
    
    beauty_print(f"Left Arm End-Effector Position (PyTorch Kinematics):")
    print(f"  p = {beauty_print_array(pos_pk_left)}")
    beauty_print(f"Left Arm End-Effector Position (Pinocchio):")
    print(f"  p = {beauty_print_array(pos_pin_left)}")
    beauty_print(f"Left Arm End-Effector Position (RoboCore):")
    print(f"  p = {beauty_print_array(pos_rc_left)}")
    
    beauty_print(f"Right Arm End-Effector Position (PyTorch Kinematics):")
    print(f"  p = {beauty_print_array(pos_pk_right)}")
    beauty_print(f"Right Arm End-Effector Position (Pinocchio):")
    print(f"  p = {beauty_print_array(pos_pin_right)}")
    beauty_print(f"Right Arm End-Effector Position (RoboCore):")
    print(f"  p = {beauty_print_array(pos_rc_right)}")
    
    # Convert to quaternion for display
    quat_pk_left = matrix_to_quaternion(rot_pk_left)
    quat_pin_left = matrix_to_quaternion(rot_pin_left)
    quat_rc_left = matrix_to_quaternion(rot_rc_left)
    quat_pk_right = matrix_to_quaternion(rot_pk_right)
    quat_pin_right = matrix_to_quaternion(rot_pin_right)
    quat_rc_right = matrix_to_quaternion(rot_rc_right)
    
    beauty_print(f"Left Arm Quaternion xyzw (PyTorch Kinematics):")
    print(f"  quat = {beauty_print_array(quat_pk_left, precision=6)}")
    beauty_print(f"Left Arm Quaternion xyzw (Pinocchio):")
    print(f"  quat = {beauty_print_array(quat_pin_left, precision=6)}")
    beauty_print(f"Left Arm Quaternion xyzw (RoboCore):")
    print(f"  quat = {beauty_print_array(quat_rc_left, precision=6)}")
    
    beauty_print(f"Right Arm Quaternion xyzw (PyTorch Kinematics):")
    print(f"  quat = {beauty_print_array(quat_pk_right, precision=6)}")
    beauty_print(f"Right Arm Quaternion xyzw (Pinocchio):")
    print(f"  quat = {beauty_print_array(quat_pin_right, precision=6)}")
    beauty_print(f"Right Arm Quaternion xyzw (RoboCore):")
    print(f"  quat = {beauty_print_array(quat_rc_right, precision=6)}")
    
    # Position comparison
    pos_diff_pk_rc_left = pos_pk_left - pos_rc_left
    pos_diff_pin_rc_left = pos_pin_left - pos_rc_left
    pos_diff_pk_pin_left = pos_pk_left - pos_pin_left
    pos_diff_pk_rc_right = pos_pk_right - pos_rc_right
    pos_diff_pin_rc_right = pos_pin_right - pos_rc_right
    pos_diff_pk_pin_right = pos_pk_right - pos_pin_right
    
    beauty_print("Left Arm Position Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_rc_left)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_rc_left):.3e} m")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pin_rc_left)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pin_rc_left):.3e} m")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_pin_left)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_pin_left):.3e} m")
    
    beauty_print("Right Arm Position Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_rc_right)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_rc_right):.3e} m")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pin_rc_right)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pin_rc_right):.3e} m")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_pin_right)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_pin_right):.3e} m")
    
    # Rotation comparison
    rot_diff_pk_rc_left = rot_pk_left - rot_rc_left
    rot_diff_pin_rc_left = rot_pin_left - rot_rc_left
    rot_diff_pk_pin_left = rot_pk_left - rot_pin_left
    rot_diff_pk_rc_right = rot_pk_right - rot_rc_right
    rot_diff_pin_rc_right = rot_pin_right - rot_rc_right
    rot_diff_pk_pin_right = rot_pk_right - rot_pin_right
    
    beauty_print("Left Arm Rotation Matrix Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_rc_left)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_rc_left, 'fro'):.3e}")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pin_rc_left)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pin_rc_left, 'fro'):.3e}")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_pin_left)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_pin_left, 'fro'):.3e}")
    
    beauty_print("Right Arm Rotation Matrix Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_rc_right)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_rc_right, 'fro'):.3e}")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pin_rc_right)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pin_rc_right, 'fro'):.3e}")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_pin_right)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_pin_right, 'fro'):.3e}")
    
    # Performance comparison
    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 100
    
    def benchmark_pk():
        q_left_t = q_left.unsqueeze(0)
        q_right_t = q_right.unsqueeze(0)
        ret_l = chain_left.forward_kinematics(q_left_t, end_only=False)
        ret_r = chain_right.forward_kinematics(q_right_t, end_only=False)
        tg_l = ret_l[args.left_end_link]
        tg_r = ret_r[args.right_end_link]
        m_l = tg_l.get_matrix()[0]
        m_r = tg_r.get_matrix()[0]
        return m_l, m_r
    
    def benchmark_pin():
        q_full = np.zeros(pin_model.nq)
        if n_dof_left <= pin_model.nq:
            q_full[:n_dof_left] = q_left_np
        if n_dof_right <= pin_model.nq - n_dof_left:
            q_full[n_dof_left:n_dof_left+n_dof_right] = q_right_np
        pinocchio.forwardKinematics(pin_model, pin_data, q_full)
        pinocchio.updateFramePlacements(pin_model, pin_data)
        if left_frame_id is not None:
            T_l = pin_data.oMf[left_frame_id]
        else:
            T_l = pin_data.oMi[left_joint_id]
        if right_frame_id is not None:
            T_r = pin_data.oMf[right_frame_id]
        else:
            T_r = pin_data.oMi[right_joint_id]
        return T_l, T_r
    
    def benchmark_rc():
        result = bimanual_forward_kinematics(
            left_model, right_model, q_left, q_right,
            return_end=True, mode=args.mode, device=device, dtype=dtype
        )
        return result['left'], result['right']
    
    def benchmark(func):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000
    
    time_pk = benchmark(benchmark_pk)
    time_pin = benchmark(benchmark_pin)
    time_rc = benchmark(benchmark_rc)
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"Pinocchio:           {time_pin:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup_pk_rc = time_pk / time_rc if time_rc > 0 else 0
    speedup_pin_rc = time_pin / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup (PK vs RC):  {speedup_pk_rc:.2f}x", type="success" if speedup_pk_rc > 1 else "info")
    beauty_print(f"Speedup (Pin vs RC): {speedup_pin_rc:.2f}x", type="success" if speedup_pin_rc > 1 else "info")
    
    # Value comparison across random configurations
    beauty_print(f"[3] Value comparison across {args.samples} random configurations", type="module", centered=False)
    pos_diffs_left = []
    pos_diffs_right = []
    rot_diffs_left = []
    rot_diffs_right = []
    
    for i in range(args.samples):
        q_left_rand = torch.tensor(left_model.random_q(seed=args.seed), dtype=dtype, device=device)
        q_right_rand = torch.tensor(right_model.random_q(seed=args.seed), dtype=dtype, device=device)
        
        # PyTorch Kinematics
        q_left_rand_tensor = q_left_rand.unsqueeze(0)
        q_right_rand_tensor = q_right_rand.unsqueeze(0)
        ret_pk_left_rand = chain_left.forward_kinematics(q_left_rand_tensor, end_only=False)
        ret_pk_right_rand = chain_right.forward_kinematics(q_right_rand_tensor, end_only=False)
        tg_pk_left_rand = ret_pk_left_rand[args.left_end_link]
        tg_pk_right_rand = ret_pk_right_rand[args.right_end_link]
        m_pk_left_rand = tg_pk_left_rand.get_matrix()[0]
        m_pk_right_rand = tg_pk_right_rand.get_matrix()[0]
        T_pk_left_rand = m_pk_left_rand.cpu().numpy()
        T_pk_right_rand = m_pk_right_rand.cpu().numpy()
        pos_pk_left_rand = T_pk_left_rand[:3, 3]
        pos_pk_right_rand = T_pk_right_rand[:3, 3]
        rot_pk_left_rand = T_pk_left_rand[:3, :3]
        rot_pk_right_rand = T_pk_right_rand[:3, :3]
        
        # RoboCore
        result_rc_rand = bimanual_forward_kinematics(
            left_model, right_model, q_left_rand, q_right_rand,
            return_end=True, mode=args.mode, device=device, dtype=dtype
        )
        T_rc_left_rand = to_numpy(result_rc_rand['left'])
        T_rc_right_rand = to_numpy(result_rc_rand['right'])
        pos_rc_left_rand = T_rc_left_rand[:3, 3]
        pos_rc_right_rand = T_rc_right_rand[:3, 3]
        rot_rc_left_rand = T_rc_left_rand[:3, :3]
        rot_rc_right_rand = T_rc_right_rand[:3, :3]
        
        # Differences
        pos_diff_left_rand = pos_pk_left_rand - pos_rc_left_rand
        pos_diff_right_rand = pos_pk_right_rand - pos_rc_right_rand
        rot_diff_left_rand = rot_pk_left_rand - rot_rc_left_rand
        rot_diff_right_rand = rot_pk_right_rand - rot_rc_right_rand
        
        pos_diffs_left.append(np.linalg.norm(pos_diff_left_rand))
        pos_diffs_right.append(np.linalg.norm(pos_diff_right_rand))
        rot_diffs_left.append(np.linalg.norm(rot_diff_left_rand, 'fro'))
        rot_diffs_right.append(np.linalg.norm(rot_diff_right_rand, 'fro'))
    
    beauty_print(f"Left Arm Position difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs_left):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs_left):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs_left):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs_left):.6e} m")
    
    beauty_print(f"Right Arm Position difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs_right):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs_right):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs_right):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs_right):.6e} m")
    
    beauty_print(f"Left Arm Rotation difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs_left):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs_left):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs_left):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs_left):.6e}")
    
    beauty_print(f"Right Arm Rotation difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs_right):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs_right):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs_right):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs_right):.6e}")
    
    beauty_print("✓ Bimanual forward kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    # Bessica is a dual-arm robot
    # Note: PyTorch Kinematics requires URDF format, not MJCF
    model_path = synriard.get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Forward Kinematics validation with Pytorch Kinematics and Pinocchio")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--q-left', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.1],
                        help='Left joint angles in radians')
    parser.add_argument('--q-right', type=float, nargs='+', default=[-0.1, -0.2, 0.3, 0.0, -0.5, 0.2, 0.1],
                        help='Right joint angles in radians')
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative', 'mirror'],
                        help='FK mode: indep (independent), relative (relative transform), mirror (mirror mode)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)





