"""Forward Kinematics validation and comparison with Pytorch Kinematics and Pinocchio

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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

from robocore.modeling import RobotModel
from robocore.kinematics.fk_utils.fk_solver_cpp import FKSolverCpp
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import matrix_to_euler, matrix_to_quaternion


def main(args):
    # Load models
    model_path = str(args.model_path)
    end_link = args.end_link
    
    # PyTorch Kinematics
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name=args.base_link)
    n_dof = len(chain.get_joint_parameter_names())
    
    # RoboCore C++ FK (single solver; same path as cpp backend without per-call construction)
    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=end_link)
    fk_cpp = FKSolverCpp(rc_model)
    
    # Pinocchio
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()

    # Map RoboCore actuated joints to Pinocchio joints
    # Get joint names from RoboCore
    chain_indices = rc_model._get_joint_indices(rc_model.base_link, rc_model.end_link)
    actuated_joint_names = [rc_model.joint_list[idx].name for idx in chain_indices]
    pin_q_indices = []
    for joint_name in actuated_joint_names:
        if pin_model.existJointName(joint_name):
            joint_id = pin_model.getJointId(joint_name)
            # Get the configuration index for this joint
            # For revolute/prismatic joints, it's typically model.idx_qs[joint_id]
            if joint_id < len(pin_model.idx_qs):
                pin_q_indices.append(pin_model.idx_qs[joint_id])
            else:
                # Fallback: assume sequential mapping
                pin_q_indices.append(joint_id)

    # Find end link frame ID
    end_frame_id = None
    end_joint_id = None
    if pin_model.existFrame(end_link):
        end_frame_id = pin_model.getFrameId(end_link)
    elif pin_model.existJointName(end_link):
        end_joint_id = pin_model.getJointId(end_link)
    else:
        beauty_print(f"Warning: Could not find {end_link} in pinocchio model. Using last joint.", type="warning")
        end_joint_id = len(pin_model.joints) - 1

    beauty_print(f"Forward Kinematics Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore C++ ({n_dof} DOF)", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Forward Kinematics Computation", type="module", centered=False)
    q = torch.tensor(args.joint_angles, dtype=dtype, device=device)
    q_np = q.cpu().numpy()
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q_np)}")
    
    # Compute FK with PyTorch Kinematics
    # Note: pytorch_kinematics always applies root link's origin transform (including world2base
    # rotation) in forward kinematics, regardless of base_link setting. This means when base_link
    # is set to a non-root link (e.g., base_link instead of world), the world2base transform is
    # still included in the result.
    q_tensor = q.unsqueeze(0)
    ret_pk = chain.forward_kinematics(q_tensor, end_only=False)
    tg_pk = ret_pk[end_link]
    m_pk = tg_pk.get_matrix()[0]  # (4, 4)
    T_pk = m_pk.cpu().numpy()
    pos_pk = T_pk[:3, 3]
    rot_pk = T_pk[:3, :3]
    
    # Compute FK with Pinocchio
    # Map joint angles to full pinocchio configuration
    q_pin_full = pinocchio.neutral(pin_model).copy()
    for i, pin_idx in enumerate(pin_q_indices):
        if pin_idx < len(q_pin_full):
            q_pin_full[pin_idx] = q_np[i]
    pinocchio.forwardKinematics(pin_model, pin_data, q_pin_full)
    if end_frame_id is not None:
        pinocchio.updateFramePlacements(pin_model, pin_data)
        T_pin = pin_data.oMf[end_frame_id].homogeneous
    elif end_joint_id is not None:
        # Use joint placement if frame not found
        T_pin = pin_data.oMi[end_joint_id].homogeneous
    else:
        raise ValueError(f"Could not find frame or joint for {end_link}")
    pos_pin = T_pin[:3, 3]
    rot_pin = T_pin[:3, :3]

    # Compute FK with RoboCore
    # Note: RoboCore does not apply root link's origin transform. When base_link == end_link,
    # it returns identity matrix. For other cases, it computes relative transform from base_link
    # to end_link without including world2base rotation.
    T_rc = fk_cpp.solve(q_np)
    T_rc_np = to_numpy(T_rc)
    pos_rc = T_rc_np[:3, 3]
    rot_rc = T_rc_np[:3, :3]
    
    beauty_print(f"End-Effector Position (PyTorch Kinematics):")
    print(f"  p = {beauty_print_array(pos_pk)}")
    beauty_print(f"End-Effector Position (Pinocchio):")
    print(f"  p = {beauty_print_array(pos_pin)}")
    beauty_print(f"End-Effector Position (RoboCore C++):")
    print(f"  p = {beauty_print_array(pos_rc)}")
    
    # Convert to quaternion for display
    quat_pk = matrix_to_quaternion(rot_pk)
    quat_pin = matrix_to_quaternion(rot_pin)
    quat_rc = matrix_to_quaternion(rot_rc)
    beauty_print(f"Quaternion xyzw (PyTorch Kinematics):")
    print(f"  quat = {beauty_print_array(quat_pk, precision=6)}")
    beauty_print(f"Quaternion xyzw (Pinocchio):")
    print(f"  quat = {beauty_print_array(quat_pin, precision=6)}")
    beauty_print(f"Quaternion xyzw (RoboCore C++):")
    print(f"  quat = {beauty_print_array(quat_rc, precision=6)}")
    
    # Position comparison
    pos_diff_pk_rc = pos_pk - pos_rc
    pos_diff_pin_rc = pos_pin - pos_rc
    pos_diff_pk_pin = pos_pk - pos_pin
    beauty_print("Position Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_rc)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_rc):.3e} m")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pin_rc)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pin_rc):.3e} m")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(pos_diff_pk_pin)):.3e} m")
    beauty_print(f"    Euclidean norm:        {np.linalg.norm(pos_diff_pk_pin):.3e} m")

    # Rotation comparison
    rot_diff_pk_rc = rot_pk - rot_rc
    rot_diff_pin_rc = rot_pin - rot_rc
    rot_diff_pk_pin = rot_pk - rot_pin
    beauty_print("Rotation Matrix Comparison:")
    beauty_print("  PyTorch Kinematics vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_rc)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_rc, 'fro'):.3e}")
    beauty_print("  Pinocchio vs RoboCore:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pin_rc)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pin_rc, 'fro'):.3e}")
    beauty_print("  PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"    Max difference:        {np.max(np.abs(rot_diff_pk_pin)):.3e}")
    beauty_print(f"    Frobenius norm:        {np.linalg.norm(rot_diff_pk_pin, 'fro'):.3e}")
    
    # Performance comparison
    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 100
    
    def benchmark(func):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000
    
    def benchmark_pin():
        q_pin_full = pinocchio.neutral(pin_model).copy()
        for i, pin_idx in enumerate(pin_q_indices):
            if pin_idx < len(q_pin_full):
                q_pin_full[pin_idx] = q_np[i]
        pinocchio.forwardKinematics(pin_model, pin_data, q_pin_full)
        if end_frame_id is not None:
            pinocchio.updateFramePlacements(pin_model, pin_data)
            return pin_data.oMf[end_frame_id]
        elif end_joint_id is not None:
            return pin_data.oMi[end_joint_id]
        else:
            raise ValueError(f"Could not find frame or joint for {end_link}")

    time_pk = benchmark(lambda: chain.forward_kinematics(q_tensor, end_only=False))
    time_pin = benchmark(benchmark_pin)
    time_rc = benchmark(lambda: fk_cpp.solve(q_np))
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"Pinocchio:           {time_pin:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup_pk_rc = time_pk / time_rc if time_rc > 0 else 0
    speedup_pin_rc = time_pin / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup (PK vs RC):  {speedup_pk_rc:.2f}x", type="success" if speedup_pk_rc > 1 else "info")
    beauty_print(f"Speedup (Pin vs RC): {speedup_pin_rc:.2f}x", type="success" if speedup_pin_rc > 1 else "info")
    
    # Value comparison across random configurations
    beauty_print(f"[3] Value comparison across {args.samples} random configurations", type="module", centered=False)
    pos_diffs_pk_rc = []
    pos_diffs_pin_rc = []
    pos_diffs_pk_pin = []
    rot_diffs_pk_rc = []
    rot_diffs_pin_rc = []
    rot_diffs_pk_pin = []
    for i in range(args.samples):
        q_rand = torch.tensor(rc_model.random_q(seed=args.seed), dtype=dtype, device=device)
        q_rand_np = q_rand.cpu().numpy()
        
        # PyTorch Kinematics
        q_rand_tensor = q_rand.unsqueeze(0)
        ret_pk_rand = chain.forward_kinematics(q_rand_tensor, end_only=False)
        tg_pk_rand = ret_pk_rand[end_link]
        m_pk_rand = tg_pk_rand.get_matrix()[0]
        T_pk_rand = m_pk_rand.cpu().numpy()
        pos_pk_rand = T_pk_rand[:3, 3]
        rot_pk_rand = T_pk_rand[:3, :3]
        
        # Pinocchio
        q_pin_full_rand = pinocchio.neutral(pin_model).copy()
        for i, pin_idx in enumerate(pin_q_indices):
            if pin_idx < len(q_pin_full_rand):
                q_pin_full_rand[pin_idx] = q_rand_np[i]
        pinocchio.forwardKinematics(pin_model, pin_data, q_pin_full_rand)
        if end_frame_id is not None:
            pinocchio.updateFramePlacements(pin_model, pin_data)
            T_pin_rand = pin_data.oMf[end_frame_id].homogeneous
        elif end_joint_id is not None:
            T_pin_rand = pin_data.oMi[end_joint_id].homogeneous
        else:
            raise ValueError(f"Could not find frame or joint for {end_link}")
        pos_pin_rand = T_pin_rand[:3, 3]
        rot_pin_rand = T_pin_rand[:3, :3]

        # RoboCore
        T_rc_rand = fk_cpp.solve(q_rand_np)
        T_rc_rand_np = to_numpy(T_rc_rand)
        pos_rc_rand = T_rc_rand_np[:3, 3]
        rot_rc_rand = T_rc_rand_np[:3, :3]
        
        # Differences
        pos_diff_pk_rc = pos_pk_rand - pos_rc_rand
        pos_diff_pin_rc = pos_pin_rand - pos_rc_rand
        pos_diff_pk_pin = pos_pk_rand - pos_pin_rand
        rot_diff_pk_rc = rot_pk_rand - rot_rc_rand
        rot_diff_pin_rc = rot_pin_rand - rot_rc_rand
        rot_diff_pk_pin = rot_pk_rand - rot_pin_rand
        
        pos_diffs_pk_rc.append(np.linalg.norm(pos_diff_pk_rc))
        pos_diffs_pin_rc.append(np.linalg.norm(pos_diff_pin_rc))
        pos_diffs_pk_pin.append(np.linalg.norm(pos_diff_pk_pin))
        rot_diffs_pk_rc.append(np.linalg.norm(rot_diff_pk_rc, 'fro'))
        rot_diffs_pin_rc.append(np.linalg.norm(rot_diff_pin_rc, 'fro'))
        rot_diffs_pk_pin.append(np.linalg.norm(rot_diff_pk_pin, 'fro'))
    
    beauty_print(f"Position difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs_pk_rc):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs_pk_rc):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs_pk_rc):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs_pk_rc):.6e} m")

    beauty_print(f"Position difference statistics (Pinocchio vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs_pin_rc):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs_pin_rc):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs_pin_rc):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs_pin_rc):.6e} m")

    beauty_print(f"Position difference statistics (PyTorch Kinematics vs Pinocchio):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs_pk_pin):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs_pk_pin):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs_pk_pin):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs_pk_pin):.6e} m")
    
    beauty_print(f"Rotation difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs_pk_rc):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs_pk_rc):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs_pk_rc):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs_pk_rc):.6e}")

    beauty_print(f"Rotation difference statistics (Pinocchio vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs_pin_rc):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs_pin_rc):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs_pin_rc):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs_pin_rc):.6e}")

    beauty_print(f"Rotation difference statistics (PyTorch Kinematics vs Pinocchio):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs_pk_pin):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs_pk_pin):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs_pk_pin):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs_pk_pin):.6e}")
    
    beauty_print("✓ Forward kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Forward Kinematics validation with Pytorch Kinematics and Pinocchio")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)