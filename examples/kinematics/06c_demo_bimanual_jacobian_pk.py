"""Bimanual Jacobian validation and comparison with Pytorch Kinematics and Pinocchio

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

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.bimanual import bimanual_jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


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
    
    # RoboCore
    left_model = RobotModel(model_path, base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(model_path, base_link=args.right_base_link, end_link=args.right_end_link)
    if args.rc_backend == 'torch':
        rc.set_backend('torch', device=args.device)
    else:
        rc.set_backend('cpp')
    
    beauty_print(f"Bimanual Jacobian Comparison: PyTorch Kinematics vs RoboCore ({args.rc_backend})", type="module")
    beauty_print(f"Left arm: {n_dof_left} DOF, Right arm: {n_dof_right} DOF", type="info")
    beauty_print(f"Mode: {args.mode}", type="info")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    chain_left = chain_left.to(dtype=dtype, device=device)
    chain_right = chain_right.to(dtype=dtype, device=device)
    
    beauty_print("[1] Jacobian Computation", type="module", centered=False)
    q_left = torch.tensor(args.q_left, dtype=dtype, device=device)
    q_right = torch.tensor(args.q_right, dtype=dtype, device=device)
    
    beauty_print(f"Left arm joint configuration (rad):")
    print(f"  q_left = {beauty_print_array(q_left.cpu().numpy())}")
    beauty_print(f"Right arm joint configuration (rad):")
    print(f"  q_right = {beauty_print_array(q_right.cpu().numpy())}")
    
    # Compute Jacobians with PyTorch Kinematics
    J_pk_left = chain_left.jacobian(q_left)
    J_pk_right = chain_right.jacobian(q_right)
    if J_pk_left.ndim == 3:
        J_pk_left = J_pk_left.squeeze(0)
    if J_pk_right.ndim == 3:
        J_pk_right = J_pk_right.squeeze(0)
    
    # Assemble block-diagonal Jacobian for independent mode
    if args.mode == 'indep':
        J_pk = torch.zeros((12, n_dof_left + n_dof_right), dtype=dtype, device=device)
        J_pk[0:6, 0:n_dof_left] = J_pk_left
        J_pk[6:12, n_dof_left:] = J_pk_right
    else:
        # For relative/mirror modes, PyTorch Kinematics doesn't have direct support
        # We'll compute individual Jacobians and note the limitation
        J_pk = torch.cat([J_pk_left, J_pk_right], dim=0)  # Stack vertically as approximation
        beauty_print("Note: PyTorch Kinematics doesn't support relative/mirror modes directly. Using stacked Jacobians for comparison.", type="info")
    
    J_pk_np = J_pk.cpu().numpy()
    
    # Compute Jacobian with RoboCore
    if args.rc_backend == 'cpp':
        q_l_rc = q_left.detach().cpu().numpy()
        q_r_rc = q_right.detach().cpu().numpy()
        J_rc = bimanual_jacobian(left_model, right_model, q_l_rc, q_r_rc, mode=args.mode)
    else:
        J_rc = bimanual_jacobian(left_model, right_model, q_left, q_right, mode=args.mode, device=device)
    J_rc_np = to_numpy(J_rc)
    
    beauty_print(f"Jacobian shape (PyTorch Kinematics): {J_pk_np.shape}")
    beauty_print(f"Jacobian shape (RoboCore): {J_rc_np.shape}")
    cond_num_pk = float(np.linalg.cond(J_pk_np))
    cond_num_rc = float(np.linalg.cond(J_rc_np))
    beauty_print(f"Condition number (PyTorch Kinematics): {cond_num_pk:.2e}")
    beauty_print(f"Condition number (RoboCore): {cond_num_rc:.2e}")
    
    beauty_print(f"Jacobian Matrix (PyTorch Kinematics, first 6 rows):")
    print(beauty_print_array(J_pk_np[:6, :], precision=6))
    if J_pk_np.shape[0] > 6:
        beauty_print(f"Jacobian Matrix (PyTorch Kinematics, last 6 rows):")
        print(beauty_print_array(J_pk_np[6:, :], precision=6))
    
    beauty_print(f"Jacobian Matrix (RoboCore, first 6 rows):")
    print(beauty_print_array(J_rc_np[:6, :], precision=6))
    if J_rc_np.shape[0] > 6:
        beauty_print(f"Jacobian Matrix (RoboCore, last 6 rows):")
        print(beauty_print_array(J_rc_np[6:, :], precision=6))
    
    # Value comparison (only for independent mode, where shapes match)
    if args.mode == 'indep' and J_pk_np.shape == J_rc_np.shape:
        diff = J_pk_np - J_rc_np
        beauty_print("PyTorch Kinematics vs RoboCore:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff, 'fro'):.3e}")
    else:
        beauty_print("Note: Shapes differ or mode not directly comparable. Skipping direct comparison.", type="info")
    
    # Performance comparison
    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 100
    
    def benchmark_pk():
        J_l = chain_left.jacobian(q_left)
        J_r = chain_right.jacobian(q_right)
        if J_l.ndim == 3:
            J_l = J_l.squeeze(0)
        if J_r.ndim == 3:
            J_r = J_r.squeeze(0)
        if args.mode == 'indep':
            J = torch.zeros((12, n_dof_left + n_dof_right), dtype=dtype, device=device)
            J[0:6, 0:n_dof_left] = J_l
            J[6:12, n_dof_left:] = J_r
        return J
    
    def benchmark_rc():
        if args.rc_backend == 'cpp':
            ql = q_left.detach().cpu().numpy()
            qr = q_right.detach().cpu().numpy()
            return bimanual_jacobian(left_model, right_model, ql, qr, mode=args.mode)
        return bimanual_jacobian(left_model, right_model, q_left, q_right, mode=args.mode, device=device)
    
    def benchmark(func):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000
    
    time_pk = benchmark(benchmark_pk)
    time_rc = benchmark(benchmark_rc)
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup = time_pk / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup:             {speedup:.2f}x", type="success" if speedup > 1 else "info")
    
    # Condition number statistics and value comparison
    beauty_print(f"[3] Condition number and value comparison across {args.samples} random configurations", type="module", centered=False)
    condition_numbers_pk = []
    condition_numbers_rc = []
    max_diffs = []
    
    for i in range(args.samples):
        q_left_rand = torch.tensor(left_model.random_q(seed=args.seed), dtype=dtype, device=device)
        q_right_rand = torch.tensor(right_model.random_q(seed=args.seed), dtype=dtype, device=device)
        
        # PyTorch Kinematics
        J_pk_left_rand = chain_left.jacobian(q_left_rand)
        J_pk_right_rand = chain_right.jacobian(q_right_rand)
        if J_pk_left_rand.ndim == 3:
            J_pk_left_rand = J_pk_left_rand.squeeze(0)
        if J_pk_right_rand.ndim == 3:
            J_pk_right_rand = J_pk_right_rand.squeeze(0)
        
        if args.mode == 'indep':
            J_pk_rand = torch.zeros((12, n_dof_left + n_dof_right), dtype=dtype, device=device)
            J_pk_rand[0:6, 0:n_dof_left] = J_pk_left_rand
            J_pk_rand[6:12, n_dof_left:] = J_pk_right_rand
        else:
            J_pk_rand = torch.cat([J_pk_left_rand, J_pk_right_rand], dim=0)
        J_pk_rand_np = J_pk_rand.cpu().numpy()
        
        # RoboCore
        if args.rc_backend == 'cpp':
            qlr = q_left_rand.detach().cpu().numpy()
            qrr = q_right_rand.detach().cpu().numpy()
            J_rc_rand = bimanual_jacobian(left_model, right_model, qlr, qrr, mode=args.mode)
        else:
            J_rc_rand = bimanual_jacobian(left_model, right_model, q_left_rand, q_right_rand, mode=args.mode, device=device)
        J_rc_rand_np = to_numpy(J_rc_rand)
        
        # Condition number
        cond_num_pk = float(np.linalg.cond(J_pk_rand_np))
        cond_num_rc = float(np.linalg.cond(J_rc_rand_np))
        condition_numbers_pk.append(cond_num_pk)
        condition_numbers_rc.append(cond_num_rc)
        
        # Value difference (only for independent mode)
        if args.mode == 'indep' and J_pk_rand_np.shape == J_rc_rand_np.shape:
            diff = J_pk_rand_np - J_rc_rand_np
            max_diffs.append(np.max(np.abs(diff)))
    
    beauty_print(f"Condition number statistics (PyTorch Kinematics):")
    beauty_print(f"  Mean:   {np.mean(condition_numbers_pk):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers_pk):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers_pk):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers_pk):.2e}")
    
    beauty_print(f"Condition number statistics (RoboCore):")
    beauty_print(f"  Mean:   {np.mean(condition_numbers_rc):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers_rc):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers_rc):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers_rc):.2e}")
    
    if max_diffs:
        beauty_print(f"Value difference statistics (PyTorch Kinematics vs RoboCore):")
        beauty_print(f"  Mean:   {np.mean(max_diffs):.6e}")
        beauty_print(f"  Median: {np.median(max_diffs):.6e}")
        beauty_print(f"  Max:    {np.max(max_diffs):.6e}")
        beauty_print(f"  Min:    {np.min(max_diffs):.6e}")
    
    beauty_print("✓ Bimanual Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    # Bessica is a dual-arm robot
    # Note: PyTorch Kinematics requires URDF format
    model_path = synriard.get_model_path("Bessica_D", version="v1_1", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Jacobian validation with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--q-left', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.1],
                        help='Left joint angles in radians')
    parser.add_argument('--q-right', type=float, nargs='+', default=[-0.1, -0.2, 0.3, 0.0, -0.5, 0.2, 0.1],
                        help='Right joint angles in radians')
    parser.add_argument('--mode', type=str, default='indep', choices=['indep', 'relative', 'mirror'],
                        help='Jacobian mode: indep (independent), relative (relative transform), mirror (mirror mode). Note: PyTorch Kinematics only supports indep mode directly.')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--rc-backend', type=str, default='torch', choices=['torch', 'cpp'],
                        help='RoboCore backend for Jacobian vs PK comparison')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
