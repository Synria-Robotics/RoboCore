"""Jacobian validation and comparison with Pytorch Kinematics

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

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def main(args):
    # Load models
    model_path = str(args.model_path)
    end_link = args.end_link
    
    # PyTorch Kinematics
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link)
    n_dof = len(chain.get_joint_parameter_names())
    
    # RoboCore
    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=end_link)
    rc.set_backend(args.backend)
    
    beauty_print(f"Jacobian Comparison: PyTorch Kinematics vs RoboCore ({n_dof} DOF)", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Jacobian Computation", type="module", centered=False)
    q = torch.zeros(n_dof, dtype=dtype, device=device)
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q.cpu().numpy())}")
    
    # Compute Jacobians
    J_pk = chain.jacobian(q)
    if J_pk.ndim == 3:
        J_pk = J_pk.squeeze(0)
    J_pk_np = J_pk.cpu().numpy()
    
    J_rc = jacobian(rc_model, q, method='analytic', device=device)
    J_rc_np = to_numpy(J_rc)
    
    beauty_print(f"Jacobian shape: {J_pk_np.shape}")
    cond_num = float(np.linalg.cond(J_pk_np))
    beauty_print(f"Condition number: {cond_num:.2e}")
    
    beauty_print(f"Jacobian Matrix (PyTorch Kinematics):")
    print(beauty_print_array(J_pk_np, precision=6))
    
    beauty_print(f"Jacobian Matrix (RoboCore Analytic):")
    print(beauty_print_array(J_rc_np, precision=6))
    
    # Value comparison
    diff = J_pk_np - J_rc_np
    beauty_print("PyTorch Kinematics vs RoboCore:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff, 'fro'):.3e}")
    
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
    
    time_pk = benchmark(lambda: chain.jacobian(q))
    time_rc = benchmark(lambda: jacobian(rc_model, q, method='analytic', device=device))
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"RoboCore Analytic:   {time_rc:.4f} ms")
    speedup = time_pk / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup:             {speedup:.2f}x", type="success" if speedup > 1 else "info")
    
    # Condition number statistics and value comparison
    beauty_print(f"[3] Condition number and value comparison across {args.samples} random configurations", type="module", centered=False)
    condition_numbers = []
    max_diffs = []
    for i in range(args.samples):
        q_rand = torch.tensor(
            rng.uniform(-np.pi/2, np.pi/2, n_dof),
            dtype=dtype,
            device=device
        )
        
        # PyTorch Kinematics
        J_pk_rand = chain.jacobian(q_rand)
        if J_pk_rand.ndim == 3:
            J_pk_rand = J_pk_rand.squeeze(0)
        J_pk_rand_np = J_pk_rand.cpu().numpy()
        
        # RoboCore
        J_rc_rand = jacobian(rc_model, q_rand, method='analytic', device=device)
        J_rc_rand_np = to_numpy(J_rc_rand)
        
        # Condition number
        cond_num = float(np.linalg.cond(J_pk_rand_np))
        condition_numbers.append(cond_num)
        
        # Value difference
        diff = J_pk_rand_np - J_rc_rand_np
        max_diffs.append(np.max(np.abs(diff)))
    
    beauty_print(f"Condition number statistics:")
    beauty_print(f"  Mean:   {np.mean(condition_numbers):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers):.2e}")
    
    beauty_print(f"Value difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(max_diffs):.6e}")
    beauty_print(f"  Median: {np.median(max_diffs):.6e}")
    beauty_print(f"  Max:    {np.max(max_diffs):.6e}")
    beauty_print(f"  Min:    {np.min(max_diffs):.6e}")
    
    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='world', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name') 
    parser.add_argument('--backend', type=str, default='torch', choices=['numpy', 'torch'],
                        help='Backend to use for RoboCore (default: numpy)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
