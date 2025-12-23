"""Jacobian validation and comparison with Pytorch Kinematics and Pinocchio

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
    
    # Pinocchio
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()

    # Map RoboCore actuated joints to Pinocchio joints
    actuated_joint_names = [js.name for js in rc_model._chain_actuated]
    pin_q_indices = []
    for joint_name in actuated_joint_names:
        if pin_model.existJointName(joint_name):
            joint_id = pin_model.getJointId(joint_name)
            if joint_id < len(pin_model.idx_qs):
                pin_q_indices.append(pin_model.idx_qs[joint_id])
            else:
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

    beauty_print(f"Jacobian Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore ({n_dof} DOF)", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Jacobian Computation", type="module", centered=False)
    q = torch.zeros(n_dof, dtype=dtype, device=device)
    q_np = q.cpu().numpy()
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q_np)}")
    
    # Compute Jacobians
    J_pk = chain.jacobian(q)
    if J_pk.ndim == 3:
        J_pk = J_pk.squeeze(0)
    J_pk_np = J_pk.cpu().numpy()
    
    # Pinocchio Jacobian
    q_pin_full = pinocchio.neutral(pin_model).copy()
    for i, pin_idx in enumerate(pin_q_indices):
        if pin_idx < len(q_pin_full):
            q_pin_full[pin_idx] = q_np[i]
    pinocchio.forwardKinematics(pin_model, pin_data, q_pin_full)
    if end_frame_id is not None:
        pinocchio.updateFramePlacements(pin_model, pin_data)
        pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full)
        J_pin_full = pinocchio.getFrameJacobian(pin_model, pin_data, end_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    elif end_joint_id is not None:
        pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full)
        J_pin_full = pinocchio.getJointJacobian(pin_model, pin_data, end_joint_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    else:
        raise ValueError(f"Could not find frame or joint for {end_link}")
    # Extract columns corresponding to actuated joints
    pin_v_indices = []
    for joint_name in actuated_joint_names:
        if pin_model.existJointName(joint_name):
            joint_id = pin_model.getJointId(joint_name)
            if joint_id < len(pin_model.idx_vs):
                pin_v_indices.append(pin_model.idx_vs[joint_id])
            else:
                pin_v_indices.append(joint_id)
    J_pin_np = J_pin_full[:, pin_v_indices] if len(pin_v_indices) > 0 else J_pin_full

    J_rc = jacobian(rc_model, q_np, method='analytic', device=device)
    J_rc_np = to_numpy(J_rc)
    
    beauty_print(f"Jacobian shape (PyTorch Kinematics): {J_pk_np.shape}")
    beauty_print(f"Jacobian shape (Pinocchio): {J_pin_np.shape}")
    beauty_print(f"Jacobian shape (RoboCore): {J_rc_np.shape}")
    cond_num_pk = float(np.linalg.cond(J_pk_np))
    cond_num_pin = float(np.linalg.cond(J_pin_np))
    cond_num_rc = float(np.linalg.cond(J_rc_np))
    beauty_print(f"Condition number (PyTorch Kinematics): {cond_num_pk:.2e}")
    beauty_print(f"Condition number (Pinocchio): {cond_num_pin:.2e}")
    beauty_print(f"Condition number (RoboCore): {cond_num_rc:.2e}")
    
    beauty_print(f"Jacobian Matrix (PyTorch Kinematics):")
    print(beauty_print_array(J_pk_np, precision=6))
    
    beauty_print(f"Jacobian Matrix (Pinocchio):")
    print(beauty_print_array(J_pin_np, precision=6))

    beauty_print(f"Jacobian Matrix (RoboCore Analytic):")
    print(beauty_print_array(J_rc_np, precision=6))
    
    # Value comparison
    diff_pk_rc = J_pk_np - J_rc_np
    diff_pin_rc = J_pin_np - J_rc_np
    diff_pk_pin = J_pk_np - J_pin_np
    beauty_print("PyTorch Kinematics vs RoboCore:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_pk_rc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_pk_rc, 'fro'):.3e}")
    beauty_print("Pinocchio vs RoboCore:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_pin_rc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_pin_rc, 'fro'):.3e}")
    beauty_print("PyTorch Kinematics vs Pinocchio:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_pk_pin)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_pk_pin, 'fro'):.3e}")
    
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
            pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full)
            J_pin_full = pinocchio.getFrameJacobian(pin_model, pin_data, end_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        elif end_joint_id is not None:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full)
            J_pin_full = pinocchio.getJointJacobian(pin_model, pin_data, end_joint_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        else:
            raise ValueError(f"Could not find frame or joint for {end_link}")
        return J_pin_full[:, pin_v_indices] if len(pin_v_indices) > 0 else J_pin_full

    time_pk = benchmark(lambda: chain.jacobian(q))
    time_pin = benchmark(benchmark_pin)
    time_rc = benchmark(lambda: jacobian(rc_model, q_np, method='analytic', device=device))
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"Pinocchio:           {time_pin:.4f} ms")
    beauty_print(f"RoboCore Analytic:   {time_rc:.4f} ms")
    speedup_pk_rc = time_pk / time_rc if time_rc > 0 else 0
    speedup_pin_rc = time_pin / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup (PK vs RC):  {speedup_pk_rc:.2f}x", type="success" if speedup_pk_rc > 1 else "info")
    beauty_print(f"Speedup (Pin vs RC): {speedup_pin_rc:.2f}x", type="success" if speedup_pin_rc > 1 else "info")
    
    # Condition number statistics and value comparison
    beauty_print(f"[3] Condition number and value comparison across {args.samples} random configurations", type="module", centered=False)
    condition_numbers_pk = []
    condition_numbers_pin = []
    condition_numbers_rc = []
    max_diffs_pk_rc = []
    max_diffs_pin_rc = []
    max_diffs_pk_pin = []
    for i in range(args.samples):
        q_rand = torch.tensor(rc_model.random_q(seed=args.seed), dtype=dtype, device=device)
        q_rand_np = q_rand.cpu().numpy()
        
        # PyTorch Kinematics
        J_pk_rand = chain.jacobian(q_rand)
        if J_pk_rand.ndim == 3:
            J_pk_rand = J_pk_rand.squeeze(0)
        J_pk_rand_np = J_pk_rand.cpu().numpy()
        
        # Pinocchio
        q_pin_full_rand = pinocchio.neutral(pin_model).copy()
        for i, pin_idx in enumerate(pin_q_indices):
            if pin_idx < len(q_pin_full_rand):
                q_pin_full_rand[pin_idx] = q_rand_np[i]
        pinocchio.forwardKinematics(pin_model, pin_data, q_pin_full_rand)
        if end_frame_id is not None:
            pinocchio.updateFramePlacements(pin_model, pin_data)
            pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full_rand)
            J_pin_full_rand = pinocchio.getFrameJacobian(pin_model, pin_data, end_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        elif end_joint_id is not None:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_pin_full_rand)
            J_pin_full_rand = pinocchio.getJointJacobian(pin_model, pin_data, end_joint_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        else:
            raise ValueError(f"Could not find frame or joint for {end_link}")
        J_pin_rand_np = J_pin_full_rand[:, pin_v_indices] if len(pin_v_indices) > 0 else J_pin_full_rand

        # RoboCore
        J_rc_rand = jacobian(rc_model, q_rand_np, method='analytic', device=device)
        J_rc_rand_np = to_numpy(J_rc_rand)
        
        # Condition number
        cond_num_pk = float(np.linalg.cond(J_pk_rand_np))
        cond_num_pin = float(np.linalg.cond(J_pin_rand_np))
        cond_num_rc = float(np.linalg.cond(J_rc_rand_np))
        condition_numbers_pk.append(cond_num_pk)
        condition_numbers_pin.append(cond_num_pin)
        condition_numbers_rc.append(cond_num_rc)
        
        # Value difference
        diff_pk_rc = J_pk_rand_np - J_rc_rand_np
        diff_pin_rc = J_pin_rand_np - J_rc_rand_np
        diff_pk_pin = J_pk_rand_np - J_pin_rand_np
        max_diffs_pk_rc.append(np.max(np.abs(diff_pk_rc)))
        max_diffs_pin_rc.append(np.max(np.abs(diff_pin_rc)))
        max_diffs_pk_pin.append(np.max(np.abs(diff_pk_pin)))

    beauty_print(f"Condition number statistics (PyTorch Kinematics):")
    beauty_print(f"  Mean:   {np.mean(condition_numbers_pk):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers_pk):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers_pk):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers_pk):.2e}")

    beauty_print(f"Condition number statistics (Pinocchio):")
    beauty_print(f"  Mean:   {np.mean(condition_numbers_pin):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers_pin):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers_pin):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers_pin):.2e}")
    
    beauty_print(f"Condition number statistics (RoboCore):")
    beauty_print(f"  Mean:   {np.mean(condition_numbers_rc):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers_rc):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers_rc):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers_rc):.2e}")
    
    beauty_print(f"Value difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(max_diffs_pk_rc):.6e}")
    beauty_print(f"  Median: {np.median(max_diffs_pk_rc):.6e}")
    beauty_print(f"  Max:    {np.max(max_diffs_pk_rc):.6e}")
    beauty_print(f"  Min:    {np.min(max_diffs_pk_rc):.6e}")

    beauty_print(f"Value difference statistics (Pinocchio vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(max_diffs_pin_rc):.6e}")
    beauty_print(f"  Median: {np.median(max_diffs_pin_rc):.6e}")
    beauty_print(f"  Max:    {np.max(max_diffs_pin_rc):.6e}")
    beauty_print(f"  Min:    {np.min(max_diffs_pin_rc):.6e}")

    beauty_print(f"Value difference statistics (PyTorch Kinematics vs Pinocchio):")
    beauty_print(f"  Mean:   {np.mean(max_diffs_pk_pin):.6e}")
    beauty_print(f"  Median: {np.median(max_diffs_pk_pin):.6e}")
    beauty_print(f"  Max:    {np.max(max_diffs_pk_pin):.6e}")
    beauty_print(f"  Min:    {np.min(max_diffs_pk_pin):.6e}")
    
    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation with Pytorch Kinematics and Pinocchio")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name') 
    parser.add_argument('--backend', type=str, default='torch', choices=['numpy', 'torch'],
                        help='Backend to use for RoboCore (default: numpy)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
