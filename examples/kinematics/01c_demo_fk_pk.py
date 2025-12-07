"""Forward Kinematics validation and comparison with Pytorch Kinematics

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
from robocore.kinematics.fk import forward_kinematics
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
    
    # RoboCore
    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=end_link)
    rc.set_backend('torch', device=args.device)
    
    beauty_print(f"Forward Kinematics Comparison: PyTorch Kinematics vs RoboCore ({n_dof} DOF)", type="module")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    
    beauty_print("[1] Forward Kinematics Computation", type="module", centered=False)
    q = torch.tensor(args.joint_angles, dtype=dtype, device=device)
    
    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q.cpu().numpy())}")
    
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
    
    # Compute FK with RoboCore
    # Note: RoboCore does not apply root link's origin transform. When base_link == end_link,
    # it returns identity matrix. For other cases, it computes relative transform from base_link
    # to end_link without including world2base rotation.
    T_rc = forward_kinematics(rc_model, q, return_end=True, device=device)
    T_rc_np = to_numpy(T_rc)
    pos_rc = T_rc_np[:3, 3]
    rot_rc = T_rc_np[:3, :3]
    
    beauty_print(f"End-Effector Position (PyTorch Kinematics):")
    print(f"  p = {beauty_print_array(pos_pk)}")
    beauty_print(f"End-Effector Position (RoboCore):")
    print(f"  p = {beauty_print_array(pos_rc)}")
    
    # Convert to quaternion for display
    quat_pk = matrix_to_quaternion(rot_pk)
    quat_rc = matrix_to_quaternion(rot_rc)
    beauty_print(f"Quaternion xyzw (PyTorch Kinematics):")
    print(f"  quat = {beauty_print_array(quat_pk, precision=6)}")
    beauty_print(f"Quaternion xyzw (RoboCore):")
    print(f"  quat = {beauty_print_array(quat_rc, precision=6)}")
    
    # Position comparison
    pos_diff = pos_pk - pos_rc
    beauty_print("Position Comparison (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(pos_diff)):.3e} m")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(pos_diff):.3e} m")

    # Rotation comparison
    rot_diff = rot_pk - rot_rc
    beauty_print("Rotation Matrix Comparison (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(rot_diff)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(rot_diff, 'fro'):.3e}")
    
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
    
    time_pk = benchmark(lambda: chain.forward_kinematics(q_tensor, end_only=False))
    time_rc = benchmark(lambda: forward_kinematics(rc_model, q, return_end=True, device=device))
    
    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup = time_pk / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup:             {speedup:.2f}x", type="success" if speedup > 1 else "info")
    
    # Value comparison across random configurations
    beauty_print(f"[3] Value comparison across {args.samples} random configurations", type="module", centered=False)
    pos_diffs = []
    rot_diffs = []
    for i in range(args.samples):
        q_rand = torch.tensor(rc_model.random_q(rng), dtype=dtype, device=device)
        
        # PyTorch Kinematics
        q_rand_tensor = q_rand.unsqueeze(0)
        ret_pk_rand = chain.forward_kinematics(q_rand_tensor, end_only=False)
        tg_pk_rand = ret_pk_rand[end_link]
        m_pk_rand = tg_pk_rand.get_matrix()[0]
        T_pk_rand = m_pk_rand.cpu().numpy()
        pos_pk_rand = T_pk_rand[:3, 3]
        rot_pk_rand = T_pk_rand[:3, :3]
        
        # RoboCore
        T_rc_rand = forward_kinematics(rc_model, q_rand, return_end=True, device=device)
        T_rc_rand_np = to_numpy(T_rc_rand)
        pos_rc_rand = T_rc_rand_np[:3, 3]
        rot_rc_rand = T_rc_rand_np[:3, :3]
        
        # Differences
        pos_diff_rand = pos_pk_rand - pos_rc_rand
        rot_diff_rand = rot_pk_rand - rot_rc_rand
        
        pos_diffs.append(np.linalg.norm(pos_diff_rand))
        rot_diffs.append(np.linalg.norm(rot_diff_rand, 'fro'))
    
    beauty_print(f"Position difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(pos_diffs):.6e} m")
    beauty_print(f"  Median: {np.median(pos_diffs):.6e} m")
    beauty_print(f"  Max:    {np.max(pos_diffs):.6e} m")
    beauty_print(f"  Min:    {np.min(pos_diffs):.6e} m")
    
    beauty_print(f"Rotation difference statistics (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Mean:   {np.mean(rot_diffs):.6e}")
    beauty_print(f"  Median: {np.median(rot_diffs):.6e}")
    beauty_print(f"  Max:    {np.max(rot_diffs):.6e}")
    beauty_print(f"  Min:    {np.min(rot_diffs):.6e}")
    
    beauty_print("✓ Forward kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Forward Kinematics validation with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='world', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)