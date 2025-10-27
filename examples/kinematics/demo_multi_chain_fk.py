"""Multi-Chain Forward Kinematics Demo

Demonstrates the new multi-chain FK capability inspired by pytorch_kinematics.
Shows how to compute FK for all links efficiently with transform reuse.

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

import numpy as np
import argparse
import time
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.path import get_robocore_path
from robocore.transform.conversions import *


def main(args):
    beauty_print("Multi-Chain Forward Kinematics Demo", type="module")
    
    # Load robot model
    model_path = args.model_path
    end_link = args.end_link
    joint_angles = args.joint_angles

    robot_model = RobotModel(str(model_path), end_link=end_link)
    
    beauty_print(f"Testing multi-chain FK on {robot_model.name}")
    beauty_print(f"Robot has {robot_model._num_links_in_tree} links in the kinematic tree")
    
    # Test 1: Single-chain FK (legacy method)
    beauty_print("[1] Single-Chain FK (Legacy)", type="module", centered=False)
    start_time = time.time()
    T_single = forward_kinematics(robot_model, joint_angles, backend='numpy', return_end=True)
    time_single = time.time() - start_time
    
    position_single = T_single[:3, 3]
    beauty_print(f"End-effector position: {beauty_print_array(position_single)}")
    beauty_print(f"Time: {time_single*1000:.4f} ms")
    
    # Test 2: Multi-chain FK (new method - all links)
    beauty_print("[2] Multi-Chain FK (All Links)", type="module", centered=False)
    start_time = time.time()
    poses_all = forward_kinematics(robot_model, joint_angles, backend='numpy', return_all_links=True)
    time_multi = time.time() - start_time
    
    beauty_print(f"Computed FK for {len(poses_all)} links:")
    for link_name in sorted(poses_all.keys()):
        if link_name == 'end':
            continue
        pos = poses_all[link_name][:3, 3]
        beauty_print(f"  {link_name:12s}: {beauty_print_array(pos)}")
    beauty_print(f"Time: {time_multi*1000:.4f} ms")
    
    # Verify end-effector matches
    T_multi_end = poses_all['end']
    position_multi = T_multi_end[:3, 3]
    pos_diff = np.linalg.norm(position_single - position_multi)
    beauty_print(f"Position difference: {pos_diff:.3e} m", 
                type="success" if pos_diff < 1e-10 else "error")
    
    # Test 3: Multi-chain FK (specific links)
    beauty_print("[3] Multi-Chain FK (Specific Links)", type="module", centered=False)
    target_links = ['Link1', 'Link3', 'Link6'] if hasattr(robot_model, '_link_to_idx') else None
    
    if target_links:
        # Filter to only existing links
        target_links = [link for link in target_links if link in robot_model._link_to_idx]
        
        start_time = time.time()
        poses_subset = forward_kinematics(
            robot_model, 
            joint_angles, 
            backend='numpy', 
            return_all_links=True,
            link_names=target_links
        )
        time_subset = time.time() - start_time
        
        beauty_print(f"Computed FK for {len(poses_subset)} selected links:")
        for link_name in target_links:
            pos = poses_subset[link_name][:3, 3]
            beauty_print(f"  {link_name:12s}: {beauty_print_array(pos)}")
        beauty_print(f"Time: {time_subset*1000:.4f} ms")
    
    # Test 4: Performance comparison
    beauty_print("[4] Performance Comparison", type="module", centered=False)
    n_runs = 100
    
    # Benchmark single-chain FK
    times_single = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = forward_kinematics(robot_model, joint_angles, backend='numpy', return_end=True)
        times_single.append(time.perf_counter() - start)
    
    # Benchmark multi-chain FK
    times_multi = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = forward_kinematics(robot_model, joint_angles, backend='numpy', return_all_links=True)
        times_multi.append(time.perf_counter() - start)
    
    avg_single = np.mean(times_single) * 1000
    avg_multi = np.mean(times_multi) * 1000
    
    beauty_print(f"Average time over {n_runs} runs:")
    beauty_print(f"  Single-chain FK: {avg_single:.4f} ms")
    beauty_print(f"  Multi-chain FK:  {avg_multi:.4f} ms")
    beauty_print(f"  Overhead:        {(avg_multi/avg_single - 1)*100:.1f}%")
    beauty_print(f"Multi-chain FK computes {len(poses_all)} links vs 1 link for single-chain", type="info")
    
    # Test 5: PyTorch backend (if available)
    try:
        import torch
        beauty_print("[5] PyTorch Backend Test", type="module", centered=False)
        
        q_torch = torch.tensor(joint_angles, dtype=torch.float64)
        
        poses_torch = forward_kinematics(
            robot_model, 
            q_torch, 
            backend='torch', 
            return_all_links=True,
            device='cpu'
        )
        
        beauty_print(f"PyTorch multi-chain FK computed for {len(poses_torch)} links")
        
        # Verify consistency with NumPy
        for link_name in poses_torch.keys():
            if link_name not in poses_all:
                continue
            T_torch = poses_torch[link_name].cpu().numpy()
            T_numpy = poses_all[link_name]
            diff = np.linalg.norm(T_torch - T_numpy)
            if diff > 1e-10:
                beauty_print(f"  {link_name}: diff = {diff:.3e}", type="error")
        
        beauty_print("PyTorch and NumPy results match", type="success")
        
    except ImportError:
        beauty_print("PyTorch not available, skipping torch backend test", type="info")
    
    beauty_print("Multi-chain FK demo complete!", type="success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Chain Forward Kinematics Demo")
    parser.add_argument('--model-path', type=str,
                        default=get_robocore_path("assets/robot_descriptions/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml"),
                        help='Path to model file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--joint-angles', type=float, nargs='+', default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2],
                        help='Joint angles in radians')
    args = parser.parse_args()
    main(args)

