"""RDF with Multi-Chain FK Demo

Demonstrates how RDF now uses multi-chain FK to compute distance fields
for the complete robot including grippers and all branches.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import torch
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def main():
    beauty_print("RDF Multi-Chain FK Integration Demo", type="module")
    
    # Load robot model (complete tree structure)
    model_path = get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml")
    robot_model = RobotModel(str(model_path))  # Don't specify end_link to load full tree
    
    beauty_print(f"Robot: {robot_model.name}")
    beauty_print(f"Total DOF: {robot_model.num_dof}")
    beauty_print(f"Links in tree: {robot_model._num_links_in_tree}")
    beauty_print(f"All links: {robot_model.all_link}")
    print()
    
    # Test joint configuration (8 DOF including gripper)
    # Joint1-6: arm, left_finger, right_finger: gripper
    q = np.array([0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.01, -0.01])
    
    beauty_print("[1] Test get_trans_dict with Multi-Chain FK", type="module", centered=False)
    
    # Get transformations for all links
    trans_dict = robot_model.get_trans_dict(q)
    
    beauty_print(f"Computed transformations for {len(trans_dict)} links:")
    for link_name in sorted(trans_dict.keys()):
        T = trans_dict[link_name]
        pos = T[:3, 3]
        beauty_print(f"  {link_name:15s}: position = {beauty_print_array(pos)}")
    print()
    
    # Verify gripper links are included
    beauty_print("[2] Verify Gripper Links", type="module", centered=False)
    
    gripper_links = ['Link7', 'Link8']
    for link in gripper_links:
        if link in trans_dict:
            T = trans_dict[link]
            pos = T[:3, 3]
            beauty_print(f"✓ {link} found: position = {beauty_print_array(pos)}", type="success")
        else:
            beauty_print(f"✗ {link} NOT found in trans_dict", type="error")
    print()
    
    # Test with different gripper positions
    beauty_print("[3] Test Gripper Motion", type="module", centered=False)
    
    q_closed = list(q[:6]) + [0.0, 0.0]  # Gripper closed
    q_open = list(q[:6]) + [0.02, -0.02]  # Gripper open
    
    trans_closed = robot_model.get_trans_dict(q_closed)
    trans_open = robot_model.get_trans_dict(q_open)
    
    for link in gripper_links:
        if link in trans_closed and link in trans_open:
            pos_closed = trans_closed[link][:3, 3]
            pos_open = trans_open[link][:3, 3]
            displacement = np.linalg.norm(pos_open - pos_closed)
            beauty_print(f"{link}: displacement = {displacement:.6f} m", type="info")
    print()
    
    # Show advantage of multi-chain FK
    beauty_print("[4] Multi-Chain FK Benefits for WDF", type="module", centered=False)
    beauty_print("✓ One FK call computes ALL link transformations", type="success")
    beauty_print("✓ Supports tree structures (branches, grippers, multiple arms)", type="success")
    beauty_print("✓ Transform reuse improves efficiency", type="success")
    beauty_print("✓ Essential for WDF: needs complete robot geometry", type="success")
    print()
    
    # Demonstrate RDF usage (if model exists)
    beauty_print("[5] RDF Integration", type="module", centered=False)
    beauty_print("RDF automatically uses multi-chain FK through get_trans_dict()")
    beauty_print("This means:")
    beauty_print("  1. All links (including gripper) are considered in SDF calculation")
    beauty_print("  2. More accurate collision detection")
    beauty_print("  3. Complete robot distance field representation")
    print()
    
    # Check if multi-chain indexing is available
    if hasattr(robot_model, '_link_to_idx'):
        beauty_print(f"✓ Multi-chain indexing system active", type="success")
        beauty_print(f"  Links indexed: {robot_model._num_links_in_tree}")
        
        # Show depth-first ordering
        beauty_print("\n  Depth-first link ordering:")
        for idx in range(min(robot_model._num_links_in_tree, 10)):
            link_name = robot_model._idx_to_link[idx]
            joint_spec = robot_model._link_joints[idx]
            if joint_spec:
                joint_info = f"{joint_spec.joint_type:10s} '{joint_spec.name}'"
            else:
                joint_info = "ROOT"
            beauty_print(f"    [{idx}] {link_name:15s} <- {joint_info}")
    else:
        beauty_print(f"✗ Multi-chain indexing NOT available", type="error")
    
    print()
    beauty_print("=" * 80)
    beauty_print("Summary", type="module")
    beauty_print("=" * 80)
    beauty_print("RDF now uses multi-chain FK to compute distance fields for the COMPLETE robot.")
    beauty_print("This includes all links in the kinematic tree, not just the main chain.")
    beauty_print("Benefits: More accurate SDF, proper gripper handling, support for complex robots.")
    beauty_print("=" * 80)


if __name__ == "__main__":
    main()

