#!/usr/bin/env python3
"""Tree-based robot model demonstration.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import torch
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print


def demo_tree_based_robot():
    """Demonstrate tree-based robot model."""
    beauty_print("=== Tree-based Robot Model Demo ===", type="module", centered=True)
    
    # Load robot model
    model_path = get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml")
    
    try:
        robot = RobotModel(model_path)
        
        beauty_print(f"Robot: {robot.name}")
        beauty_print(f"DOF: {robot.num_dof}")
        beauty_print(f"Base Link: {robot.base_link}")
        beauty_print(f"End Link: {robot.end_link}")
        
        # Print tree structure
        beauty_print("\nTree Structure:", type="info")
        robot.print_tree()
        
        # Auto-discover chains
        beauty_print("\nAuto-discovered Chains:", type="info")
        chains = robot.auto_discover_chains()
        for name, end_link in chains.items():
            beauty_print(f"  {name}: {end_link}")
        
        # Test joint information
        beauty_print("\nJoint Information:", type="info")
        beauty_print(f"Joint List: {robot.joint_list}")
        beauty_print(f"Joint Limits: {robot.joint_limit}")
        
        # Test link information
        beauty_print("\nLink Information:", type="info")
        beauty_print(f"Real Links: {len(robot.real_link)}")
        beauty_print(f"All Links: {len(robot.all_link)}")
        
        # Test random joint configuration
        beauty_print("\nRandom Joint Configuration:", type="info")
        q = robot.random_q()
        beauty_print(f"Joint Config: {q}")
        
        # Test forward kinematics for all frames
        beauty_print("\nForward Kinematics for All Frames:", type="info")
        try:
            fk_result = robot.forward_kinematics(q)
            beauty_print(f"✓ FK computed for {len(fk_result)} frames")
            
            # Show key frame transforms
            key_frames = ['base_link', 'Link6', 'Link7', 'Link8']
            for frame_name in key_frames:
                if frame_name in fk_result:
                    transform = fk_result[frame_name]
                    beauty_print(f"  {frame_name}: transform shape {transform.shape}")
            
            # Test batch FK
            beauty_print("\nBatch Forward Kinematics:", type="info")
            batch_q = torch.rand(3, robot.num_dof)  # 3 random configurations
            batch_fk = robot.forward_kinematics(batch_q)
            beauty_print(f"✓ Batch FK computed for {len(batch_fk)} frames")
            beauty_print(f"  Batch shape: {batch_fk[robot.end_link].shape}")
            
        except Exception as e:
            beauty_print(f"⚠️ FK computation failed: {e}", type="warning")
        
        # Test serial chain extraction
        if chains:
            beauty_print("\nSerial Chain Extraction:", type="info")
            chain_name = list(chains.keys())[0]
            end_link = chains[chain_name]
            
            chain = robot.extract_chain(end_link)
            beauty_print(f"Chain Name: {chain_name}")
            beauty_print(f"Chain DOF: {chain.num_dof}")
            beauty_print(f"Chain Joints: {chain.joint_list}")
            
            # Test chain-specific FK
            try:
                chain_q = q[:chain.num_dof]  # Use first N joints for chain
                chain_fk = chain.forward_kinematics(chain_q)
                beauty_print(f"✓ Chain FK computed: {type(chain_fk)}")
            except Exception as e:
                beauty_print(f"⚠️ Chain FK failed: {e}", type="warning")
        
        # Print summary
        robot.summary(show_chain=True)
        
        beauty_print("\n✓ Demo completed successfully", type="success")
        
    except Exception as e:
        beauty_print(f"✗ Demo failed: {e}", type="error")
        import traceback
        traceback.print_exc()


def demo_multi_chain_fk():
    """Demonstrate multi-chain forward kinematics like pytorch_kinematics."""
    beauty_print("\n=== Multi-Chain FK Demo (pytorch_kinematics style) ===", type="module", centered=True)
    
    model_path = get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml")
    
    try:
        robot = RobotModel(model_path)
        
        beauty_print("🤖 Alicia Robot Structure:", type="info")
        robot.print_tree()
        
        # Generate random joint configuration
        q = robot.random_q()
        beauty_print(f"\nJoint Configuration: {q}")
        
        # Compute FK for ALL frames at once (like pytorch_kinematics)
        beauty_print("\n🚀 Computing FK for ALL frames simultaneously:", type="info")
        fk_result = robot.forward_kinematics(q)
        
        beauty_print(f"✓ Computed transforms for {len(fk_result)} frames:")
        for frame_name, transform in fk_result.items():
            beauty_print(f"  {frame_name}: {transform.shape}")
        
        # Show the key insight: Link6 is shared by both branches
        beauty_print("\n💡 Key Insight - Shared Transform Reuse:", type="success")
        beauty_print("• base_link → Link6: Shared path for both branches")
        beauty_print("• Link6 → Link7: Left finger branch")
        beauty_print("• Link6 → Link8: Right finger branch")
        beauty_print("• pytorch_kinematics computes Link6 once, reuses for both branches")
        
        # Demonstrate batch computation
        beauty_print("\n📊 Batch FK Computation:", type="info")
        batch_size = 5
        batch_q = torch.rand(batch_size, robot.num_dof)
        batch_fk = robot.forward_kinematics(batch_q)
        
        beauty_print(f"✓ Batch FK for {batch_size} configurations:")
        beauty_print(f"  Each frame transform shape: {batch_fk[robot.end_link].shape}")
        
        # Show parent indices (the key to pytorch_kinematics efficiency)
        beauty_print("\n🔗 Parent Indices (Depth-First Indexing):", type="info")
        beauty_print("This is how pytorch_kinematics achieves efficiency:")
        for i, parent_path in enumerate(robot.tree_model.parents_indices):
            frame_name = robot.tree_model.idx_to_frame[i]
            path_str = " → ".join([robot.tree_model.idx_to_frame[p.item()] for p in parent_path])
            beauty_print(f"  {frame_name}: [{path_str}]")
        
        beauty_print("\n✨ Efficiency Benefits:", type="success")
        beauty_print("• Single computation for all frames")
        beauty_print("• Automatic transform reuse (Link6 computed once)")
        beauty_print("• Batch processing support")
        beauty_print("• No manual chain splitting needed")
        
        beauty_print("\n✓ Multi-chain FK demo completed", type="success")
        
    except Exception as e:
        beauty_print(f"✗ Multi-chain FK demo failed: {e}", type="error")
        import traceback
        traceback.print_exc()


def demo_compatibility():
    """Demonstrate compatibility interfaces."""
    beauty_print("\n=== Compatibility Interface Demo ===", type="module", centered=True)
    
    model_path = get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml")
    
    try:
        robot = RobotModel(model_path)
        
        # Test compatibility methods
        beauty_print("Testing compatibility methods:")
        
        # Joint parameter names
        joint_names = robot.get_joint_parameter_names()
        beauty_print(f"✓ Joint Parameter Names: {len(joint_names)} joints")
        
        # Frame names
        frame_names = robot.get_frame_names()
        beauty_print(f"✓ Frame Names: {len(frame_names)} frames")
        
        # Link names
        link_names = robot.get_link_names()
        beauty_print(f"✓ Link Names: {len(link_names)} links")
        
        # Joint limits
        limits = robot.get_joint_limits()
        beauty_print(f"✓ Joint Limits: {len(limits[0])} joints")
        
        # Leaf links
        leaf_links = robot.available_leaf_links()
        beauty_print(f"✓ Leaf Links: {leaf_links}")
        
        # Find functions
        if frame_names:
            frame = robot.find_frame(frame_names[0])
            beauty_print(f"✓ Find Frame '{frame_names[0]}': {frame is not None}")
        
        if link_names:
            link = robot.find_link(link_names[0])
            beauty_print(f"✓ Find Link '{link_names[0]}': {link is not None}")
        
        if joint_names:
            joint = robot.find_joint(joint_names[0])
            beauty_print(f"✓ Find Joint '{joint_names[0]}': {joint is not None}")
        
        beauty_print("✓ Compatibility interface demo completed", type="success")
        
    except Exception as e:
        beauty_print(f"✗ Compatibility demo failed: {e}", type="error")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_tree_based_robot()
    demo_multi_chain_fk()
    demo_compatibility()
    
    beauty_print("\n=== Demo Complete ===", type="module", centered=True)
