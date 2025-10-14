#!/usr/bin/env python3
"""
Multi-Chain Robot Model Demo

This script demonstrates how to use the new MultiChainRobotModel class
to work with multiple serial chains from a single robot model.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
from robocore.modeling.robot_model import RobotModel, MultiChainRobotModel
from robocore.utils.beauty_logger import beauty_print


def demo_multi_chain_robot():
    """Demonstrate multi-chain robot model usage."""
    
    # Example robot model path (replace with your actual robot model)
    model_path = "/Users/Jonas/Documents/Github/Robotics/RoboCore/robocore/assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml"
    
    beauty_print("=== Multi-Chain Robot Model Demo ===", type="module", centered=True)
    
    # 1. First, let's explore available end links
    beauty_print("1. Exploring available end links...", type="info")
    try:
        robot = RobotModel(model_path)
        available_links = robot.available_end_links()
        beauty_print(f"Available end links: {available_links}", type="success")
        
        # Show robot structure
        robot.print_tree(show_joints=True, show_fixed=False)
        
    except Exception as e:
        beauty_print(f"Error loading robot model: {e}", type="error")
        beauty_print("Using mock data for demonstration...", type="warning")
        
        # Mock data for demonstration
        available_links = ["left_gripper", "right_gripper", "head_camera"]
    
    # 2. Create multi-chain robot model
    beauty_print("\n2. Creating multi-chain robot model...", type="info")
    
    # Define chains based on available end links
    if len(available_links) >= 2:
        chains = {
            "left_arm": available_links[0],  # Link7
            "right_arm": available_links[1]  # Link8
        }
    else:
        # Fallback for demonstration
        chains = {
            "main_chain": available_links[0] if available_links else "Link8"
        }
    
    try:
        multi_robot = MultiChainRobotModel(model_path, chains)
    except Exception as e:
        beauty_print(f"Error creating multi-chain model: {e}", type="error")
        beauty_print("This is expected if the robot model doesn't exist.", type="info")
        return
    
    # 3. Demonstrate multi-chain forward kinematics
    beauty_print("\n3. Multi-chain Forward Kinematics...", type="info")
    
    # Generate random joint configurations for each chain
    q_by_chain = {}
    for chain_name in chains.keys():
        chain_model = multi_robot.get_chain(chain_name)
        q_by_chain[chain_name] = chain_model.random_q()
        q_array = np.array(q_by_chain[chain_name])
        print(f"{chain_name}: q = {q_array}")
    
    # Compute FK for all chains
    fk_results = multi_robot.fk_multi(q_by_chain, backend='numpy')
    
    beauty_print("Forward Kinematics Results:", type="success")
    for chain_name, pose in fk_results.items():
        if isinstance(pose, dict):
            # Multiple poses returned
            for link_name, transform in pose.items():
                pos = transform[:3, 3]
                print(f"  {chain_name}.{link_name}: pos = {pos}")
        else:
            # Single pose returned
            pos = pose[:3, 3]
            print(f"  {chain_name}: pos = {pos}")
    
    # 4. Demonstrate multi-chain Jacobian computation
    beauty_print("\n4. Multi-chain Jacobian Computation...", type="info")
    
    jacobians = multi_robot.jacobian_multi(q_by_chain, backend='numpy')
    
    beauty_print("Jacobian Results:", type="success")
    for chain_name, J in jacobians.items():
        print(f"  {chain_name}: Jacobian shape = {J.shape}")
    
    # 5. Demonstrate individual chain access
    beauty_print("\n5. Individual Chain Access...", type="info")
    
    for chain_name in chains.keys():
        chain_model = multi_robot.get_chain(chain_name)
        print(f"Chain '{chain_name}':")
        print(f"  DOF: {chain_model.num_dof}")
        print(f"  Joints: {chain_model.joint_list}")
        print(f"  End link: {chain_model.end_link}")
    
    # 6. Show summary
    beauty_print("\n6. Multi-Chain Robot Summary...", type="info")
    multi_robot.summary(show_chains=True)


def demo_pytorch_kinematics_comparison():
    """Compare with pytorch_kinematics approach."""
    
    beauty_print("\n=== Comparison with pytorch_kinematics ===", type="module", centered=True)
    
    beauty_print("pytorch_kinematics approach:", type="info")
    beauty_print("""
    # 1. Load full robot model
    chain = pk.build_chain_from_urdf(urdf_data)
    
    # 2. Extract serial chains
    left_chain = pk.SerialChain(chain, "left_end_link", "base_link")
    right_chain = pk.SerialChain(chain, "right_end_link", "base_link")
    
    # 3. Independent FK computation
    left_fk = left_chain.forward_kinematics(q_left)
    right_fk = right_chain.forward_kinematics(q_right)
    """)
    
    beauty_print("RoboCore approach:", type="info")
    beauty_print("""
    # 1. Create multi-chain robot model
    multi_robot = MultiChainRobotModel(model_path, {
        "left_arm": "left_end_link",
        "right_arm": "right_end_link"
    })
    
    # 2. Simultaneous FK computation
    q_by_chain = {"left_arm": q_left, "right_arm": q_right}
    fk_results = multi_robot.fk_multi(q_by_chain)
    
    # 3. Individual chain access
    left_chain = multi_robot.get_chain("left_arm")
    right_chain = multi_robot.get_chain("right_arm")
    """)
    
    beauty_print("Key advantages of RoboCore approach:", type="success")
    beauty_print("• Unified interface for multi-chain operations")
    beauty_print("• Automatic chain discovery and management")
    beauty_print("• Consistent API with existing RobotModel")
    beauty_print("• Built-in support for complex multi-chain scenarios")


if __name__ == "__main__":
    demo_multi_chain_robot()
    demo_pytorch_kinematics_comparison()
    
    beauty_print("\n=== Demo Complete ===", type="module", centered=True)
