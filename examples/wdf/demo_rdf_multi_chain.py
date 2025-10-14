"""
Multi-Chain Robot Distance Field (RDF) Demo
============================================

This example demonstrates how to use RDF with multi-chain robots (e.g., arms with grippers).
Shows multi-chain usage patterns with 3D visualization.
"""

import argparse
import os
import numpy as np
import torch

from robocore.wdf.rdf import RDF
from robocore.modeling.robot_model import RobotModel
from robocore.utils.backend import set_backend, get_backend
from robocore.utils.path import get_robocore_path


def demo_multi_chain(args):
    """New multi-chain RDF usage"""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Chain RDF (New Feature)")
    print("="*60)
    
    # Create robot model WITHOUT specifying end_link (use full robot)
    asset_path = os.path.join(args.assetRoot, args.assetFile)
    robot = RobotModel(
        asset_path,
        base_link=args.baseLink,
        # NO end_link specified - use full robot
        load_mesh_flag=True
    )
    
    print(f"✓ Loaded full robot: {robot.num_dof} DOF")
    
    # Discover available end-effectors
    leaf_links = robot.available_leaf_links()
    print(f"  Available leaf links: {leaf_links}")
    
    # Define multiple chains (groups)
    # NOTE: Uncomment Link7/Link8 in MJCF first, or use available links
    groups = {
        'main_arm': 'Link6',  # Main arm chain
        # 'left_gripper': 'Link7',   # Uncomment if Link7 exists in MJCF
        # 'right_gripper': 'Link8',  # Uncomment if Link8 exists in MJCF
    }
    
    print(f"\n📦 Creating multi-chain RDF with groups: {list(groups.keys())}")
    
    # Create RDF with groups (multi-chain mode)
    rdf = RDF(args, robot, model_type=args.modelType, groups=groups)
    
    # Each group gets its own robot model
    for group_name, group_model in rdf.groups.items():
        print(f"  ✓ {group_name}: {group_model.num_chain_dof} DOF, end={group_model.end_link}")
    
    # Train if needed (trains for all groups)
    rdf_dir = os.path.join(os.path.dirname(asset_path), "rdf")
    model_name = f"{args.modelType}_{'h'+str(args.hiddenDim)+'_e'+str(args.trainEpochs) if args.modelType=='NN' else str(args.numFuncs)}"
    model_path = os.path.join(rdf_dir, args.modelType, f"{model_name}.pt")
    
    if not os.path.exists(model_path) or args.forceTrain:
        print("\nTraining multi-chain RDF...")
        rdf.train()
    
    # Load model
    if args.device == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        model = torch.load(model_path, weights_only=False)
    
    print(f"✓ Loaded model from {model_path}")
    
    # Example 1: Query specific chain
    print("\n🔍 Example 1: Query specific chain (main_arm)")
    joint_value_main = np.zeros(rdf.groups['main_arm'].num_chain_dof)
    points = np.random.rand(1, 64, 3) * 2.0 - 1.0
    
    sdf_main, grad_main = rdf.get_whole_body_sdf_batch(
        points,
        joint_value_main,
        model,
        base_trans=np.eye(4),
        use_derivative=True,
        group_name='main_arm'
    )
    
    # Convert to numpy if needed
    if hasattr(sdf_main, 'cpu'):
        sdf_main = sdf_main.detach().cpu().numpy()
    if grad_main is not None and hasattr(grad_main, 'cpu'):
        grad_main = grad_main.detach().cpu().numpy()
    
    print(f"✓ Main arm SDF shape: {sdf_main.shape}")
    print(f"  Min SDF: {np.min(sdf_main):.4f}, Max: {np.max(sdf_main):.4f}")
    
    # Example 2: Query all chains (minimum SDF)
    # NOTE: This feature is not yet fully implemented
    # TODO: Add support for dict-based joint_value to merge multiple chain SDFs
    print("\n🔍 Example 2: Query all chains - NOT YET IMPLEMENTED")
    print("  (Requires per-group SDF computation and merging logic)")
    
    """
    # Future implementation:
    print("\n🔍 Example 2: Query all chains (minimum SDF across all groups)")
    
    # Prepare joint values for all groups
    joint_values_all = {
        'main_arm': np.zeros(rdf.groups['main_arm'].num_chain_dof),
        # Add more groups if defined:
        # 'left_gripper': np.zeros(rdf.groups['left_gripper'].num_chain_dof),
        # 'right_gripper': np.zeros(rdf.groups['right_gripper'].num_chain_dof),
    }
    
    # Query with dict of joint values
    sdf_all, grad_all = rdf.get_whole_body_sdf_batch(
        points,
        joint_values_all,  # Pass dict for multi-chain
        model,
        base_trans=np.eye(4),
        use_derivative=True
    )
    
    # Convert to numpy if needed
    if hasattr(sdf_all, 'cpu'):
        sdf_all = sdf_all.detach().cpu().numpy()
    if grad_all is not None and hasattr(grad_all, 'cpu'):
        grad_all = grad_all.detach().cpu().numpy()
    
    print(f"✓ All chains SDF shape: {sdf_all.shape}")
    print(f"  Min SDF: {np.min(sdf_all):.4f}, Max: {np.max(sdf_all):.4f}")
    """
    
    print("\n✨ Multi-chain demo completed!")
    print("Note: Full multi-chain merging (querying all chains at once) is not yet implemented.")


def demo_backend_switching(args):
    """Demonstrate backend switching between numpy and torch
    
    NOTE: This demo is not yet fully implemented. Backend is currently set globally
    and RDF uses explicit torch conversion for NN inference.
    """
    print("\n" + "="*60)
    print("DEMO 3: Backend Switching - NOT YET FULLY IMPLEMENTED")
    print("="*60)
    print("Current implementation:")
    print("  - Backend is set globally via set_backend()")
    print("  - RDF always converts inputs to torch for NN inference")
    print("  - Outputs are converted back based on input type")
    print("\n✨ Backend demo skipped!")
    return
    
    """
    # Future implementation:
    asset_path = os.path.join(args.assetRoot, args.assetFile)
    robot = RobotModel(
        asset_path,
        base_link=args.baseLink,
        end_link="Link6",
        load_mesh_flag=True
    )
    
    # Test with NumPy backend
    print("\n🔧 Using NumPy backend...")
    set_backend('numpy')
    print(f"  Current backend: {get_backend()}")
    
    rdf_numpy = RDF(args, robot, model_type=args.modelType)
    print(f"  RDF device: {rdf_numpy.device}")
    
    # Test with Torch backend
    if torch.cuda.is_available():
        print("\n🔧 Using PyTorch backend (GPU)...")
        set_backend('torch', device='cuda:0')
        print(f"  Current backend: {get_backend()}")
        
        # Need to update args.device for RDF
        args_torch = argparse.Namespace(**vars(args))
        args_torch.device = 'cuda:0'
        
        rdf_torch = RDF(args_torch, robot, model_type=args.modelType)
        print(f"  RDF device: {rdf_torch.device}")
    else:
        print("\n🔧 Using PyTorch backend (CPU)...")
        set_backend('torch', device='cpu')
        print(f"  Current backend: {get_backend()}")
        
        args_torch = argparse.Namespace(**vars(args))
        args_torch.device = 'cpu'
        
        rdf_torch = RDF(args_torch, robot, model_type=args.modelType)
        print(f"  RDF device: {rdf_torch.device}")
    
    print("\n✓ Backend switching works correctly!")
    print("  Note: RDF internally adapts to the backend automatically")
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Chain RDF Demo')
    
    # Device args
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
    
    # Training args
    parser.add_argument('--trainEpochs', default=300, type=int)
    parser.add_argument('--forceTrain', action='store_true')
    parser.add_argument('--saveMeshDict', action='store_false')
    parser.add_argument('--samplePoints', action='store_false')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--domainMax', default=1.0, type=float)
    parser.add_argument('--domainMin', default=-1.0, type=float)
    
    # Model args
    parser.add_argument('--modelType', default="NN", type=str, choices=["NN", "BP"])
    parser.add_argument('--hiddenDim', default=256, type=int)
    parser.add_argument('--learningRate', default=1e-4, type=float)
    parser.add_argument('--nnBatchSize', default=819200, type=int)
    parser.add_argument('--numFuncs', default=16, type=int)
    
    # Asset args
    parser.add_argument('--assetRoot', default=get_robocore_path("assets"), type=str)
    parser.add_argument('--assetFile', default="robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml", type=str)
    parser.add_argument('--baseLink', default="base_link", type=str)
    
    # Demo selection
    parser.add_argument('--demo', default='all', choices=['single', 'multi', 'backend', 'all'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    # Set global backend
    set_backend('torch' if 'cuda' in args.device else 'numpy', device=args.device)
    print(f"🌍 Global backend: {get_backend()}")
    
    # Run selected demos
    if args.demo in ['single', 'all']:
        demo_single_chain(args)
    
    if args.demo in ['multi', 'all']:
        demo_multi_chain(args)
    
    if args.demo in ['backend', 'all']:
        demo_backend_switching(args)
    
    print("\n" + "="*60)
    print("✨ All demos completed!")
    print("="*60)
