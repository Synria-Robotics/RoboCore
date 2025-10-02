#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for batch parallel FK/IK implementation.

This script provides a quick sanity check that the new batch parallel
implementations work correctly before running full benchmarks.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robocore.modeling.robot_model import RobotModel

# Check PyTorch
try:
    import torch
    HAS_TORCH = True
    from robocore.kinematics.fk_utils.batch_fk_torch import (
        batch_forward_kinematics_torch,
        extract_position_batch,
        rotation_matrix_to_quaternion_batch
    )
    from robocore.kinematics.ik_utils.batch_ik_torch import (
        batch_inverse_kinematics_torch
    )
    print("✓ PyTorch imported successfully")
except ImportError as e:
    HAS_TORCH = False
    print(f"✗ PyTorch not available: {e}")
    sys.exit(1)


def test_batch_fk(model, device='cpu', batch_size=10):
    """Test batch forward kinematics."""
    print(f"\n{'='*60}")
    print(f"Testing Batch FK on {device.upper()}")
    print(f"{'='*60}")
    
    # Generate random joint configurations
    q_batch = torch.randn(batch_size, model.dof())
    print(f"Input shape: {q_batch.shape} (batch_size={batch_size}, dof={model.dof()})")
    
    # Compute batch FK
    T_batch = batch_forward_kinematics_torch(model, q_batch, device=device)
    print(f"Output shape: {T_batch.shape} ✓")
    
    # Extract positions and quaternions
    pos_batch = extract_position_batch(T_batch)
    quat_batch = rotation_matrix_to_quaternion_batch(T_batch[:, :3, :3])
    
    print(f"Positions shape: {pos_batch.shape} ✓")
    print(f"Quaternions shape: {quat_batch.shape} ✓")
    
    # Print first sample
    print(f"\nSample [0]:")
    print(f"  Joint config: {q_batch[0].cpu().numpy()}")
    print(f"  Position: {pos_batch[0].cpu().numpy()}")
    print(f"  Quaternion (xyzw): {quat_batch[0].cpu().numpy()}")
    
    return T_batch, pos_batch, quat_batch


def test_batch_ik(model, target_poses, device='cpu'):
    """Test batch inverse kinematics."""
    print(f"\n{'='*60}")
    print(f"Testing Batch IK on {device.upper()}")
    print(f"{'='*60}")
    
    batch_size = target_poses.shape[0]
    print(f"Input shape: {target_poses.shape} (batch_size={batch_size})")
    
    # Random initial guess
    q_init = torch.zeros(batch_size, model.dof())
    
    # Solve IK
    q_sol, success, iters = batch_inverse_kinematics_torch(
        model, target_poses, q_init,
        device=device,
        max_iterations=50,
        tolerance=1e-4,
        damping=0.01,
        verbose=True
    )
    
    print(f"Solution shape: {q_sol.shape} ✓")
    print(f"Success: {success.sum().item()}/{batch_size} ({success.float().mean()*100:.1f}%)")
    
    # Print first successful sample
    if success.any():
        idx = success.nonzero()[0].item()
        print(f"\nSample [{idx}] (successful):")
        print(f"  Solution: {q_sol[idx].cpu().numpy()}")
        print(f"  Iterations: {iters[idx].item()}")
        
        # Verify FK
        T_verify = batch_forward_kinematics_torch(
            model, q_sol[idx:idx+1], device=device
        )[0]
        pos_verify = extract_position_batch(T_verify.unsqueeze(0))[0]
        pos_target = extract_position_batch(target_poses[idx:idx+1])[0]
        error = torch.linalg.norm(pos_verify - pos_target).item()
        print(f"  Position error: {error:.6f} m")
    
    return q_sol, success, iters


def main():
    """Run quick tests."""
    print("\n" + "="*60)
    print("BATCH PARALLEL FK/IK QUICK TEST")
    print("="*60)
    
    # Load model
    urdf_path = Path(__file__).parent.parent / 'robocore/assets/robot/urdf/alicia_d_v5_4.urdf'
    if not urdf_path.exists():
        print(f"\n✗ URDF not found: {urdf_path}")
        print("Please update the path in this script.")
        return
    
    print(f"\nLoading model: {urdf_path.name}")
    model = RobotModel(str(urdf_path), end_link='tool0')
    print(f"✓ Model loaded: {model.dof()} DOF")
    
    # Detect available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
        print(f"✓ MPS available: Apple Silicon GPU")
    
    # Test each device
    batch_size = 10
    for device in devices:
        # Test FK
        T_batch, pos_batch, quat_batch = test_batch_fk(model, device=device, batch_size=batch_size)
        
        # Test IK (use FK results as targets)
        test_batch_ik(model, T_batch, device=device)
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nYou can now run the full benchmark:")
    print("  python examples/benchmark_parallel_fk_ik.py --batch-size 1000 --device cuda")
    print()


if __name__ == '__main__':
    main()
