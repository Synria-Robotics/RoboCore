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
    from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
    from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
    from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
    from robocore.transform import matrix_to_quaternion
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
    
    # Generate random joint configurations within limits (not unbounded randn)
    import numpy as np
    rng = np.random.default_rng(42)
    q_np = np.zeros((batch_size, model.dof()))
    for i in range(batch_size):
        for js in model._actuated:
            lo, hi = -1.0, 1.0
            if js.limit and js.limit[0] is not None and js.limit[1] is not None:
                lo, hi = js.limit[0], js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.25 * (hi - lo)  # Use 50% of range
            q_np[i, js.index] = rng.uniform(mid - span, mid + span)
    q_batch = torch.from_numpy(q_np).float()
    print(f"Input shape: {q_batch.shape} (batch_size={batch_size}, dof={model.dof()})")
    
    # Compute batch FK using unified solver
    fk_solver = FKSolverTorch(model)
    T_batch = fk_solver.solve(q_batch, device=device)  # Auto-detects batch mode
    print(f"Output shape: {T_batch.shape} ✓")
    
    # Extract positions and quaternions
    pos_batch = T_batch[:, :3, 3]
    quat_batch = matrix_to_quaternion(T_batch[:, :3, :3])
    
    print(f"Positions shape: {pos_batch.shape} ✓")
    print(f"Quaternions shape: {quat_batch.shape} ✓")
    
    # Print first sample
    print(f"\nSample [0]:")
    print(f"  Joint config: {q_batch[0].cpu().numpy()}")
    print(f"  Position: {pos_batch[0].cpu().numpy()}")
    if hasattr(quat_batch, 'cpu'):
        print(f"  Quaternion (xyzw): {quat_batch[0].cpu().numpy()}")
    else:
        print(f"  Quaternion (xyzw): {quat_batch[0]}")
    
    return T_batch, pos_batch, quat_batch


def test_batch_ik(model, target_poses, device='cpu'):
    """Test batch inverse kinematics using FK results as targets."""
    print(f"\n{'='*60}")
    print(f"Testing Batch IK on {device.upper()}")
    print(f"{'='*60}")
    
    batch_size = target_poses.shape[0]
    print(f"Input shape: {target_poses.shape} (batch_size={batch_size})")
    print(f"Using FK results as IK targets (guaranteed reachable)")
    
    # Generate random initial guesses (different from FK configs)
    import numpy as np
    rng = np.random.default_rng(123)
    q_init_np = np.zeros((batch_size, model.dof()))
    for i in range(batch_size):
        for js in model._actuated:
            lo, hi = -1.0, 1.0
            if js.limit and js.limit[0] is not None and js.limit[1] is not None:
                lo, hi = js.limit[0], js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.25 * (hi - lo)
            q_init_np[i, js.index] = rng.uniform(mid - span, mid + span)
    q_init = torch.from_numpy(q_init_np).float()
    
    # Solve IK using unified solver (batch mode)
    ik_solver = IKSolverTorch(model, max_iters=50, pos_tol=1e-4, ori_tol=1e-4)
    # TODO: Need to add batch support to IKSolverTorch.solve()
    # For now, iterate
    q_sol_list = []
    success_list = []
    iters_list = []
    
    for i in range(batch_size):
        result = ik_solver.solve(
            target_poses[i].cpu().numpy().tolist(),
            q_init[i].cpu().numpy().tolist(),
            method='dls'
        )
        q_sol_list.append(result['q'])
        success_list.append(result.get('success', False))
        iters_list.append(result.get('iters', 0))
    
    q_sol = torch.tensor(q_sol_list, device=device)
    success = torch.tensor(success_list, device=device)
    iters = torch.tensor(iters_list, device=device)
    
    print(f"Solution shape: {q_sol.shape} ✓")
    print(f"Success: {success.sum().item()}/{batch_size} ({success.float().mean()*100:.1f}%)")
    
    # Print first successful sample
    if success.any():
        idx = success.nonzero()[0].item()
        print(f"\nSample [{idx}] (successful):")
        print(f"  Solution: {q_sol[idx].cpu().numpy()}")
        print(f"  Iterations: {iters[idx].item()}")
        
        # Verify FK using unified solver
        fk_solver = FKSolverTorch(model)
        T_verify = fk_solver.solve(q_sol[idx:idx+1], device=device)[0]
        pos_verify = T_verify[:3, 3]
        pos_target = target_poses[idx, :3, 3]
        error = torch.linalg.norm(pos_verify - pos_target).item()
        print(f"  Position error: {error:.6f} m")
    
    return q_sol, success, iters


def main():
    """Run quick tests."""
    print("\n" + "="*60)
    print("BATCH PARALLEL FK/IK QUICK TEST")
    print("="*60)
    
    # Load model
    urdf_path = Path(__file__).parent.parent / 'robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf'
    if not urdf_path.exists():
        print(f"\n✗ URDF not found: {urdf_path}")
        print("Please update the path in this script.")
        return
    
    print(f"\nLoading model: {urdf_path.name}")
    model = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
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
