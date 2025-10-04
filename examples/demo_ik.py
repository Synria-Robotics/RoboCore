"""Comprehensive IK comparison: methods, backends, and performance.

Compares different IK methods (pinv, dls, transpose) across NumPy and PyTorch backends.
Provides detailed statistics on success rate, convergence, and timing.

Usage:
    python examples/demo_ik.py --samples 10 --backends numpy torch
    python examples/demo_ik.py --methods pinv dls --torch-device cpu
"""
from __future__ import annotations
import os
import argparse
import time
import numpy as np
from pathlib import Path
from robocore import RobotModel
from robocore.kinematics import inverse_kinematics
from robocore.utils.beauty_logger import beauty_print


def random_q(model, rng, scale=0.5):
    """Generate random joint configuration within limits."""
    q = [0.0] * model.dof()
    for js in model._actuated:  # type: ignore[attr-defined]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        q[js.index] = float(rng.uniform(mid - span, mid + span))
    return q


def summarize(name, stats):
    """Print summary statistics for one method/backend combination."""
    if not stats['iters']:
        print(f"{name:35s} | NO DATA")
        return
    
    succ_rate = 100.0 * np.mean(stats['success'])
    mean_iters = np.mean(stats['iters'])
    mean_pos = np.mean(stats['pos_err'])
    mean_ori = np.mean(stats['ori_err'])
    mean_time = 1e3 * np.mean(stats['time'])
    
    print(
        f"{name:35s} | "
        f"succ={succ_rate:5.1f}% | "
        f"iters={mean_iters:6.2f} | "
        f"pos_err={mean_pos:.3e} | "
        f"ori_err={mean_ori:.3e} | "
        f"time(ms)={mean_time:7.2f}"
    )


def main(args):
    rng = np.random.default_rng(args.seed)
    
    # Load model
    base = Path(__file__).resolve().parents[1]
    urdf = os.path.join(base, "robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf")
    model = RobotModel(str(urdf), end_link='tool0')
    
    beauty_print(f"IK Comparison: {model.name} ({model.dof()} DOF)", type="module")
    beauty_print(f"Samples: {args.samples}, Methods: {args.methods}, Backends: {args.backends}", type="info")
    beauty_print(f"Tolerances: pos={args.pos_tol:.1e}, ori={args.ori_tol:.1e}", type="info")
    if args.multi_start > 0:
        beauty_print(f"Multi-start: {args.multi_start} restarts, noise={args.multi_noise}", type="info")
    
    # Generate test cases
    qs = [random_q(model, rng) for _ in range(args.samples)]
    poses = [model.forward_kinematics(q)['end'] for q in qs]
    
    # Build test matrix
    tests = []
    for backend in args.backends:
        for method in args.methods:
            tests.append((f"{backend}_{method}", backend, method))
    
    # Run tests
    results = {}
    for name, backend, method in tests:
        stats = {k: [] for k in ['iters', 'pos_err', 'ori_err', 'time', 'success']}
        
        for pose in poses:
            q0 = np.zeros(model.dof())
            
            # Prepare torch-specific kwargs
            extra = {}
            if backend == 'torch':
                extra['torch_device'] = args.torch_device
                if args.torch_dtype:
                    try:
                        import torch
                        dtype_map = {
                            'float32': torch.float32, 'fp32': torch.float32,
                            'float': torch.float32,
                            'float64': torch.float64, 'double': torch.float64,
                        }
                        extra['torch_dtype'] = dtype_map.get(args.torch_dtype.lower(), torch.float32)
                    except ImportError:
                        pass
            
            t0 = time.perf_counter()
            try:
                res = inverse_kinematics(
                    model, pose, q0,
                    backend=backend,
                    method=method,
                    pos_tol=args.pos_tol,
                    ori_tol=args.ori_tol,
                    multi_start=args.multi_start,
                    multi_noise=args.multi_noise,
                    **extra,
                )
            except Exception as e:
                print(f"  {name} error: {e}")
                continue
            
            dt = time.perf_counter() - t0
            
            stats['iters'].append(res.get('iters', 0))
            stats['pos_err'].append(res.get('pos_err', np.nan))
            stats['ori_err'].append(res.get('ori_err', np.nan))
            stats['time'].append(dt)
            stats['success'].append(1.0 if res.get('success') else 0.0)
        
        results[name] = stats
    
    # Print results
    beauty_print("\nResults", type="module")
    for name, _, _ in tests:
        if name in results:
            summarize(name, results[name])
    
    beauty_print("✓ IK comparison complete", type="success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IK methods comparison")
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=77, help='Random seed')
    parser.add_argument('--methods', nargs='+', default=['pinv', 'dls'],
                        help='IK methods to test (pinv, dls, transpose). Note: transpose is slow')
    parser.add_argument('--backends', nargs='+', default=['numpy', 'torch'],
                        help='Backends to test')
    parser.add_argument('--multi-start', type=int, default=0,
                        help='Number of random restarts (0 to disable)')
    parser.add_argument('--multi-noise', type=float, default=0.3,
                        help='Noise scale for restarts (radians)')
    parser.add_argument('--pos-tol', type=float, default=1e-4,
                        help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-4,
                        help='Orientation tolerance (rad)')
    parser.add_argument('--torch-device', type=str, default='cpu',
                        help='PyTorch device (cpu, cuda)')
    parser.add_argument('--torch-dtype', type=str, default=None,
                        help='PyTorch dtype (float32, float64)')
    args = parser.parse_args()
    main(args)
