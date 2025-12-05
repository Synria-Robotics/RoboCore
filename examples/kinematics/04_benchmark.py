#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kinematics FK/IK/Jacobian Benchmark

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

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from robocore.modeling import RobotModel
from robocore.kinematics import forward_kinematics, inverse_kinematics, jacobian
from robocore.utils.beauty_logger import beauty_print
from robocore.utils.path import get_robocore_path
import robocore

# Check PyTorch availability
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def benchmark_fk(model, q, backend, n_runs=1000, device=None):
    """Benchmark FK performance."""
    # Set global backend
    if backend == 'torch' and device:
        robocore.set_backend('torch', device=device)
    elif backend == 'numpy':
        robocore.set_backend('numpy')
    else:
        robocore.set_backend(backend)

    forward_kinematics(model, q, device=device)
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = forward_kinematics(model, q, device=device)
    return (time.perf_counter() - start) / n_runs * 1000


# ============================================================================
# Subcommand 1: Performance Benchmark (FK/IK/Jacobian speed)
# ============================================================================

def cmd_performance(args, model):
    """Run performance benchmark."""
    beauty_print("Performance Benchmark: FK/IK/Jacobian", type="module", centered=True)
    
    q = [0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.0][:model.num_chain_dof]
    
    beauty_print(f"FK runs: {args.fk_runs}", type="info")
    
    # FK benchmark
    beauty_print("Forward Kinematics")
    time_np = benchmark_fk(model, q, 'numpy', n_runs=args.fk_runs)
    beauty_print(f"  NumPy: {time_np:.4f} ms")
    
    if _HAS_TORCH:
        device = torch.device(args.torch_device)
        time_torch = benchmark_fk(model, q, 'torch', n_runs=args.fk_runs, device=device)
        beauty_print(f"  Torch ({device}): {time_torch:.4f} ms")
        speedup = time_torch/time_np if time_np < time_torch else time_np/time_torch
        faster = "NumPy" if time_np < time_torch else "Torch"
        beauty_print(f"  {faster} is {speedup:.2f}x faster", type="success")
    
    beauty_print("✓ Performance benchmark complete", type="success")


# ============================================================================
# Subcommand 2: IK Methods Comparison
# ============================================================================

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


def cmd_ik_compare(args, model):
    """Run IK methods comparison."""
    beauty_print("IK Methods Comparison", type="module", centered=True)
    
    rng = np.random.default_rng(args.seed)
    
    beauty_print(f"Samples: {args.samples}, Methods: {args.methods}, Backends: {args.backends}", type="info")
    beauty_print(f"Tolerances: pos={args.pos_tol:.1e}, ori={args.ori_tol:.1e}", type="info")
    if args.multi_start > 0:
        beauty_print(f"Multi-start: {args.multi_start} restarts, noise={args.multi_noise}", type="info")
    
    # Generate test cases
    qs = [model.random_q(rng) for _ in range(args.samples)]
    poses = [model.fk(q)['end'] for q in qs]
    
    # Build test matrix
    tests = []
    for backend in args.backends:
        for method in args.methods:
            tests.append((f"{backend}_{method}", backend, method))
    
    # Run tests
    results = {}
    for name, backend, method in tests:
        stats = {k: [] for k in ['iters', 'pos_err', 'ori_err', 'time', 'success']}

        # Prepare torch-specific kwargs
        extra = {}
        if backend == 'torch':
            extra['torch_device'] = args.torch_device
            if args.torch_dtype:
                try:
                    dtype_map = {
                        'float32': torch.float32, 'fp32': torch.float32,
                        'float': torch.float32,
                        'float64': torch.float64, 'double': torch.float64,
                    }
                    extra['torch_dtype'] = dtype_map.get(args.torch_dtype.lower(), torch.float32)
                except ImportError:
                    pass

        # Set global backend once per backend/method combination
        if backend == 'torch':
            robocore.set_backend('torch', device=extra.get('torch_device', 'cpu'))
        else:
            robocore.set_backend(backend)

        for pose in poses:
            q0 = np.zeros(model.num_chain_dof)

            t0 = time.perf_counter()
            try:
                res = inverse_kinematics(
                    model, pose, q0,
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
    beauty_print("Results")
    for name, _, _ in tests:
        if name in results:
            summarize(name, results[name])
    
    beauty_print("✓ IK comparison complete", type="success")


# ============================================================================
# Subcommand 3: Parallel/Batch Processing Benchmark
# ============================================================================

def benchmark_fk_numpy_batch(model, q_batch: np.ndarray, warmup: int = 5) -> Dict:
    """Benchmark FK computation using NumPy (sequential)."""
    batch_size = q_batch.shape[0]

    # Warmup
    warmup_batch = q_batch[:min(warmup, batch_size)]
    _ = forward_kinematics(model, warmup_batch, return_end=True)
    
    # Benchmark - use batch interface
    start_time = time.perf_counter()
    T_batch = forward_kinematics(model, q_batch, return_end=True)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    # Convert to list for compatibility
    if T_batch.ndim == 3:
        results = [T_batch[i] for i in range(batch_size)]
    else:
        results = [T_batch]

    return {
        'backend': 'numpy',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'results': results
    }


def benchmark_fk_torch_batch(model, q_batch: np.ndarray, device: str = 'cpu', warmup: int = 5) -> Dict:
    """Benchmark FK computation using PyTorch with TRUE BATCH operations."""
    if not _HAS_TORCH:
        return None
    
    batch_size = q_batch.shape[0]

    # Warmup - process a small batch
    warmup_batch = q_batch[:min(warmup, batch_size)]
    _ = forward_kinematics(model, warmup_batch, return_end=True, device=device)
    
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # Benchmark - use unified batch interface
    start_time = time.perf_counter()
    T_batch = forward_kinematics(model, q_batch, return_end=True, device=device)  # [B, 4, 4]
    
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    # Convert to list for compatibility
    if T_batch.ndim == 3:
        results = [T_batch[i] for i in range(batch_size)]
    else:
        results = [T_batch]
    
    return {
        'backend': f'torch_{device}',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'results': results
    }


def benchmark_ik_batch(model, target_poses: np.ndarray, q0_batch: np.ndarray,
                       backend: str, device: str = 'cpu', method: str = 'dls') -> Dict:
    """Benchmark IK batch computation."""
    batch_size = target_poses.shape[0]

    # Set backend
    if backend == 'torch':
        robocore.set_backend('torch', device=device)
    else:
        robocore.set_backend('numpy')

    # Warmup
    warmup_size = min(5, batch_size)
    warmup_poses = target_poses[:warmup_size]
    warmup_q0 = q0_batch[:warmup_size]
    _ = inverse_kinematics(model, warmup_poses, warmup_q0, method=method)

    if backend == 'torch' and _HAS_TORCH and device.startswith('cuda'):
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    results = inverse_kinematics(model, target_poses, q0_batch, method=method)
    if backend == 'torch' and _HAS_TORCH and device.startswith('cuda'):
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / batch_size
    successes = sum(1 for r in results if r.get('success', False))

    return {
        'backend': f'{backend}_{device}' if backend == 'torch' else backend,
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'success_rate': successes / batch_size,
        'results': results
    }


def benchmark_jacobian_batch(model, q_batch: np.ndarray, backend: str,
                             device: str = 'cpu', method: str = 'analytic') -> Dict:
    """Benchmark Jacobian batch computation."""
    batch_size = q_batch.shape[0]

    # Set backend
    if backend == 'torch':
        robocore.set_backend('torch', device=device)
    else:
        robocore.set_backend('numpy')

    # Warmup
    warmup_size = min(5, batch_size)
    warmup_batch = q_batch[:warmup_size]
    _ = jacobian(model, warmup_batch, method=method, device=device)

    if backend == 'torch' and _HAS_TORCH and device.startswith('cuda'):
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    J_batch = jacobian(model, q_batch, method=method, device=device)
    if backend == 'torch' and _HAS_TORCH and device.startswith('cuda'):
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / batch_size

    return {
        'backend': f'{backend}_{device}' if backend == 'torch' else backend,
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'results': J_batch
    }


def cmd_parallel(args, model):
    """Run parallel/batch processing benchmark."""
    beauty_print("Parallel/Batch Processing Benchmark", type="module", centered=True)
    beauty_print(f"Batch size: {args.batch_size}", type="info")
    
    # Generate random batch
    q_batch = model.random_q_batch(args.batch_size, seed=args.seed)
    
    # ========== Forward Kinematics ==========
    beauty_print("Forward Kinematics", type="module", centered=True)

    # NumPy benchmark
    beauty_print("NumPy (Batch Interface)")
    robocore.set_backend('numpy')
    result_np = benchmark_fk_numpy_batch(model, q_batch, warmup=5)
    beauty_print(f"  Total time: {result_np['total_time']:.4f} s")
    beauty_print(f"  Avg time: {result_np['avg_time']*1000:.4f} ms/sample")
    beauty_print(f"  Throughput: {result_np['throughput']:.2f} samples/s")
    
    # PyTorch CPU benchmark
    if _HAS_TORCH and 'torch-cpu' in args.backends:
        beauty_print("PyTorch CPU (Batch)")
        result_torch_cpu = benchmark_fk_torch_batch(model, q_batch, device='cpu', warmup=5)
        beauty_print(f"  Total time: {result_torch_cpu['total_time']:.4f} s")
        beauty_print(f"  Avg time: {result_torch_cpu['avg_time']*1000:.4f} ms/sample")
        beauty_print(f"  Throughput: {result_torch_cpu['throughput']:.2f} samples/s")
        speedup = result_np['total_time'] / result_torch_cpu['total_time']
        beauty_print(f"  Speedup over NumPy: {speedup:.2f}x", type="success" if speedup > 1 else "info")
    
    # PyTorch CUDA benchmark
    if _HAS_TORCH and torch.cuda.is_available() and 'torch-cuda' in args.backends:
        beauty_print("PyTorch CUDA (Batch)")
        result_torch_cuda = benchmark_fk_torch_batch(model, q_batch, device='cuda:0', warmup=5)
        beauty_print(f"  Total time: {result_torch_cuda['total_time']:.4f} s")
        beauty_print(f"  Avg time: {result_torch_cuda['avg_time']*1000:.4f} ms/sample")
        beauty_print(f"  Throughput: {result_torch_cuda['throughput']:.2f} samples/s")
        speedup = result_np['total_time'] / result_torch_cuda['total_time']
        beauty_print(f"  Speedup over NumPy: {speedup:.2f}x", type="success")
    
    # ========== Inverse Kinematics ==========
    if args.include_ik:
        beauty_print("Inverse Kinematics", type="module", centered=True)

        # Generate target poses from FK
        target_poses = []
        for q in q_batch[:min(50, args.batch_size)]:  # Limit IK batch size for speed
            T = forward_kinematics(model, q, return_end=True)
            target_poses.append(T)
        target_poses = np.array(target_poses)
        q0_batch = np.zeros((len(target_poses), model.num_chain_dof))

        # NumPy IK
        beauty_print("NumPy (Batch Interface)")
        result_ik_np = benchmark_ik_batch(model, target_poses, q0_batch, 'numpy', method='dls')
        beauty_print(f"  Total time: {result_ik_np['total_time']:.4f} s")
        beauty_print(f"  Avg time: {result_ik_np['avg_time']*1000:.4f} ms/sample")
        beauty_print(f"  Throughput: {result_ik_np['throughput']:.2f} samples/s")
        beauty_print(f"  Success rate: {result_ik_np['success_rate']*100:.1f}%")

        # PyTorch IK
        if _HAS_TORCH and 'torch-cpu' in args.backends:
            beauty_print("PyTorch CPU (Batch)")
            result_ik_torch = benchmark_ik_batch(model, target_poses, q0_batch, 'torch', 'cpu', method='dls')
            beauty_print(f"  Total time: {result_ik_torch['total_time']:.4f} s")
            beauty_print(f"  Avg time: {result_ik_torch['avg_time']*1000:.4f} ms/sample")
            beauty_print(f"  Throughput: {result_ik_torch['throughput']:.2f} samples/s")
            beauty_print(f"  Success rate: {result_ik_torch['success_rate']*100:.1f}%")
            speedup = result_ik_np['total_time'] / result_ik_torch['total_time']
            beauty_print(f"  Speedup over NumPy: {speedup:.2f}x", type="success" if speedup > 1 else "info")

    # ========== Jacobian ==========
    if args.include_jacobian:
        beauty_print("Jacobian", type="module", centered=True)

        # NumPy Jacobian
        beauty_print("NumPy (Batch Interface)")
        result_jac_np = benchmark_jacobian_batch(model, q_batch, 'numpy', method='analytic')
        beauty_print(f"  Total time: {result_jac_np['total_time']:.4f} s")
        beauty_print(f"  Avg time: {result_jac_np['avg_time']*1000:.4f} ms/sample")
        beauty_print(f"  Throughput: {result_jac_np['throughput']:.2f} samples/s")

        # PyTorch Jacobian
        if _HAS_TORCH and 'torch-cpu' in args.backends:
            beauty_print("PyTorch CPU (Batch)")
            result_jac_torch = benchmark_jacobian_batch(model, q_batch, 'torch', 'cpu', method='analytic')
            beauty_print(f"  Total time: {result_jac_torch['total_time']:.4f} s")
            beauty_print(f"  Avg time: {result_jac_torch['avg_time']*1000:.4f} ms/sample")
            beauty_print(f"  Throughput: {result_jac_torch['throughput']:.2f} samples/s")
            speedup = result_jac_np['total_time'] / result_jac_torch['total_time']
            beauty_print(f"  Speedup over NumPy: {speedup:.2f}x", type="success" if speedup > 1 else "info")

    beauty_print("✓ Parallel benchmark complete", type="success")


# ============================================================================
# Main Entry Point
# ============================================================================

def main(args):

    model = RobotModel(str(args.urdf), end_link=args.end_link)
    
    # Execute subcommand
    if args.command == 'performance':
        cmd_performance(args, model)
    elif args.command == 'ik-compare':
        cmd_ik_compare(args, model)
    elif args.command == 'parallel':
        cmd_parallel(args, model)
    elif args.command == 'all':
        # Run all benchmarks with default settings
        class DefaultArgs:
            fk_runs = 1000
            torch_device = args.torch_device
            samples = 100
            methods = ['pinv', 'dls']
            backends = ['numpy', 'torch']
            multi_start = 0
            multi_noise = 0.3
            pos_tol = 1e-4
            ori_tol = 1e-4
            torch_dtype = None
            batch_size = 1000
            include_ik = True
            include_jacobian = True
        
        default_args = DefaultArgs()
        default_args.seed = args.seed
        
        cmd_performance(default_args, model)
        cmd_ik_compare(default_args, model)
        cmd_parallel(default_args, model)
    else:
        parser.print_help()


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Unified Benchmark Suite for RoboCore Kinematics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global options
    parser.add_argument('--urdf', type=str, default=model_path, help='Path to URDF file')
    parser.add_argument('--end-link', type=str, default='tool0',
                        help='End-effector link name')
    parser.add_argument('--seed', type=int, default=77,
                        help='Random seed')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Benchmark type')

    # Subcommand: performance
    parser_perf = subparsers.add_parser('performance', help='FK/IK/Jacobian speed benchmark')
    parser_perf.add_argument('--fk-runs', type=int, default=1000,
                             help='Number of FK runs')
    parser_perf.add_argument('--torch-device', type=str, default='cuda',
                             help='PyTorch device (cpu, cuda)')

    # Subcommand: ik-compare
    parser_ik = subparsers.add_parser('ik-compare', help='Compare IK methods')
    parser_ik.add_argument('--samples', type=int, default=100,
                           help='Number of test samples')
    parser_ik.add_argument('--methods', nargs='+', default=['pinv', 'dls'],
                           help='IK methods to test (pinv, dls, transpose)')
    parser_ik.add_argument('--backends', nargs='+', default=['numpy', 'torch'],
                           help='Backends to test')
    parser_ik.add_argument('--multi-start', type=int, default=0,
                           help='Number of random restarts (0 to disable)')
    parser_ik.add_argument('--multi-noise', type=float, default=0.3,
                           help='Noise scale for restarts (radians)')
    parser_ik.add_argument('--pos-tol', type=float, default=1e-4,
                           help='Position tolerance (m)')
    parser_ik.add_argument('--ori-tol', type=float, default=1e-4,
                           help='Orientation tolerance (rad)')
    parser_ik.add_argument('--torch_device', type=str, default='cpu',
                           help='PyTorch device (cpu, cuda)')
    parser_ik.add_argument('--torch-dtype', type=str, default=None,
                           help='PyTorch dtype (float32, float64)')

    # Subcommand: parallel
    parser_par = subparsers.add_parser('parallel', help='Parallel/batch processing benchmark')
    parser_par.add_argument('--batch-size', type=int, default=100,
                            help='Batch size for parallel processing')
    parser_par.add_argument('--backends', nargs='+', default=['numpy', 'torch-cpu'],
                            help='Backends to test (numpy, torch-cpu, torch-cuda)')
    parser_par.add_argument('--include-ik', action='store_true',
                            help='Include IK batch benchmark')
    parser_par.add_argument('--include-jacobian', action='store_true',
                            help='Include Jacobian batch benchmark')

    # Subcommand: all
    parser_all = subparsers.add_parser('all', help='Run all benchmarks')
    parser_all.add_argument('--torch-device', type=str, default='cpu',
                            help='PyTorch device (cpu, cuda)')
    args = parser.parse_args()
    main(args)
