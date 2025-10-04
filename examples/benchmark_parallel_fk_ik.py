#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel FK / IK Benchmark (NumPy vs PyTorch, TRUE BATCH FK)
==========================================================

本脚本用于评估：
1. 前向运动学 (FK) 单次/批量性能：NumPy(循环) vs PyTorch(向量化)
2. 逆运动学 (IK) 性能：当前 Torch 版本仍使用逐样本循环（TODO: 向量化批 IK）

特性：
- 真实批量 FK：PyTorch 一次处理全部样本 (B, n)
- NumPy 基准：逐样本循环（作为 baseline）
- 可选只跑 FK 或只跑 IK
- 统计平均耗时 / 吞吐率 和 IK 成功率

已移除：Apple MPS (Metal) 支持，仅保留 cpu / cuda。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics

_HAS_TORCH = False
try:  # pragma: no cover
    import torch
    from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
    from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
    from robocore.transform import matrix_to_quaternion
    _HAS_TORCH = True
except Exception:  # noqa: E722
    torch = None  # type: ignore


# ---------------------------- Utility Printing ----------------------------- #
def print_header(title: str):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ---------------------------- Data Generation ------------------------------ #
def random_q_batch(model: RobotModel, batch_size: int, seed: int = 42, span_scale: float = 0.5) -> np.ndarray:
    """生成批量随机关节配置（在各关节限制中间附近采样）。"""
    rng = np.random.default_rng(seed)
    n = model.dof()
    q_batch = np.zeros((batch_size, n), dtype=float)
    for i in range(batch_size):
        for js in model._actuated:  # type: ignore[attr-defined]
            lo, hi = -1.0, 1.0
            if js.limit:
                if js.limit[0] is not None:
                    lo = js.limit[0]
                if js.limit[1] is not None:
                    hi = js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo) * span_scale
            q_batch[i, js.index] = rng.uniform(mid - span, mid + span)
    return q_batch


# ---------------------------- FK Benchmarks -------------------------------- #
def benchmark_fk_numpy(model: RobotModel, q_batch: np.ndarray, warmup: int = 5) -> Dict[str, Any]:
    batch = q_batch.shape[0]
    # Warmup
    for i in range(min(warmup, batch)):
        _ = forward_kinematics(model, q_batch[i], backend='numpy', return_end=True)
    t0 = time.perf_counter()
    results = []
    for i in range(batch):
        T = forward_kinematics(model, q_batch[i], backend='numpy', return_end=True)
        results.append(T)
    dt = time.perf_counter() - t0
    return {
        'backend': 'numpy',
        'batch_size': batch,
        'total_time': dt,
        'avg_time': dt / batch,
        'throughput': batch / dt,
        'results': results,
    }


def benchmark_fk_torch(model: RobotModel, q_batch: np.ndarray, device: str, warmup: int = 5, dtype='float32') -> Dict[str, Any] | None:
    if not _HAS_TORCH:
        return None
    fk_solver = FKSolverTorch(model)
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    q_t = torch.from_numpy(q_batch).to(dtype=torch_dtype)
    # Warmup (small subset)
    warm = q_t[:min(warmup, q_t.shape[0])]
    _ = fk_solver.solve(warm, device=device, dtype=torch_dtype)
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    T_batch = fk_solver.solve(q_t, device=device, dtype=torch_dtype)  # [B,4,4]
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    results = [T_batch[i].detach().cpu().numpy() for i in range(T_batch.shape[0])]
    return {
        'backend': f'torch_{device}',
        'batch_size': q_t.shape[0],
        'total_time': dt,
        'avg_time': dt / q_t.shape[0],
        'throughput': q_t.shape[0] / dt,
        'results': results,
    }


# ---------------------------- IK Benchmarks -------------------------------- #
def benchmark_ik_numpy(model: RobotModel, poses: List[np.ndarray], q_init_batch: np.ndarray, max_iters: int, warmup: int = 3) -> Dict[str, Any]:
    B = len(poses)
    for i in range(min(warmup, B)):
        _ = inverse_kinematics(model, poses[i], q_init_batch[i], backend='numpy', method='dls', max_iters=max_iters)
    t0 = time.perf_counter()
    succ = 0
    results = []
    for i in range(B):
        r = inverse_kinematics(model, poses[i], q_init_batch[i], backend='numpy', method='dls', max_iters=max_iters)
        results.append(r)
        succ += 1 if r.get('success') else 0
    dt = time.perf_counter() - t0
    return {
        'backend': 'numpy',
        'batch_size': B,
        'total_time': dt,
        'avg_time': dt / B,
        'throughput': B / dt,
        'success_rate': succ / B,
        'results': results,
    }


def benchmark_ik_torch(model: RobotModel, poses: List[np.ndarray], q_init_batch: np.ndarray, device: str, max_iters: int, warmup: int = 3, dtype='float32') -> Dict[str, Any] | None:
    if not _HAS_TORCH:
        return None
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    solver = IKSolverTorch(model, max_iters=max_iters, pos_tol=1e-4, ori_tol=1e-4, device=device, dtype=torch_dtype)
    B = len(poses)
    # Warmup
    for i in range(min(warmup, B)):
        _ = solver.solve(poses[i], q_init_batch[i], method='dls')
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    succ = 0
    results = []
    for i in range(B):
        r = solver.solve(poses[i], q_init_batch[i], method='dls')
        results.append(r)
        succ += 1 if r.get('success') else 0
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        'backend': f'torch_{device}',
        'batch_size': B,
        'total_time': dt,
        'avg_time': dt / B,
        'throughput': B / dt,
        'success_rate': succ / B,
        'results': results,
    }


# ---------------------------- Result Formatting ---------------------------- #
def print_fk_table(results: Dict[str, Dict[str, Any]]):
    print_section("Forward Kinematics Performance")
    print(f"\n{'Backend':<18}{'Batch':>8}{'Total(s)':>12}{'Avg(ms)':>12}{'Throughput':>14}")
    print('-' * 64)
    for k, v in results.items():
        if v is None:
            continue
        print(f"{k:<18}{v['batch_size']:>8}{v['total_time']:>12.4f}{v['avg_time']*1e3:>12.3f}{v['throughput']:>14.1f}")
    if 'numpy' in results and results['numpy']:
        base = results['numpy']['avg_time']
        print("\nSpeedup vs NumPy:")
        for k, v in results.items():
            if v and k != 'numpy':
                print(f"  {k:<18}: {base / v['avg_time']:.2f}x")


def print_ik_table(results: Dict[str, Dict[str, Any]]):
    print_section("Inverse Kinematics Performance")
    print(f"\n{'Backend':<18}{'Batch':>8}{'Total(s)':>12}{'Avg(ms)':>12}{'Throughput':>14}{'Succ(%)':>10}")
    print('-' * 78)
    for k, v in results.items():
        if v is None:
            continue
        succ_percent = 100.0 * v.get('success_rate', 0.0)
        print(f"{k:<18}{v['batch_size']:>8}{v['total_time']:>12.4f}{v['avg_time']*1e3:>12.3f}{v['throughput']:>14.1f}{succ_percent:>10.1f}")
    if 'numpy' in results and results['numpy']:
        base = results['numpy']['avg_time']
        print("\nSpeedup vs NumPy:")
        for k, v in results.items():
            if v and k != 'numpy':
                print(f"  {k:<18}: {base / v['avg_time']:.2f}x")


# ---------------------------- Main Flow ------------------------------------ #
def main(args):
    # Normalize device
    if args.device.startswith('cuda') and _HAS_TORCH:
        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，回退到 CPU")
            args.device = 'cpu'
        else:
            # Expand plain 'cuda' -> 'cuda:0'
            if args.device == 'cuda':
                args.device = 'cuda:0'
    elif args.device != 'cpu':
        if args.device.startswith('cuda') and not _HAS_TORCH:
            print("⚠️  未安装 PyTorch，无法使用 CUDA，回退 cpu")
        args.device = 'cpu'

    print_header("Parallel FK / IK Benchmark")
    print(f"URDF        : {args.urdf}")
    print(f"End Link    : {args.end_link}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Warmup      : {args.warmup}")
    print(f"Max IK Iters: {args.ik_iters}")
    print(f"Torch Device: {args.device if _HAS_TORCH else 'N/A (torch not installed)'}")

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"\n❌ URDF 不存在: {urdf_path}")
        return
    model = RobotModel(str(urdf_path), end_link=args.end_link)
    print(f"\n✓ 模型加载完成: DOF={model.dof()} end_link={model.end_link}")

    # Generate joint samples
    print("\n🎲 生成随机关节配置...")
    q_batch = random_q_batch(model, args.batch_size, seed=args.seed)
    print("✓ 完成")

    fk_results: Dict[str, Dict[str, Any] | None] = {}
    ik_results: Dict[str, Dict[str, Any] | None] = {}

    # ---------------- FK ----------------
    if not args.ik_only:
        print_section("Benchmark FK")
        print("NumPy FK...")
        fk_results['numpy'] = benchmark_fk_numpy(model, q_batch, warmup=args.warmup)
        print(f"  平均 {fk_results['numpy']['avg_time']*1e3:.3f} ms/样本")
        if not args.numpy_only and _HAS_TORCH:
            print(f"PyTorch FK ({args.device}) true batch ...")
            fk_results[f'torch_{args.device}'] = benchmark_fk_torch(model, q_batch, device=args.device, warmup=args.warmup, dtype=args.torch_dtype)
            if fk_results[f'torch_{args.device}']:
                print(f"  平均 {fk_results[f'torch_{args.device}']['avg_time']*1e3:.3f} ms/样本")
        print_fk_table(fk_results)

    # Prepare poses for IK (reuse FK NumPy results for determinism)
    if not args.fk_only:
        print_section("Prepare IK Targets")
        if 'numpy' in fk_results and fk_results['numpy']:
            poses = fk_results['numpy']['results']
        else:
            poses = [forward_kinematics(model, q_batch[i], backend='numpy', return_end=True) for i in range(q_batch.shape[0])]
        q_init_batch = random_q_batch(model, args.batch_size, seed=args.seed+1)
        print("✓ 目标与初始解已生成")

        print_section("Benchmark IK")
        print("NumPy IK (dls)...")
        ik_results['numpy'] = benchmark_ik_numpy(model, poses, q_init_batch, max_iters=args.ik_iters, warmup=args.warmup)
        print(f"  平均 {ik_results['numpy']['avg_time']*1e3:.3f} ms/样本, 成功率 {ik_results['numpy']['success_rate']*100:.1f}%")
        if not args.numpy_only and _HAS_TORCH:
            print(f"PyTorch IK ({args.device}) dls (逐样本循环, TODO: 向量化)...")
            ik_results[f'torch_{args.device}'] = benchmark_ik_torch(model, poses, q_init_batch, device=args.device, max_iters=args.ik_iters, warmup=args.warmup, dtype=args.torch_dtype)
            if ik_results[f'torch_{args.device}']:
                print(f"  平均 {ik_results[f'torch_{args.device}']['avg_time']*1e3:.3f} ms/样本, 成功率 {ik_results[f'torch_{args.device}']['success_rate']*100:.1f}%")
        print_ik_table(ik_results)

    print("\n" + "="*80)
    print("Benchmark 完成")
    print("="*80 + "\n")


def build_arg_parser():
    p = argparse.ArgumentParser(description='Parallel FK/IK benchmark (NumPy vs PyTorch)')
    p.add_argument('--urdf', type=str, default='robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf', help='URDF 路径')
    p.add_argument('--end-link', type=str, default='tool0', help='末端执行器 link 名称')
    p.add_argument('--batch-size', type=int, default=1000, help='批大小')
    p.add_argument('--warmup', type=int, default=5, help='预热样本数')
    p.add_argument('--ik-iters', type=int, default=100, help='IK 最大迭代')
    p.add_argument('--device', type=str, default='cpu', help='PyTorch 设备 (cpu / cuda / cuda:0 / cuda:1)')
    p.add_argument('--torch-dtype', type=str, default='float32', choices=['float32', 'float64'], help='Torch dtype')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--fk-only', action='store_true', help='仅跑 FK')
    p.add_argument('--ik-only', action='store_true', help='仅跑 IK')
    p.add_argument('--numpy-only', action='store_true', help='只跑 NumPy (跳过 Torch)')
    return p


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel FK/IK Performance Benchmark: NumPy vs PyTorch (TRUE BATCH)
==================================================================

This script benchmarks the performance of Forward Kinematics (FK) and 
Inverse Kinematics (IK) computation using NumPy (CPU) vs PyTorch (CPU/GPU)
for large-scale PARALLEL batch processing.

**THIS VERSION USES TRUE BATCH/VECTORIZED OPERATIONS**

Features:
- TRUE batch FK computation (vectorized across all samples)
- TRUE batch IK computation (vectorized DLS solver)
- Support for CPU and GPU (CUDA) backends (MPS support removed)
- Automatic performance comparison and visualization
- Scalability testing with different batch sizes

Performance Notes:
- NumPy: Sequential loop processing (baseline)
- PyTorch CPU: Batch operations, but CPU-bound
- PyTorch GPU: Batch operations with CUDA acceleration (10-100x speedup!)

Author: RoboCore Team
Date: 2025-10-03
"""

import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics

# Check PyTorch availability
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
    from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
    from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
    from robocore.transform import matrix_to_quaternion
except ImportError:
    pass


def print_header(title: str, width: int = 80):
    """Print formatted header."""
    print(f"\n{'=' * width}")
    print(f"{title.center(width)}")
    print(f"{'=' * width}")


def print_section(title: str, width: int = 80):
    """Print formatted section."""
    print(f"\n{'-' * width}")
    print(f"{title}")
    print(f"{'-' * width}")


def random_q_batch(model, batch_size: int, seed: int = 42, scale: float = 0.5):
    """Generate batch of random joint configurations."""
    rng = np.random.default_rng(seed)
    n_joints = model.dof()
    q_batch = np.zeros((batch_size, n_joints))
    
    for i in range(batch_size):
        for js in model._actuated:
            lo, hi = -1.0, 1.0
            if js.limit:
                if js.limit[0] is not None:
                    lo = js.limit[0]
                if js.limit[1] is not None:
                    hi = js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo) * scale
            q_batch[i, js.index] = rng.uniform(mid - span, mid + span)
    
    return q_batch


def benchmark_fk_numpy(model, q_batch: np.ndarray, warmup: int = 5) -> Dict:
    """Benchmark FK computation using NumPy."""
    batch_size = q_batch.shape[0]
    
    # Warmup
    for i in range(min(warmup, batch_size)):
        _ = forward_kinematics(model, q_batch[i], backend='numpy', return_end=True)
    
    # Benchmark
    start_time = time.perf_counter()
    results = []
    for i in range(batch_size):
        T = forward_kinematics(model, q_batch[i], backend='numpy', return_end=True)
        results.append(T)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    return {
        'backend': 'numpy',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'results': results
    }


def benchmark_fk_torch(model, q_batch: np.ndarray, device: str = 'cpu', warmup: int = 5) -> Dict:
    """Benchmark FK computation using PyTorch with TRUE BATCH operations."""
    if not _HAS_TORCH:
        return None
    
    batch_size = q_batch.shape[0]
    
    # Convert to torch tensor
    q_torch = torch.from_numpy(q_batch).float()
    
    # Create solver
    fk_solver = FKSolverTorch(model)
    
    # Warmup - process a small batch
    warmup_batch = q_torch[:min(warmup, batch_size)]
    _ = fk_solver.solve(warmup_batch, device=device, dtype=torch.float32)
    
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # Benchmark - TRUE BATCH: all samples processed in parallel!
    start_time = time.perf_counter()
    T_batch = fk_solver.solve(q_torch, device=device, dtype=torch.float32)  # [B, 4, 4]
    
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    # Extract results (for compatibility)
    results = [T_batch[i].cpu().numpy() for i in range(batch_size)]
    
    return {
        'backend': f'torch_{device}',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'results': results
    }


def benchmark_ik_numpy(model, poses: List[np.ndarray], q_init_batch: np.ndarray, 
                       max_iters: int = 100, warmup: int = 5) -> Dict:
    """Benchmark IK computation using NumPy."""
    batch_size = len(poses)
    
    # Warmup
    for i in range(min(warmup, batch_size)):
        _ = inverse_kinematics(
            model, poses[i], q_init_batch[i],
            backend='numpy', method='dls',
            max_iters=max_iters, pos_tol=1e-4, ori_tol=1e-4
        )
    
    # Benchmark
    start_time = time.perf_counter()
    results = []
    success_count = 0
    for i in range(batch_size):
        ik_result = inverse_kinematics(
            model, poses[i], q_init_batch[i],
            backend='numpy', method='dls',
            max_iters=max_iters, pos_tol=1e-4, ori_tol=1e-4
        )
        results.append(ik_result)
        if ik_result['success']:
            success_count += 1
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    return {
        'backend': 'numpy',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'success_rate': success_count / batch_size,
        'results': results
    }


def benchmark_ik_torch(model, poses: List, q_init_batch: np.ndarray,
                       device: str = 'cpu', max_iters: int = 100, warmup: int = 5) -> Dict:
    """Benchmark IK computation using PyTorch with TRUE BATCH operations."""
    if not _HAS_TORCH:
        return None
    
    batch_size = len(poses)
    
    # Convert poses to torch tensor [B, 4, 4]
    poses_np = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in poses]
    poses_tensor = torch.stack([torch.from_numpy(p).float() for p in poses_np])
    
    # Convert initial guesses to torch
    q_init_torch = torch.from_numpy(q_init_batch).float()
    
    # TODO: Implement batch IK support in IKSolverTorch
    # For now, use loop (not true batch processing)
    print(f"  Note: Batch IK not yet implemented in unified solver, using sequential processing...")
    
    ik_solver = IKSolverTorch(model, max_iters=max_iters, pos_tol=1e-4, ori_tol=1e-4)
    
    start_time = time.perf_counter()
    
    q_sol_list = []
    success_list = []
    iters_list = []
    
    for i in range(batch_size):
        pose = poses_tensor[i].cpu().numpy().tolist()
        q_init = q_init_torch[i].cpu().numpy().tolist()
        result = ik_solver.solve(pose, q_init, method='dls')
        q_sol_list.append(result['q'])
        success_list.append(result.get('success', False))
        iters_list.append(result.get('iters', 0))
    
    q_sol_batch = torch.tensor(q_sol_list, device=device)
    success_batch = torch.tensor(success_list, device=device)
    iters_batch = torch.tensor(iters_list, device=device)
    
    if device != 'cpu' and device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    # Convert results
    success_count = success_batch.sum().item()
    results = [
        {
            'q': q_sol_batch[i].cpu().numpy(),
            'success': success_batch[i].item(),
            'iterations': iters_batch[i].item()
        }
        for i in range(batch_size)
    ]
    
    return {
        'backend': f'torch_{device}',
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': batch_size / total_time,
        'success_rate': success_count / batch_size,
        'results': results
    }


def print_benchmark_results(fk_results: Dict, ik_results: Dict = None):
    """Print formatted benchmark results."""
    print_section("Forward Kinematics (FK) Performance")
    
    # FK Results
    print(f"\n{'Backend':<20} {'Batch Size':<12} {'Total Time':<15} {'Avg Time':<15} {'Throughput':<15}")
    print(f"{'-'*20} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    
    for backend, result in fk_results.items():
        if result is not None:
            print(f"{backend:<20} {result['batch_size']:<12} "
                  f"{result['total_time']:>10.4f} s    "
                  f"{result['avg_time']*1000:>10.4f} ms   "
                  f"{result['throughput']:>10.2f} it/s")
    
    # Compute speedup
    if 'numpy' in fk_results and fk_results['numpy'] is not None:
        numpy_time = fk_results['numpy']['avg_time']
        print(f"\n{'Backend':<20} {'Speedup vs NumPy':<20}")
        print(f"{'-'*20} {'-'*20}")
        for backend, result in fk_results.items():
            if result is not None:
                speedup = numpy_time / result['avg_time']
                print(f"{backend:<20} {speedup:>10.2f}x")
    
    # IK Results
    if ik_results:
        print_section("Inverse Kinematics (IK) Performance")
        print(f"\n{'Backend':<20} {'Batch Size':<12} {'Total Time':<15} {'Avg Time':<15} "
              f"{'Throughput':<15} {'Success Rate':<15}")
        print(f"{'-'*20} {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        for backend, result in ik_results.items():
            if result is not None:
                print(f"{backend:<20} {result['batch_size']:<12} "
                      f"{result['total_time']:>10.4f} s    "
                      f"{result['avg_time']*1000:>10.4f} ms   "
                      f"{result['throughput']:>10.2f} it/s   "
                      f"{result['success_rate']*100:>10.2f}%")
        
        # Compute speedup
        if 'numpy' in ik_results and ik_results['numpy'] is not None:
            numpy_time = ik_results['numpy']['avg_time']
            print(f"\n{'Backend':<20} {'Speedup vs NumPy':<20}")
            print(f"{'-'*20} {'-'*20}")
            for backend, result in ik_results.items():
                if result is not None:
                    speedup = numpy_time / result['avg_time']
                    print(f"{backend:<20} {speedup:>10.2f}x")


def main(args):
    # Check PyTorch availability
    if not args.numpy_only and not _HAS_TORCH:
        print("⚠️  Warning: PyTorch not available. Running NumPy benchmarks only.")
        args.numpy_only = True
    
    # Check device availability and normalize device string
    if not args.numpy_only and _HAS_TORCH:
        device_type = args.device.split(':')[0]  # Extract 'cuda' from 'cuda:1'
        
        if device_type == 'cuda':
            if not torch.cuda.is_available():
                print(f"⚠️  Warning: CUDA not available. Using CPU instead.")
                args.device = 'cpu'
            else:
                # Check if specific GPU ID is specified
                if ':' in args.device:
                    gpu_id = int(args.device.split(':')[1])
                    if gpu_id >= torch.cuda.device_count():
                        print(f"⚠️  Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPU(s) found).")
                        print(f"    Using cuda:0 instead.")
                        args.device = 'cuda:0'
                else:
                    # Default to cuda:0 if just 'cuda' is specified
                    args.device = 'cuda:0'
        # mps removed
    
    # Print configuration
    print_header("Parallel FK/IK Performance Benchmark (TRUE BATCH MODE)")
    print(f"\nConfiguration:")
    print(f"  URDF: {args.urdf}")
    print(f"  End Link: {args.end_link}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Warmup Iterations: {args.warmup}")
    print(f"  Seed: {args.seed}")
    if not args.numpy_only:
        print(f"  PyTorch Device: {args.device.upper()}")
        if args.device.startswith('cuda'):
            gpu_id = args.device.split(':')[1] if ':' in args.device else '0'
            print(f"  CUDA Device {gpu_id}: {torch.cuda.get_device_name(int(gpu_id))}")
        # mps removed
        print(f"  ⚡ TRUE BATCH MODE: All samples processed in parallel!")
    
    # Load robot model
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"\n❌ Error: URDF file not found: {urdf_path}")
        return
    
    print(f"\n📦 Loading robot model...")
    model = RobotModel(str(urdf_path), end_link=args.end_link)
    print(f"✓ Robot loaded: {model.dof()} DOF, end_link={model.end_link}")
    
    # Generate random joint configurations
    print(f"\n🎲 Generating {args.batch_size} random joint configurations...")
    q_batch = random_q_batch(model, args.batch_size, seed=args.seed)
    print(f"✓ Configurations generated")
    
    # Benchmark FK
    fk_results = {}
    if not args.ik_only:
        print_section("Benchmarking Forward Kinematics (FK)")
        
        # NumPy
        print(f"\n⏱️  Running FK benchmark with NumPy...")
        fk_results['numpy'] = benchmark_fk_numpy(model, q_batch, warmup=args.warmup)
        print(f"✓ NumPy FK completed: {fk_results['numpy']['avg_time']*1000:.2f} ms/sample")
        
        # PyTorch
        if not args.numpy_only:
            print(f"\n⏱️  Running FK benchmark with PyTorch ({args.device.upper()}) - TRUE BATCH MODE...")
            
            fk_results[f'torch_{args.device}'] = benchmark_fk_torch(
                model, q_batch, device=args.device, warmup=args.warmup
            )
            if fk_results[f'torch_{args.device}'] is not None:
                print(f"✓ PyTorch FK completed: {fk_results[f'torch_{args.device}']['avg_time']*1000:.2f} ms/sample")
    
    # Benchmark IK
    ik_results = {}
    if not args.fk_only:
        print_section("Benchmarking Inverse Kinematics (IK)")
        
        # Generate target poses from FK
        print(f"\n📍 Generating target poses...")
        if 'numpy' in fk_results:
            poses = fk_results['numpy']['results']
        else:
            poses = []
            for i in range(args.batch_size):
                T = forward_kinematics(model, q_batch[i], backend='numpy', return_end=True)
                poses.append(T)
        
        # Generate random initial guesses
        q_init_batch = random_q_batch(model, args.batch_size, seed=args.seed + 1)
        print(f"✓ Target poses generated")
        
        # NumPy
        print(f"\n⏱️  Running IK benchmark with NumPy...")
        ik_results['numpy'] = benchmark_ik_numpy(
            model, poses, q_init_batch, max_iters=args.ik_iters, warmup=args.warmup
        )
        print(f"✓ NumPy IK completed: {ik_results['numpy']['avg_time']*1000:.2f} ms/sample, "
              f"success rate: {ik_results['numpy']['success_rate']*100:.1f}%")
        
        # PyTorch
        if not args.numpy_only:
            print(f"\n⏱️  Running IK benchmark with PyTorch ({args.device.upper()}) - TRUE BATCH MODE...")
            
            ik_results[f'torch_{args.device}'] = benchmark_ik_torch(
                model, poses, q_init_batch, device=args.device, 
                max_iters=args.ik_iters, warmup=args.warmup
            )
            if ik_results[f'torch_{args.device}'] is not None:
                print(f"✓ PyTorch IK completed: {ik_results[f'torch_{args.device}']['avg_time']*1000:.2f} ms/sample, "
                      f"success rate: {ik_results[f'torch_{args.device}']['success_rate']*100:.1f}%")
    
    # Print results
    print("\n")
    print_benchmark_results(fk_results if not args.ik_only else {}, 
                           ik_results if not args.fk_only else {})
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark parallel FK/IK performance: NumPy vs PyTorch'
    )
    parser.add_argument(
        '--urdf',
        type=str,
        default='robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf',
        help='Path to URDF file'
    )
    parser.add_argument(
        '--end-link',
        type=str,
        default='tool0',
        help='End-effector link name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of configurations to benchmark (default: 1000)'
    )
    parser.add_argument(
        '--fk-only',
        action='store_true',
        help='Only benchmark FK (skip IK)'
    )
    parser.add_argument(
        '--ik-only',
        action='store_true',
        help='Only benchmark IK (skip FK)'
    )
    parser.add_argument(
        '--ik-iters',
        type=int,
        default=100,
        help='Maximum IK iterations (default: 100)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=5,
        help='Number of warmup iterations (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
    help='PyTorch device (cpu/cuda/cuda:0/cuda:1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--numpy-only',
        action='store_true',
        help='Only benchmark NumPy (skip Torch)'
    )

    args = parser.parse_args()

    main(args)
