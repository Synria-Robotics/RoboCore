"""Jacobian validation and comparison.

Compares analytic, numeric (finite-difference), and autograd (PyTorch) Jacobian
implementations to verify correctness and measure performance.

Usage:
    python examples/demo_jacobian.py
    python examples/demo_jacobian.py --backend torch --device cpu
"""
from __future__ import annotations
import os
import argparse
import time
import numpy as np
from pathlib import Path
from robocore import RobotModel, jacobian
from robocore.utils.beauty_logger import beauty_print


def main():
    parser = argparse.ArgumentParser(description="Jacobian validation")
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy',
                        help='Backend to test')
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load model
    base = Path(__file__).resolve().parents[1]
    urdf = os.path.join(base, "../robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf")
    model = RobotModel(str(urdf), end_link='tool0')
    
    beauty_print(f"Jacobian Validation: {model.name} ({model.dof()} DOF)", type="module")
    beauty_print(f"Backend: {args.backend}", type="info")
    
    rng = np.random.default_rng(args.seed)
    
    if args.backend == 'numpy':
        # NumPy backend comparison
        beauty_print("\n[1] Analytic vs Numeric Jacobian (NumPy)", type="module")
        
        q = np.zeros(model.dof())
        
        # Warmup
        jacobian(model, q, backend='numpy', method='analytic')
        jacobian(model, q, backend='numpy', method='numeric')
        
        # Timing
        n_runs = 100
        t0 = time.perf_counter()
        for _ in range(n_runs):
            Ja = jacobian(model, q, backend='numpy', method='analytic')
        time_analytic = (time.perf_counter() - t0) / n_runs * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            Jn = jacobian(model, q, backend='numpy', method='numeric')
        time_numeric = (time.perf_counter() - t0) / n_runs * 1000
        
        # Compare
        Ja = jacobian(model, q, backend='numpy', method='analytic')
        Jn = jacobian(model, q, backend='numpy', method='numeric')
        diff = Ja - Jn
        
        beauty_print(f"Analytic time:  {time_analytic:.4f} ms")
        beauty_print(f"Numeric time:   {time_numeric:.4f} ms")
        beauty_print(f"Speedup:        {time_numeric/time_analytic:.2f}x", type="success")
        beauty_print(f"\nDifference (analytic - numeric):")
        beauty_print(f"  Linear block max:      {np.max(np.abs(diff[:3, :])):.3e}")
        beauty_print(f"  Angular block max:     {np.max(np.abs(diff[3:6, :])):.3e}")
        beauty_print(f"  Overall max:           {np.max(np.abs(diff)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff, 'fro'):.3e}")
        
        # Test multiple configurations
        beauty_print(f"\n[2] Accuracy across {args.samples} random configurations", type="module")
        
        max_diffs = []
        for i in range(args.samples):
            q_rand = []
            for js in model._actuated:  # type: ignore[attr-defined]
                lo, hi = -1.0, 1.0
                if js.limit:
                    if js.limit[0] is not None: lo = js.limit[0]
                    if js.limit[1] is not None: hi = js.limit[1]
                q_rand.append(float(rng.uniform(lo * 0.7, hi * 0.7)))
            
            Ja = jacobian(model, q_rand, backend='numpy', method='analytic')
            Jn = jacobian(model, q_rand, backend='numpy', method='numeric')
            max_diff = np.max(np.abs(Ja - Jn))
            max_diffs.append(max_diff)
        
        beauty_print(f"Max difference statistics:")
        beauty_print(f"  Mean:   {np.mean(max_diffs):.3e}")
        beauty_print(f"  Median: {np.median(max_diffs):.3e}")
        beauty_print(f"  Max:    {np.max(max_diffs):.3e}")
        beauty_print(f"  Min:    {np.min(max_diffs):.3e}")
        
    else:  # torch
        try:
            import torch
        except ImportError:
            beauty_print("PyTorch not available", type="error")
            return
        
        device = torch.device(args.device)
        beauty_print(f"PyTorch device: {device}", type="info")
        beauty_print("\n[1] Analytic vs Numeric vs Autograd Jacobian (PyTorch)", type="module")
        
        q = torch.zeros(model.dof(), dtype=torch.float64, device=device)
        
        # Compute all three using unified interface
        Ja = jacobian(model, q, backend='torch', method='analytic', device=device)
        Jn = jacobian(model, q, backend='torch', method='numeric', device=device)
        Jg = jacobian(model, q, backend='torch', method='autograd', device=device)
        
        # Compare
        diff_an = (Ja - Jn).cpu().numpy()
        diff_ag = (Ja - Jg).cpu().numpy()
        diff_ng = (Jn - Jg).cpu().numpy()
        
        beauty_print("Analytic vs Numeric:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_an)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_an, 'fro'):.3e}")
        
        beauty_print("\nAnalytic vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ag)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ag, 'fro'):.3e}")
        
        beauty_print("\nNumeric vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ng)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ng, 'fro'):.3e}")
        
        # Timing
        beauty_print("\n[2] Performance comparison", type="module")
        n_runs = 50
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, backend='torch', method='analytic', device=device)
        time_analytic = (time.perf_counter() - t0) / n_runs * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, backend='torch', method='numeric', device=device)
        time_numeric = (time.perf_counter() - t0) / n_runs * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, backend='torch', method='autograd', device=device)
        time_autograd = (time.perf_counter() - t0) / n_runs * 1000
        
        beauty_print(f"Analytic:   {time_analytic:.4f} ms")
        beauty_print(f"Numeric:    {time_numeric:.4f} ms")
        beauty_print(f"Autograd:   {time_autograd:.4f} ms")
    
    beauty_print("\n✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    main()
