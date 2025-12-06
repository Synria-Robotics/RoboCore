"""Jacobian validation and comparison.

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

from __future__ import annotations
import argparse
import time
import numpy as np

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def print_jacobian_comparison(Ja, Jn, Jg=None):
    """Print Jacobian matrices and comparison statistics."""
    Ja_np = to_numpy(Ja)
    Jn_np = to_numpy(Jn)
    Jg_np = to_numpy(Jg) if Jg is not None else None
    
    beauty_print(f"Jacobian shape: {Ja_np.shape}")
    cond_num = float(np.linalg.cond(Ja_np))
    beauty_print(f"Condition number: {cond_num:.2e}")
    
    beauty_print(f"Jacobian Matrix (Analytic):")
    print(beauty_print_array(Ja_np, precision=6))
    
    beauty_print(f"Jacobian Matrix (Numeric):")
    print(beauty_print_array(Jn_np, precision=6))
    
    if Jg_np is not None:
        beauty_print(f"Jacobian Matrix (Autograd):")
        print(beauty_print_array(Jg_np, precision=6))
    
    # Compare
    diff_an = Ja_np - Jn_np
    beauty_print("Analytic vs Numeric:")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_an)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_an, 'fro'):.3e}")
    
    if Jg_np is not None:
        diff_ag = Ja_np - Jg_np
        diff_ng = Jn_np - Jg_np
        
        beauty_print("Analytic vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ag)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ag, 'fro'):.3e}")
        
        beauty_print("Numeric vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ng)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ng, 'fro'):.3e}")


def benchmark_jacobian(model, q, methods, n_runs, device=None):
    """Benchmark Jacobian computation for given methods."""
    times = {}
    for method in methods:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            if device is not None:
                _ = jacobian(model, q, method=method, device=device)
            else:
                _ = jacobian(model, q, method=method)
        times[method] = (time.perf_counter() - t0) / n_runs * 1000
    
    for method, t in times.items():
        beauty_print(f"{method.capitalize():8s}   {t:.4f} ms")
    
    return times


def compute_condition_statistics(model, rng, samples, method='analytic', device=None):
    """Compute condition number statistics across random configurations."""
    condition_numbers = []
    for i in range(samples):
        q_rand = model.random_q(rng)
        
        # Convert to appropriate format
        if device is not None:
            import torch
            if isinstance(q_rand, np.ndarray):
                q_rand = torch.from_numpy(q_rand).to(dtype=torch.float64, device=device)
            else:
                q_rand = torch.tensor(q_rand, dtype=torch.float64, device=device)
        
        # Compute Jacobian
        if device is not None:
            J_rand = jacobian(model, q_rand, method=method, device=device)
        else:
            J_rand = jacobian(model, q_rand, method=method)
        
        J_rand_np = to_numpy(J_rand)
        cond_num = float(np.linalg.cond(J_rand_np))
        condition_numbers.append(cond_num)
    
    beauty_print(f"Condition number statistics:")
    beauty_print(f"  Mean:   {np.mean(condition_numbers):.2e}")
    beauty_print(f"  Median: {np.median(condition_numbers):.2e}")
    beauty_print(f"  Max:    {np.max(condition_numbers):.2e}")
    beauty_print(f"  Min:    {np.min(condition_numbers):.2e}")


def main(args):
    model = RobotModel(args.model_path, end_link=args.end_link)

    beauty_print(f"Jacobian Validation: {model.name} ({model.num_dof} DOF)", type="module")

    rng = np.random.default_rng(args.seed)

    # Setup backend
    if args.backend == 'numpy':
        rc.set_backend('numpy')
        device = None
        q = np.zeros(model.num_chain_dof)
        methods = ['analytic', 'numeric']
        title = "[1] Analytic vs Numeric Jacobian (NumPy)"
        n_runs = 100
    else:
        import torch
        device = torch.device(args.device)
        rc.set_backend('torch', device=str(device))
        q = torch.zeros(model.num_chain_dof, dtype=torch.float64, device=device)
        methods = ['analytic', 'numeric', 'autograd']
        title = "[1] Analytic vs Numeric vs Autograd Jacobian (PyTorch)"
        n_runs = 100

    beauty_print(title, type="module", centered=False)
    
    beauty_print(f"Joint configuration (rad):")
    q_display = q.cpu().numpy() if hasattr(q, 'cpu') else q
    print(f"  q = {beauty_print_array(q_display)}")

    # Compute Jacobians
    Ja = jacobian(model, q, method='analytic', device=device)
    Jn = jacobian(model, q, method='numeric', device=device)
    Jg = jacobian(model, q, method='autograd', device=device) if 'autograd' in methods else None

    # Print comparison
    print_jacobian_comparison(Ja, Jn, Jg)

    # Timing
    beauty_print("[2] Performance comparison", type="module", centered=False)
    benchmark_jacobian(model, q, methods, n_runs, device=device)

    # Condition number statistics
    beauty_print(f"[3] Condition number across {args.samples} random configurations", type="module", centered=False)
    compute_condition_statistics(model, rng, args.samples, method='analytic', device=device)

    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy', help='Backend to test')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')    
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)

    """_results_
    ================================================================================
            Jacobian Validation: Alicia_D_v5_6_gripper_100mm.urdf (10 DOF)         
    ================================================================================
    [RoboCore:INFO] Backend set to numpy on device cpu with dtype <class 'numpy.float64'>
    [RoboCore:MODULE] [1] Analytic vs Numeric Jacobian (NumPy)
    [RoboCore:INFO] Joint configuration (rad):
    q = [+0.00000, +0.00000, +0.00000, +0.00000, +0.00000, +0.00000]
    [RoboCore:INFO] Jacobian shape: (6, 6)
    [RoboCore:INFO] Condition number: 6.76e+02
    [RoboCore:INFO] Jacobian Matrix (Analytic):
    [
    [+0.000102  -0.027725  +0.195946  -0.000347  +0.037936  +0.000000]
    [+0.220999  +0.000044  -0.000312  +0.003851  -0.000060  +0.000000]
    [-0.000000  +0.220999  +0.198407  -0.000354  +0.043381  -0.000000]
    [+0.000000  -0.001593  -0.001593  -0.707106  -0.001593  -0.707106]
    [+0.000000  -0.999999  -0.999999  +0.001126  -0.999999  +0.001126]
    [+1.000000  -0.000000  -0.000000  +0.707107  -0.000000  +0.707107]
    ]
    [RoboCore:INFO] Jacobian Matrix (Numeric):
    [
    [+0.000102  -0.027725  +0.195946  -0.000347  +0.037936  +0.000000]
    [+0.220999  +0.000044  -0.000312  +0.003851  -0.000060  +0.000000]
    [+0.000000  +0.220999  +0.198407  -0.000354  +0.043381  +0.000000]
    [+0.000000  -0.001593  -0.001593  -0.707106  -0.001593  -0.707106]
    [+0.000000  -0.999999  -0.999999  +0.001126  -0.999999  +0.001126]
    [+1.000000  +0.000000  -0.000000  +0.707107  -0.000000  +0.707107]
    ]
    [RoboCore:INFO] Analytic vs Numeric:
    [RoboCore:INFO]   Max difference:        4.147e-08
    [RoboCore:INFO]   Frobenius norm:        8.305e-08
    [RoboCore:MODULE] [2] Performance comparison
    [RoboCore:INFO] Analytic   0.1212 ms
    [RoboCore:INFO] Numeric    2.0630 ms
    [RoboCore:MODULE] [3] Condition number across 10 random configurations
    [RoboCore:INFO] Condition number statistics:
    [RoboCore:INFO]   Mean:   1.02e+02
    [RoboCore:INFO]   Median: 5.21e+01
    [RoboCore:INFO]   Max:    3.73e+02
    [RoboCore:INFO]   Min:    2.11e+01
    """
