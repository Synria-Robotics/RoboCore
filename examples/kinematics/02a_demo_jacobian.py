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


def compute_condition_statistics(model, seed, samples, method='analytic', device=None):
    """Compute condition number statistics across random configurations."""
    condition_numbers = []
    for i in range(samples):
        q_rand = model.random_q(seed)
        
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


def compute_jacobian_results(robot_model, backend, q, device=None):
    """Compute Jacobian results for given backend.
    
    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy' or 'torch')
    :param q: Joint configuration
    :param device: Device for torch backend
    :return: Dictionary with Jacobian matrices and computation time
    """
    rc.set_backend(backend)
    if backend == 'torch' and device is None:
        import torch
        device = torch.device('cpu')
        rc.set_backend('torch', device=str(device))

    start_time = time.perf_counter()
    Ja = jacobian(robot_model, q, method='analytic', device=device)
    time_analytic = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    Jn = jacobian(robot_model, q, method='numeric', device=device)
    time_numeric = (time.perf_counter() - start_time) * 1000

    return {
        'Ja': Ja,
        'Jn': Jn,
        'time_analytic': time_analytic,
        'time_numeric': time_numeric
    }


def main(args):
    robot_model = RobotModel(args.model_path, base_link=args.base_link, end_link=args.end_link)
    beauty_print(f"Jacobian Validation: {robot_model.name} ({robot_model.num_dof} DOF)", type="module")

    q = np.zeros(robot_model.num_dof)

    beauty_print(f"Joint configuration (rad):")
    print(f"  q = {beauty_print_array(q)}")

    # Compute with both backends
    results_np = compute_jacobian_results(robot_model, 'numpy', q)
    import torch
    device = torch.device(args.device)
    q_torch = torch.zeros(robot_model.num_dof, dtype=torch.float64, device=device)
    results_torch = compute_jacobian_results(robot_model, 'torch', q_torch, device)

    # Convert to numpy for comparison
    Ja_np_np = to_numpy(results_np['Ja'])
    Jn_np_np = to_numpy(results_np['Jn'])
    Ja_torch_np = to_numpy(results_torch['Ja'])
    Jn_torch_np = to_numpy(results_torch['Jn'])

    beauty_print("[1] Jacobian Comparison (NumPy vs Torch)", type="module", centered=False)
    beauty_print(f"Jacobian shape: {Ja_np_np.shape}")
    beauty_print(f"Condition number (NumPy): {np.linalg.cond(Ja_np_np):.2e}")
    beauty_print(f"Condition number (Torch): {np.linalg.cond(Ja_torch_np):.2e}")

    beauty_print(f"Analytic Jacobian (NumPy):")
    print(beauty_print_array(Ja_np_np, precision=6))
    beauty_print(f"Analytic Jacobian (Torch):")
    print(beauty_print_array(Ja_torch_np, precision=6))

    diff_analytic = Ja_np_np - Ja_torch_np
    diff_numeric = Jn_np_np - Jn_torch_np
    beauty_print("NumPy vs Torch (Analytic):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_analytic)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_analytic, 'fro'):.3e}")
    beauty_print("NumPy vs Torch (Numeric):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_numeric)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_numeric, 'fro'):.3e}")

    beauty_print("[2] Performance Comparison", type="module", centered=False)
    beauty_print("Analytic:")
    print(f"  NumPy:  {results_np['time_analytic']:.4f} ms")
    print(f"  Torch:  {results_torch['time_analytic']:.4f} ms")
    print(f"  Ratio:  {results_torch['time_analytic'] / results_np['time_analytic']:.2f}x")
    beauty_print("Numeric:")
    print(f"  NumPy:  {results_np['time_numeric']:.4f} ms")
    print(f"  Torch:  {results_torch['time_numeric']:.4f} ms")
    print(f"  Ratio:  {results_torch['time_numeric'] / results_np['time_numeric']:.2f}x")

    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy', help='Backend to test (ignored - both are tested)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')    
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)

    """_results_
    ================================================================================
             Jacobian Validation: Alicia_D_v5_6_gripper_100mm.urdf (8 DOF)          
    ================================================================================
    [RoboCore:INFO] Joint configuration (rad):
      q = [+0.00000, +0.00000, +0.00000, +0.00000, +0.00000, +0.00000, +0.00000, +0.00000]
    [RoboCore:INFO] Backend set to numpy on device cpu with dtype <class 'numpy.float64'>
    [RoboCore:INFO] Backend set to torch on device cpu with dtype torch.float64
    [RoboCore:MODULE] [1] Jacobian Comparison (NumPy vs Torch)
    [RoboCore:INFO] Jacobian shape: (6, 8)
    [RoboCore:INFO] Condition number (NumPy): 6.76e+02
    [RoboCore:INFO] Condition number (Torch): 6.76e+02
    [RoboCore:INFO] Analytic Jacobian (NumPy):
    [
      [-0.000348  -0.027724  +0.195946  -0.000353  +0.037936  +0.000000  +0.000000  +0.000000]
      [+0.220900  +0.000000  -0.000001  +0.003850  -0.000001  +0.000000  +0.000000  +0.000000]
      [+0.000000  +0.220900  +0.198407  -0.000353  +0.043381  -0.000000  +0.000000  +0.000000]
      [+0.000000  -0.000004  -0.000004  -0.707108  -0.000014  -0.707108  +0.000000  +0.000000]
      [+0.000000  -1.000000  -1.000000  +0.000006  -1.000000  +0.000014  +0.000000  +0.000000]
      [+1.000000  +0.000000  +0.000000  +0.707105  -0.000000  +0.707105  +0.000000  +0.000000]
    ]
    [RoboCore:INFO] Analytic Jacobian (Torch):
    [
      [-0.000348  -0.027724  +0.195946  -0.000353  +0.037936  +0.000000  +0.000000  +0.000000]
      [+0.220900  +0.000000  -0.000001  +0.003850  -0.000001  +0.000000  +0.000000  +0.000000]
      [+0.000000  +0.220900  +0.198407  -0.000353  +0.043381  -0.000000  +0.000000  +0.000000]
      [+0.000000  -0.000004  -0.000004  -0.707108  -0.000014  -0.707108  +0.000000  +0.000000]
      [+0.000000  -1.000000  -1.000000  +0.000006  -1.000000  +0.000014  +0.000000  +0.000000]
      [+1.000000  +0.000000  +0.000000  +0.707105  -0.000000  +0.707105  +0.000000  +0.000000]
    ]
    [RoboCore:INFO] NumPy vs Torch (Analytic):
    [RoboCore:INFO]   Max difference:        4.337e-18
    [RoboCore:INFO]   Frobenius norm:        5.554e-18
    [RoboCore:INFO] NumPy vs Torch (Numeric):
    [RoboCore:INFO]   Max difference:        2.209e-11
    [RoboCore:INFO]   Frobenius norm:        4.218e-11
    [RoboCore:MODULE] [2] Performance Comparison
    [RoboCore:INFO] Analytic:
      NumPy:  0.6775 ms
      Torch:  68.9991 ms
      Ratio:  101.84x
    [RoboCore:INFO] Numeric:
      NumPy:  10.2467 ms
      Torch:  27.3747 ms
      Ratio:  2.67x
    """
