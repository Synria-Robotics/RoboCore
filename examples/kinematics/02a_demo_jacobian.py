"""Jacobian validation and comparison.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param q: Joint configuration
    :param device: Device for torch backend
    :return: Dictionary with Jacobian matrices and computation time
    """
    if backend == 'torch':
        import torch
        if device is None:
            device = torch.device('cpu')
        rc.set_backend('torch', device=str(device))
    else:
        rc.set_backend(backend)

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
    results_cpp = compute_jacobian_results(robot_model, 'cpp', q)

    # Convert to numpy for comparison
    Ja_np_np = to_numpy(results_np['Ja'])
    Jn_np_np = to_numpy(results_np['Jn'])
    Ja_torch_np = to_numpy(results_torch['Ja'])
    Jn_torch_np = to_numpy(results_torch['Jn'])
    Ja_cpp_np = to_numpy(results_cpp['Ja'])
    Jn_cpp_np = to_numpy(results_cpp['Jn'])

    beauty_print("[1] Jacobian Comparison (NumPy vs Torch vs C++)", type="module", centered=False)
    beauty_print(f"Jacobian shape: {Ja_np_np.shape}")
    beauty_print(f"Condition number (NumPy): {np.linalg.cond(Ja_np_np):.2e}")
    beauty_print(f"Condition number (Torch): {np.linalg.cond(Ja_torch_np):.2e}")
    beauty_print(f"Condition number (C++):   {np.linalg.cond(Ja_cpp_np):.2e}")

    beauty_print(f"Analytic Jacobian (NumPy):")
    print(beauty_print_array(Ja_np_np, precision=6))
    beauty_print(f"Analytic Jacobian (Torch):")
    print(beauty_print_array(Ja_torch_np, precision=6))
    beauty_print(f"Analytic Jacobian (C++):")
    print(beauty_print_array(Ja_cpp_np, precision=6))

    diff_analytic_nt = Ja_np_np - Ja_torch_np
    diff_analytic_nc = Ja_np_np - Ja_cpp_np
    diff_analytic_tc = Ja_torch_np - Ja_cpp_np
    diff_numeric_nt = Jn_np_np - Jn_torch_np
    diff_numeric_nc = Jn_np_np - Jn_cpp_np
    diff_numeric_tc = Jn_torch_np - Jn_cpp_np
    beauty_print("NumPy vs Torch (Analytic):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_analytic_nt)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_analytic_nt, 'fro'):.3e}")
    beauty_print("NumPy vs C++ (Analytic):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_analytic_nc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_analytic_nc, 'fro'):.3e}")
    beauty_print("Torch vs C++ (Analytic):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_analytic_tc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_analytic_tc, 'fro'):.3e}")
    beauty_print("NumPy vs Torch (Numeric):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_numeric_nt)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_numeric_nt, 'fro'):.3e}")
    beauty_print("NumPy vs C++ (Numeric):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_numeric_nc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_numeric_nc, 'fro'):.3e}")
    beauty_print("Torch vs C++ (Numeric):")
    beauty_print(f"  Max difference:        {np.max(np.abs(diff_numeric_tc)):.3e}")
    beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_numeric_tc, 'fro'):.3e}")

    beauty_print("[2] Performance Comparison", type="module", centered=False)
    beauty_print("Analytic:")
    print(f"  NumPy:  {results_np['time_analytic']:.4f} ms")
    print(f"  Torch:  {results_torch['time_analytic']:.4f} ms")
    print(f"  C++:    {results_cpp['time_analytic']:.4f} ms")
    tnp_a = max(results_np['time_analytic'], 1e-15)
    print(f"  torch/np: {results_torch['time_analytic'] / tnp_a:.2f}x   cpp/np: {results_cpp['time_analytic'] / tnp_a:.2f}x")
    beauty_print("Numeric:")
    print(f"  NumPy:  {results_np['time_numeric']:.4f} ms")
    print(f"  Torch:  {results_torch['time_numeric']:.4f} ms")
    print(f"  C++:    {results_cpp['time_numeric']:.4f} ms")
    tnp_n = max(results_np['time_numeric'], 1e-15)
    print(f"  torch/np: {results_torch['time_numeric'] / tnp_n:.2f}x   cpp/np: {results_cpp['time_numeric'] / tnp_n:.2f}x")

    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--backend', choices=['numpy', 'torch', 'cpp'], default='numpy',
                        help='Legacy option (ignored — NumPy, Torch, and C++ are all tested)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
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
    [RoboCore:INFO] Backend set to cpp on device cpu with dtype <class 'numpy.float64'>
    [RoboCore:MODULE] [1] Jacobian Comparison (NumPy vs Torch vs C++)
    [RoboCore:INFO] Jacobian shape: (6, 8)
    [RoboCore:INFO] Condition number (NumPy): 6.76e+02
    [RoboCore:INFO] Condition number (Torch): 6.76e+02
    [RoboCore:INFO] Condition number (C++):   6.76e+02
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
    [RoboCore:INFO] Analytic Jacobian (C++):
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
    [RoboCore:INFO] NumPy vs C++ (Analytic):
    [RoboCore:INFO]   Max difference:        3.123e-17
    [RoboCore:INFO]   Frobenius norm:        6.624e-17
    [RoboCore:INFO] Torch vs C++ (Analytic):
    [RoboCore:INFO]   Max difference:        2.776e-17
    [RoboCore:INFO]   Frobenius norm:        6.360e-17
    [RoboCore:INFO] NumPy vs Torch (Numeric):
    [RoboCore:INFO]   Max difference:        2.209e-11
    [RoboCore:INFO]   Frobenius norm:        4.218e-11
    [RoboCore:INFO] NumPy vs C++ (Numeric):
    [RoboCore:INFO]   Max difference:        8.327e-13
    [RoboCore:INFO]   Frobenius norm:        8.777e-13
    [RoboCore:INFO] Torch vs C++ (Numeric):
    [RoboCore:INFO]   Max difference:        2.209e-11
    [RoboCore:INFO]   Frobenius norm:        4.206e-11
    [RoboCore:MODULE] [2] Performance Comparison
    [RoboCore:INFO] Analytic:
    NumPy:  0.2326 ms
    Torch:  1.1135 ms
    C++:    0.2174 ms
    torch/np: 4.79x   cpp/np: 0.93x
    [RoboCore:INFO] Numeric:
    NumPy:  2.0451 ms
    Torch:  8.7756 ms
    C++:    0.8293 ms
    torch/np: 4.29x   cpp/np: 0.41x
    """
