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
import os
import argparse
import time
import numpy as np
from pathlib import Path

from robocore.modeling import RobotModel
from robocore.kinematics.jacobian import jacobian
from robocore.utils.beauty_logger import beauty_print
from robocore.utils.path import get_robocore_path
import robocore


def main(args):
    # Load model
    model_path = args.model_path
    model = RobotModel(model_path, end_link=args.end_link)

    beauty_print(f"Jacobian Validation: {model.name} ({model.num_dof} DOF)", type="module")
    beauty_print(f"Backend: {args.backend}", type="info")

    rng = np.random.default_rng(args.seed)

    if args.backend == 'numpy':
        # Set global backend
        robocore.set_backend('numpy')

        # NumPy backend comparison
        beauty_print("[1] Analytic vs Numeric Jacobian (NumPy)", type="module", centered=False)

        q = np.zeros(model.num_chain_dof)

        # Warmup
        jacobian(model, q, method='analytic')
        jacobian(model, q, method='numeric')

        # Timing
        n_runs = 100
        t0 = time.perf_counter()
        for _ in range(n_runs):
            Ja = jacobian(model, q, method='analytic')
        time_analytic = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            Jn = jacobian(model, q, method='numeric')
        time_numeric = (time.perf_counter() - t0) / n_runs * 1000

        # Compare
        Ja = jacobian(model, q, method='analytic')
        Jn = jacobian(model, q, method='numeric')
        diff = Ja - Jn

        beauty_print(f"Analytic time:  {time_analytic:.4f} ms")
        beauty_print(f"Numeric time:   {time_numeric:.4f} ms")
        beauty_print(f"Speedup:        {time_numeric/time_analytic:.2f}x", type="success")
        beauty_print(f"Difference (analytic - numeric):")
        beauty_print(f"  Linear block max:      {np.max(np.abs(diff[:3, :])):.3e}")
        beauty_print(f"  Angular block max:     {np.max(np.abs(diff[3:6, :])):.3e}")
        beauty_print(f"  Overall max:           {np.max(np.abs(diff)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff, 'fro'):.3e}")

        # Test multiple configurations
        beauty_print(f"[2] Accuracy across {args.samples} random configurations", type="module", centered=False)

        max_diffs = []
        for i in range(args.samples):
            q_rand = model.random_q(rng)

            Ja = jacobian(model, q_rand, method='analytic')
            Jn = jacobian(model, q_rand, method='numeric')
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
        # Set global backend
        robocore.set_backend('torch', device=str(device))

        beauty_print(f"PyTorch device: {device}", type="info")
        beauty_print("[1] Analytic vs Numeric vs Autograd Jacobian (PyTorch)", type="module")

        q = torch.zeros(model.num_chain_dof, dtype=torch.float64, device=device)

        # Compute all three using unified interface
        Ja = jacobian(model, q, method='analytic', device=device)
        Jn = jacobian(model, q, method='numeric', device=device)
        Jg = jacobian(model, q, method='autograd', device=device)

        # Compare
        diff_an = (Ja - Jn).cpu().numpy()
        diff_ag = (Ja - Jg).cpu().numpy()
        diff_ng = (Jn - Jg).cpu().numpy()

        beauty_print("Analytic vs Numeric:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_an)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_an, 'fro'):.3e}")

        beauty_print("Analytic vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ag)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ag, 'fro'):.3e}")

        beauty_print("Numeric vs Autograd:")
        beauty_print(f"  Max difference:        {np.max(np.abs(diff_ng)):.3e}")
        beauty_print(f"  Frobenius norm:        {np.linalg.norm(diff_ng, 'fro'):.3e}")

        # Timing
        beauty_print("[2] Performance comparison", type="module")
        n_runs = 50

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, method='analytic', device=device)
        time_analytic = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, method='numeric', device=device)
        time_numeric = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = jacobian(model, q, method='autograd', device=device)
        time_autograd = (time.perf_counter() - t0) / n_runs * 1000

        beauty_print(f"Analytic:   {time_analytic:.4f} ms")
        beauty_print(f"Numeric:    {time_numeric:.4f} ms")
        beauty_print(f"Autograd:   {time_autograd:.4f} ms")

    beauty_print("✓ Jacobian validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Jacobian validation")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='torch', help='Backend to test')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')    
    parser.add_argument('--device', default='cpu', help='PyTorch device (if torch backend)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
