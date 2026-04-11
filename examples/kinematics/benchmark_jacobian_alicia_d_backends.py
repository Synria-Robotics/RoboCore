"""Benchmark Jacobian (analytic): NumPy vs PyTorch vs Eigen/pybind (Alicia-D URDF).

Run from repo root or RoboCore after installing the package and building the C++ extension::

    pip install pybind11
    # Eigen3 headers (e.g. brew install eigen), then:
    pip install -e .

    python examples/kinematics/benchmark_jacobian_alicia_d_backends.py

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import robocore as rc
from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
from robocore.kinematics.jacobian_utils.jacobian_solver_cpp import JacobianSolverCpp
from robocore.modeling import RobotModel


def _default_urdf() -> Path:
    here = Path(__file__).resolve()
    # .../RoboCore/examples/kinematics/this.py -> parents[3] = workspace (ws_robocore)
    ws = here.parents[3]
    return (
        ws
        / "Synria-Robot-Descriptions"
        / "synriard"
        / "urdf"
        / "Alicia_D_v5_6"
        / "Alicia_D_v5_6_gripper_100mm.urdf"
    )


def _bench(fn, *, repeats: int, inner_loops: int) -> float:
    """Return seconds for ``inner_loops`` calls, repeated ``repeats`` times (best of)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner_loops):
            fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def _print_jacobian(title: str, J: np.ndarray) -> None:
    """Pretty-print a 6 x n Jacobian."""
    print(title)
    with np.printoptions(precision=8, suppress=True, floatmode="fixed", linewidth=140):
        print(np.asarray(J, dtype=np.float64))


def _diff_report(name: str, ref: np.ndarray, other: np.ndarray, *, atol: float, rtol: float) -> None:
    """Print max/mean abs diff and allclose vs ref."""
    d = other.astype(np.float64) - ref.astype(np.float64)
    print(
        f"  vs numpy ({name}): max|diff|={np.max(np.abs(d)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d)):.3e}  "
        f"allclose(atol={atol}, rtol={rtol})={np.allclose(ref, other, atol=atol, rtol=rtol)}"
    )


def _print_diff_jacobian(label: str, diff: np.ndarray) -> None:
    """Print Jacobian difference (scientific, show small entries)."""
    print(f"  {label} (other - numpy):")
    with np.printoptions(precision=4, suppress=False, floatmode="maxprec_equal", linewidth=140):
        print(np.asarray(diff, dtype=np.float64))


def _torch_j_to_numpy(J_th) -> np.ndarray:
    """Convert torch Jacobian to NumPy (CPU float64)."""
    if hasattr(J_th, "detach"):
        return J_th.detach().cpu().numpy().astype(np.float64)
    return np.asarray(J_th, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Path to Alicia-D URDF (default: Synria-Robot-Descriptions/.../Alicia_D_v5_6_gripper_100mm.urdf)",
    )
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="link6", help="6-DOF arm tip (before gripper).")
    parser.add_argument(
        "--target-link",
        default=None,
        help="Partial chain Jacobian (e.g. link4); default uses end_link.",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner-single", type=int, default=2000, help="Jacobian calls per trial (batch=1).")
    parser.add_argument("--inner-batch", type=int, default=200, help="Jacobian calls per trial (batch=100).")
    parser.add_argument("--device", default="cpu", help="Torch device (e.g. cpu, cuda).")
    args = parser.parse_args()

    urdf = args.urdf or _default_urdf()
    if not urdf.is_file():
        print(f"URDF not found: {urdf}", file=sys.stderr)
        print("Pass --urdf or clone Synria-Robot-Descriptions next to RoboCore.", file=sys.stderr)
        sys.exit(1)

    rc.set_backend("numpy")
    model_np = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    jac_np = JacobianSolverNumPy(model_np)

    rc.set_backend("torch", device=args.device)
    model_th = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    jac_th = JacobianSolverTorch(model_th)

    rng = np.random.default_rng(0)
    n = model_np.num_chain_dof
    q1 = rng.standard_normal(n).astype(np.float64)
    q100 = rng.standard_normal((100, n)).astype(np.float64)

    atol, rtol = 1e-9, 1e-9

    tl_kw = {"method": "analytic", "target_link": args.target_link}

    J_np1 = jac_np.solve(q1, **tl_kw)
    J_th1 = jac_th.solve(q1, device=args.device, **tl_kw)
    J_th1 = _torch_j_to_numpy(J_th1)

    J_np100 = jac_np.solve(q100, **tl_kw)
    J_th100 = jac_th.solve(q100, device=args.device, **tl_kw)
    J_th100 = _torch_j_to_numpy(J_th100)

    jac_cpp = JacobianSolverCpp(model_np, target_link=args.target_link)
    J_c1 = jac_cpp.solve(q1, method="analytic")
    J_c100 = jac_cpp.solve(q100, method="analytic")

    # --- Numeric correctness (reference = NumPy) ---
    print("=" * 72)
    print("Correctness (analytic Jacobian 6 x n, reference = numpy)")
    print("=" * 72)
    print(f"q (single, chain DOF={n}): {np.array2string(q1, precision=6, suppress_small=True)}")
    print(f"target_link={args.target_link!r}")
    print()
    _print_jacobian("numpy  J (batch=1):", J_np1)
    print()
    print("Diff vs numpy (single):")
    _diff_report("torch", J_np1, J_th1, atol=atol, rtol=rtol)
    _print_diff_jacobian("torch - numpy", J_th1 - J_np1)
    _diff_report("cpp+eigen", J_np1, J_c1, atol=atol, rtol=rtol)
    _print_diff_jacobian("cpp - numpy", J_c1 - J_np1)
    print()
    print(f"batch=100: shape {J_np100.shape}")
    d_th = J_th100 - J_np100
    print(
        f"  torch vs numpy: max|diff|={np.max(np.abs(d_th)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d_th)):.3e}  "
        f"allclose={np.allclose(J_np100, J_th100, atol=atol, rtol=rtol)}"
    )
    d_cpp = J_c100 - J_np100
    print(
        f"  cpp   vs numpy: max|diff|={np.max(np.abs(d_cpp)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d_cpp)):.3e}  "
        f"allclose={np.allclose(J_np100, J_c100, atol=atol, rtol=rtol)}"
    )
    print("=" * 72)
    print()

    # Warmup
    for _ in range(20):
        jac_np.solve(q1, **tl_kw)
        jac_th.solve(q1, device=args.device, **tl_kw)
    for _ in range(20):
        jac_cpp.solve(q1, method="analytic")

    t_np_1 = _bench(lambda: jac_np.solve(q1, **tl_kw), repeats=args.repeats, inner_loops=args.inner_single)
    t_th_1 = _bench(
        lambda: jac_th.solve(q1, device=args.device, **tl_kw),
        repeats=args.repeats,
        inner_loops=args.inner_single,
    )
    t_cpp_1 = _bench(
        lambda: jac_cpp.solve(q1, method="analytic"), repeats=args.repeats, inner_loops=args.inner_single
    )

    t_np_100 = _bench(lambda: jac_np.solve(q100, **tl_kw), repeats=args.repeats, inner_loops=args.inner_batch)
    t_th_100 = _bench(
        lambda: jac_th.solve(q100, device=args.device, **tl_kw),
        repeats=args.repeats,
        inner_loops=args.inner_batch,
    )
    t_cpp_100 = _bench(
        lambda: jac_cpp.solve(q100, method="analytic"), repeats=args.repeats, inner_loops=args.inner_batch
    )

    def ms_per_call(t_sec: float, inner: int) -> float:
        return t_sec / inner * 1e3

    print(f"Model: {urdf.name}  chain DOF={n}  end_link={args.end_link}")
    print(f"Torch device: {args.device}")
    print()
    print(f"{'backend':<10} {'batch=1 (ms/call)':<22} {'batch=100 (ms/call)':<24}")
    print("-" * 56)
    print(f"{'numpy':<10} {ms_per_call(t_np_1, args.inner_single):<22.6f} {ms_per_call(t_np_100, args.inner_batch):<24.6f}")
    print(f"{'torch':<10} {ms_per_call(t_th_1, args.inner_single):<22.6f} {ms_per_call(t_th_100, args.inner_batch):<24.6f}")
    print(
        f"{'cpp+eigen':<10} {ms_per_call(t_cpp_1, args.inner_single):<22.6f} "
        f"{ms_per_call(t_cpp_100, args.inner_batch):<24.6f}"
    )


if __name__ == "__main__":
    main()
