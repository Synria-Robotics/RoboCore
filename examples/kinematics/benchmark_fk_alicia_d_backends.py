"""Benchmark FK: NumPy vs PyTorch vs Eigen/pybind (Alicia-D URDF).

Run from repo root or RoboCore after installing the package and building the C++ extension::

    pip install pybind11
    # Eigen3 headers (e.g. brew install eigen), then:
    pip install -e .

    python examples/kinematics/benchmark_fk_alicia_d_backends.py

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import robocore as rc
from robocore.utils.backend import get_backend_manager
from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy
from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
from robocore.kinematics.fk_utils.fk_solver_cpp import FKSolverCpp
from robocore.modeling import RobotModel
from robocore.configs import resolve_description_path


def _default_urdf() -> str:
    return resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")


def _bench(fn, *, repeats: int, inner_loops: int, sync=None) -> float:
    """Return seconds for ``inner_loops`` calls, repeated ``repeats`` times (best of)."""
    best = float("inf")
    for _ in range(repeats):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        for _ in range(inner_loops):
            fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def _cuda_sync(device) -> None:
    if device is None:
        return
    device_str = str(device)
    if device_str.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device_str)


def _print_matrix(title: str, T: np.ndarray) -> None:
    """Pretty-print a 4x4 pose."""
    print(title)
    with np.printoptions(precision=8, suppress=True, floatmode="fixed"):
        print(T)


def _diff_report(name: str, ref: np.ndarray, other: np.ndarray, *, atol: float, rtol: float) -> None:
    """Print max/mean abs diff and allclose vs ref."""
    d = other.astype(np.float64) - ref.astype(np.float64)
    print(
        f"  vs numpy ({name}): max|diff|={np.max(np.abs(d)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d)):.3e}  "
        f"allclose(atol={atol}, rtol={rtol})={np.allclose(ref, other, atol=atol, rtol=rtol)}"
    )


def _print_diff_matrix(label: str, diff: np.ndarray) -> None:
    """Print a 4x4 difference matrix (scientific, show small entries)."""
    print(f"  {label} (other - numpy):")
    with np.printoptions(precision=4, suppress=False, floatmode="maxprec_equal"):
        print(np.asarray(diff, dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to Alicia-D URDF (default: synriard://Alicia_D/v5_6/gripper_100mm/urdf)",
    )
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="link6", help="6-DOF arm tip (before gripper).")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner-single", type=int, default=2000, help="FK calls per trial (batch=1).")
    parser.add_argument("--inner-batch", type=int, default=200, help="FK calls per trial (batch=100).")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, cuda, cuda:0, …). Default: auto (CUDA if available, else CPU).",
    )
    parser.add_argument(
        "--no-pk-pin",
        action="store_true",
        help="Skip pytorch_kinematics + pinocchio timing (optional deps).",
    )
    args = parser.parse_args()

    urdf = args.urdf or _default_urdf()
    if "://" in str(urdf):
        urdf = resolve_description_path(urdf)
    urdf = Path(urdf)

    # NumPy + C++ first while global backend stays numpy (torch would poison array types).
    rc.set_backend("numpy")
    model_np = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    fk_np = FKSolverNumPy(model_np)
    fk_cpp = FKSolverCpp(model_np)

    rng = np.random.default_rng(0)
    n = model_np.num_chain_dof
    q1 = rng.standard_normal(n).astype(np.float64)
    q100 = rng.standard_normal((100, n)).astype(np.float64)

    atol, rtol = 1e-9, 1e-9

    rc.set_backend("numpy")
    out_np1 = fk_np.solve(q1, return_end_only=True)
    out_np100 = fk_np.solve(q100, return_end_only=True)

    out_c1 = fk_cpp.solve(q1, return_end_only=True)
    out_c100 = fk_cpp.solve(q100, return_end_only=True)

    rc.set_backend("torch", device=args.device)
    model_th = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    fk_th = FKSolverTorch(model_th)
    torch_device_str = get_backend_manager().get_device()

    out_th1 = fk_th.solve(q1, return_end_only=True, device=args.device)
    if hasattr(out_th1, "detach"):
        out_th1 = out_th1.detach().cpu().numpy()

    out_th100 = fk_th.solve(q100, return_end_only=True, device=args.device)
    if hasattr(out_th100, "detach"):
        out_th100 = out_th100.detach().cpu().numpy()

    pk_pin_ctx = None
    out_pin1 = out_pin100 = None
    pk_pin_skip_msg = None
    if not args.no_pk_pin:
        import benchmark_alicia_pk_pin as _bpk

        ok_pk, msg_pk = _bpk.pk_pin_status()
        if ok_pk:
            pk_pin_ctx = _bpk.AliciaPkPinFKJac.build(model_np, urdf, args.base_link, args.end_link, torch_device_str)
            out_pin1 = pk_pin_ctx.pin_fk_matrix(q1)
            out_pin100 = pk_pin_ctx.pin_fk_batch_matrix(q100)
        else:
            pk_pin_skip_msg = msg_pk

    # --- Numeric correctness: poses and diffs (reference = NumPy) ---
    print("=" * 72)
    print("Correctness (FK end-effector 4x4, reference = numpy)")
    print("=" * 72)
    print(f"q (single, chain DOF={n}): {np.array2string(q1, precision=6, suppress_small=True)}")
    print()
    _print_matrix("numpy  T (batch=1):", np.asarray(out_np1, dtype=np.float64))
    print()
    print("Diff vs numpy (single):")
    _diff_report("torch", out_np1, out_th1, atol=atol, rtol=rtol)
    _print_diff_matrix("torch - numpy", np.asarray(out_th1, dtype=np.float64) - np.asarray(out_np1, dtype=np.float64))
    _diff_report("cpp+eigen", out_np1, out_c1, atol=atol, rtol=rtol)
    _print_diff_matrix("cpp - numpy", np.asarray(out_c1, dtype=np.float64) - np.asarray(out_np1, dtype=np.float64))
    if out_pin1 is not None:
        _diff_report("pinocchio", out_np1, out_pin1, atol=atol, rtol=rtol)
        _print_diff_matrix(
            "pinocchio - numpy",
            np.asarray(out_pin1, dtype=np.float64) - np.asarray(out_np1, dtype=np.float64),
        )
    elif pk_pin_skip_msg is not None:
        print(f"  pinocchio skipped: {pk_pin_skip_msg}")
    print()
    print(f"batch=100: shape {out_np100.shape}")
    d_th = np.asarray(out_th100, dtype=np.float64) - np.asarray(out_np100, dtype=np.float64)
    print(
        f"  torch vs numpy: max|diff|={np.max(np.abs(d_th)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d_th)):.3e}  "
        f"allclose={np.allclose(out_np100, out_th100, atol=atol, rtol=rtol)}"
    )
    d_cpp = np.asarray(out_c100, dtype=np.float64) - np.asarray(out_np100, dtype=np.float64)
    print(
        f"  cpp   vs numpy: max|diff|={np.max(np.abs(d_cpp)):.3e}  "
        f"mean|diff|={np.mean(np.abs(d_cpp)):.3e}  "
        f"allclose={np.allclose(out_np100, out_c100, atol=atol, rtol=rtol)}"
    )
    if out_pin100 is not None:
        d_pin = np.asarray(out_pin100, dtype=np.float64) - np.asarray(out_np100, dtype=np.float64)
        print(
            f"  pin   vs numpy: max|diff|={np.max(np.abs(d_pin)):.3e}  "
            f"mean|diff|={np.mean(np.abs(d_pin)):.3e}  "
            f"allclose={np.allclose(out_np100, out_pin100, atol=atol, rtol=rtol)}"
        )
    print("=" * 72)
    print()

    # Warmup: numpy → cpp → torch (same order as correctness and timing).
    rc.set_backend("numpy")
    for _ in range(20):
        fk_np.solve(q1, return_end_only=True)
    for _ in range(20):
        fk_cpp.solve(q1, return_end_only=True)
    rc.set_backend("torch", device=args.device)
    for _ in range(20):
        fk_th.solve(q1, return_end_only=True, device=args.device)
    _cuda_sync(torch_device_str)

    rc.set_backend("numpy")
    t_np_1 = _bench(lambda: fk_np.solve(q1, return_end_only=True), repeats=args.repeats, inner_loops=args.inner_single)
    t_cpp_1 = _bench(
        lambda: fk_cpp.solve(q1, return_end_only=True), repeats=args.repeats, inner_loops=args.inner_single
    )
    rc.set_backend("torch", device=args.device)
    t_th_1 = _bench(
        lambda: fk_th.solve(q1, return_end_only=True, device=args.device),
        repeats=args.repeats,
        inner_loops=args.inner_single,
        sync=lambda: _cuda_sync(torch_device_str),
    )

    rc.set_backend("numpy")
    t_np_100 = _bench(lambda: fk_np.solve(q100, return_end_only=True), repeats=args.repeats, inner_loops=args.inner_batch)
    t_cpp_100 = _bench(
        lambda: fk_cpp.solve(q100, return_end_only=True), repeats=args.repeats, inner_loops=args.inner_batch
    )
    rc.set_backend("torch", device=args.device)
    t_th_100 = _bench(
        lambda: fk_th.solve(q100, return_end_only=True, device=args.device),
        repeats=args.repeats,
        inner_loops=args.inner_batch,
        sync=lambda: _cuda_sync(torch_device_str),
    )

    t_pk_1 = t_pk_100 = t_pin_1 = t_pin_100 = None
    if not args.no_pk_pin:
        if pk_pin_ctx is not None:
            ctx = pk_pin_ctx
            for _ in range(20):
                ctx.pk_fk1(q1)
                ctx.pin_fk1(q1)
            ctx.sync_torch()
            t_pk_1 = _bench(lambda: ctx.pk_fk1(q1), repeats=args.repeats, inner_loops=args.inner_single)
            t_pin_1 = _bench(lambda: ctx.pin_fk1(q1), repeats=args.repeats, inner_loops=args.inner_single)
            t_pk_100 = _bench(lambda: ctx.pk_fk_batch(q100), repeats=args.repeats, inner_loops=args.inner_batch)
            t_pin_100 = _bench(lambda: ctx.pin_fk_batch(q100), repeats=args.repeats, inner_loops=args.inner_batch)
            ctx.sync_torch()
        elif pk_pin_skip_msg is not None:
            print(f"(skip pytorch_kinematics/pinocchio FK: {pk_pin_skip_msg})", flush=True)

    def ms_per_call(t_sec: float, inner: int) -> float:
        return t_sec / inner * 1e3

    col_w = 22
    print(f"Model: {Path(urdf).name}  chain DOF={n}  end_link={args.end_link}")
    print(f"Torch device: {torch_device_str}" + ("" if args.device is None else f" (arg --device={args.device!r})"))
    print()
    print(f"{'backend':<22} {'batch=1 (ms/call)':<{col_w}} {'batch=100 (ms/call)':<24}")
    print("-" * (22 + col_w + 24 + 2))
    print(f"{'numpy':<22} {ms_per_call(t_np_1, args.inner_single):<{col_w}.6f} {ms_per_call(t_np_100, args.inner_batch):<24.6f}")
    print(
        f"{'cpp+eigen':<22} {ms_per_call(t_cpp_1, args.inner_single):<{col_w}.6f} "
        f"{ms_per_call(t_cpp_100, args.inner_batch):<24.6f}"
    )
    print(f"{'torch (RoboCore)':<22} {ms_per_call(t_th_1, args.inner_single):<{col_w}.6f} {ms_per_call(t_th_100, args.inner_batch):<24.6f}")
    if t_pk_1 is not None:
        print(
            f"{'pytorch_kinematics':<22} {ms_per_call(t_pk_1, args.inner_single):<{col_w}.6f} "
            f"{ms_per_call(t_pk_100, args.inner_batch):<24.6f}"
        )
        print(
            f"{'pinocchio':<22} {ms_per_call(t_pin_1, args.inner_single):<{col_w}.6f} "
            f"{ms_per_call(t_pin_100, args.inner_batch):<24.6f}"
        )


if __name__ == "__main__":
    main()
