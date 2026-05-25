"""Unified kinematics benchmark CLI.

The benchmark intentionally records policy metadata next to timings so release
results can be reproduced and compared over time.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robocore.configs import resolve_description_path
from robocore.kinematics import forward_kinematics, inverse_kinematics, jacobian
from robocore.modeling import RobotModel


_RC4_CPP_BASELINE_MS = {
    ("fk", "single"): 0.003,
    ("fk", "batch"): 0.055,
    ("jacobian", "single"): 0.003,
    ("jacobian", "batch"): 0.063,
    ("ik", "single"): 0.016,
    ("ik", "batch"): 0.400,
}


@dataclass
class BenchmarkCase:
    operation: str
    backend: str
    mode: str
    ms_per_call: float | None
    speedup_vs_cpp: float | None
    max_abs_error: float | None
    success_count: int | None
    total_count: int | None
    status: str
    message: str


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _default_urdf(robot: str) -> str:
    if robot == "alicia_d":
        return resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")
    raise ValueError(f"Unsupported benchmark robot: {robot}")


def _bench(fn: Callable[[], Any], *, repeats: int, inner_loops: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner_loops):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / inner_loops * 1e3


def _pinocchio_context(model: RobotModel, urdf: Path, end_link: str):
    try:
        import pinocchio as pin
    except ImportError as exc:
        return None, f"pinocchio unavailable: {exc}"

    pin_model = pin.buildModelFromUrdf(str(urdf))
    pin_data = pin_model.createData()
    chain_indices = model._get_joint_indices(model.base_link, model.end_link)
    joint_names = [model.joint_list[idx].name for idx in chain_indices]
    q_indices: list[int] = []
    v_indices: list[int] = []
    for name in joint_names:
        if not pin_model.existJointName(name):
            return None, f"pinocchio joint missing: {name}"
        jid = pin_model.getJointId(name)
        q_indices.append(pin_model.idx_qs[jid])
        v_indices.append(pin_model.idx_vs[jid])

    if pin_model.existFrame(end_link):
        frame_id = pin_model.getFrameId(end_link)
    else:
        return None, f"pinocchio frame missing: {end_link}"
    return (pin, pin_model, pin_data, q_indices, v_indices, frame_id), "ok"


def _pin_q(pin, pin_model, q_chain: np.ndarray, q_indices: list[int]) -> np.ndarray:
    q_full = pin.neutral(pin_model).copy()
    for i, qidx in enumerate(q_indices):
        q_full[qidx] = float(q_chain[i])
    return q_full


def _pin_fk(ctx, q: np.ndarray) -> np.ndarray:
    pin, pin_model, pin_data, q_indices, _v_indices, frame_id = ctx
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q_full = _pin_q(pin, pin_model, q, q_indices)
        pin.forwardKinematics(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)
        return np.asarray(pin_data.oMf[frame_id].homogeneous, dtype=np.float64)
    return np.stack([_pin_fk(ctx, row) for row in q], axis=0)


def _pin_jacobian(ctx, q: np.ndarray) -> np.ndarray:
    pin, pin_model, pin_data, q_indices, v_indices, frame_id = ctx
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q_full = _pin_q(pin, pin_model, q, q_indices)
        pin.computeJointJacobians(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)
        J_full = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.asarray(J_full[:, v_indices], dtype=np.float64)
    return np.stack([_pin_jacobian(ctx, row) for row in q], axis=0)


def _max_abs(ref: np.ndarray, other: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(ref, dtype=np.float64) - np.asarray(other, dtype=np.float64))))


def _format_table(cases: list[BenchmarkCase]) -> str:
    lines = [
        "| Operation | Backend | Mode | ms/call | C++ speedup | Error | Validation |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for c in cases:
        ms = "-" if c.ms_per_call is None else f"{c.ms_per_call:.3f}"
        speed = "-" if c.speedup_vs_cpp is None else f"{c.speedup_vs_cpp:.1f}x"
        err = "-" if c.max_abs_error is None else f"{c.max_abs_error:.2e}"
        if c.success_count is None:
            validation = c.status
        else:
            validation = f"{c.success_count}/{c.total_count} {c.status}"
        backend = f"**{c.backend}**" if c.backend == "cpp" else c.backend
        lines.append(f"| {c.operation} | {backend} | {c.mode} | {ms} | {speed} | {err} | {validation} |")
    return "\n".join(lines)


def run_kinematics_benchmark(args: Any) -> int:
    backends = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    urdf = Path(args.urdf or _default_urdf(args.robot))
    model = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)

    rng = np.random.default_rng(args.seed)
    n = model.num_chain_dof
    q_single = model.random_q(seed=args.seed, scale=0.5)
    q_batch = model.random_q_batch(args.samples, seed=args.seed + 1, scale=0.5)
    q_ik = model.random_q_batch(args.ik_samples, seed=args.seed + 2, scale=0.5)
    targets_ik = forward_kinematics(model, q_ik, return_end=True, backend="cpp")
    q0_ik = np.clip(
        q_ik + rng.normal(0.0, float(args.ik_noise), size=q_ik.shape),
        model.chain_joint_limit_min,
        model.chain_joint_limit_max,
    )

    fk_ref_single = forward_kinematics(model, q_single, return_end=True, backend="numpy")
    fk_ref_batch = forward_kinematics(model, q_batch, return_end=True, backend="numpy")
    jac_ref_single = jacobian(model, q_single, method="analytic", backend="numpy")
    jac_ref_batch = jacobian(model, q_batch, method="analytic", backend="numpy")

    pin_ctx, pin_msg = _pinocchio_context(model, urdf, args.end_link)
    cases: list[BenchmarkCase] = []
    timings: dict[tuple[str, str, str], float] = {}

    def add_case(operation: str, backend: str, mode: str, ms: float | None, error: float | None, status: str, message: str, success: int | None = None, total: int | None = None) -> None:
        cpp_ms = ms if backend == "cpp" else timings.get((operation, "cpp", mode))
        speedup = None if ms is None or cpp_ms is None or cpp_ms == 0 else ms / cpp_ms
        if backend == "cpp" and ms is not None:
            timings[(operation, backend, mode)] = ms
            baseline = _RC4_CPP_BASELINE_MS.get((operation, mode))
            if baseline is not None and ms > baseline * 1.2:
                message = f"{message} WARNING: slower than rc4 baseline ({baseline:.3f} ms) by >20%."
        cases.append(BenchmarkCase(operation, backend, mode, ms, speedup, error, success, total, status, message))

    for backend in backends:
        if backend in {"cpp", "numpy", "torch"}:
            try:
                fk_single = forward_kinematics(model, q_single, return_end=True, backend=backend, device=args.device)
                fk_batch = forward_kinematics(model, q_batch, return_end=True, backend=backend, device=args.device)
                ms_single = _bench(lambda b=backend: forward_kinematics(model, q_single, return_end=True, backend=b, device=args.device), repeats=args.repeats, inner_loops=args.inner_single)
                ms_batch = _bench(lambda b=backend: forward_kinematics(model, q_batch, return_end=True, backend=b, device=args.device), repeats=args.repeats, inner_loops=args.inner_batch)
                add_case("fk", backend, "single", ms_single, _max_abs(fk_ref_single, fk_single), "ok", "FK matched NumPy reference.")
                add_case("fk", backend, "batch", ms_batch, _max_abs(fk_ref_batch, fk_batch), "ok", "FK batch matched NumPy reference.")

                jac_single = jacobian(model, q_single, method="analytic", backend=backend, device=args.device)
                jac_batch = jacobian(model, q_batch, method="analytic", backend=backend, device=args.device)
                ms_single = _bench(lambda b=backend: jacobian(model, q_single, method="analytic", backend=b, device=args.device), repeats=args.repeats, inner_loops=args.inner_single)
                ms_batch = _bench(lambda b=backend: jacobian(model, q_batch, method="analytic", backend=b, device=args.device), repeats=args.repeats, inner_loops=args.inner_batch)
                add_case("jacobian", backend, "single", ms_single, _max_abs(jac_ref_single, jac_single), "ok", "Jacobian matched NumPy reference.")
                add_case("jacobian", backend, "batch", ms_batch, _max_abs(jac_ref_batch, jac_batch), "ok", "Jacobian batch matched NumPy reference.")

                ik_result = inverse_kinematics(model, targets_ik, q0_ik, backend=backend, method="dls", max_iters=200, use_analytic_jacobian=True)
                ik_success = sum(1 for r in ik_result if r.get("success"))
                ms_batch = _bench(lambda b=backend: inverse_kinematics(model, targets_ik, q0_ik, backend=b, method="dls", max_iters=200, use_analytic_jacobian=True), repeats=args.repeats, inner_loops=max(1, args.inner_batch // 2))
                add_case("ik", backend, "batch", ms_batch, None, "ok" if ik_success == args.ik_samples else "failed", "IK closure validation.", ik_success, args.ik_samples)
            except Exception as exc:
                add_case("all", backend, "all", None, None, "error", str(exc))
        elif backend == "pinocchio":
            if pin_ctx is None:
                add_case("all", backend, "all", None, None, "skipped", pin_msg)
                continue
            fk_single = _pin_fk(pin_ctx, q_single)
            fk_batch = _pin_fk(pin_ctx, q_batch)
            ms_single = _bench(lambda: _pin_fk(pin_ctx, q_single), repeats=args.repeats, inner_loops=args.inner_single)
            ms_batch = _bench(lambda: _pin_fk(pin_ctx, q_batch), repeats=args.repeats, inner_loops=args.inner_batch)
            add_case("fk", backend, "single", ms_single, _max_abs(fk_ref_single, fk_single), "ok", "Pinocchio FK comparison.")
            add_case("fk", backend, "batch", ms_batch, _max_abs(fk_ref_batch, fk_batch), "ok", "Pinocchio FK loop comparison.")
            jac_single = _pin_jacobian(pin_ctx, q_single)
            jac_batch = _pin_jacobian(pin_ctx, q_batch)
            ms_single = _bench(lambda: _pin_jacobian(pin_ctx, q_single), repeats=args.repeats, inner_loops=args.inner_single)
            ms_batch = _bench(lambda: _pin_jacobian(pin_ctx, q_batch), repeats=args.repeats, inner_loops=args.inner_batch)
            add_case("jacobian", backend, "single", ms_single, _max_abs(jac_ref_single, jac_single), "ok", "Pinocchio Jacobian comparison.")
            add_case("jacobian", backend, "batch", ms_batch, _max_abs(jac_ref_batch, jac_batch), "ok", "Pinocchio Jacobian loop comparison.")
        else:
            add_case("all", backend, "all", None, None, "skipped", f"unknown backend: {backend}")

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "robot": args.robot,
            "urdf": str(urdf),
            "base_link": args.base_link,
            "end_link": args.end_link,
            "chain_dof": n,
            "samples": args.samples,
            "ik_samples": args.ik_samples,
            "seed": args.seed,
            "repeats": args.repeats,
            "inner_single": args.inner_single,
            "inner_batch": args.inner_batch,
            "ik_noise": args.ik_noise,
            "tolerance": {"fk_jacobian_max_abs": 1e-8},
            "versions": {
                "robocore": _version("synria-robocore"),
                "numpy": np.__version__,
                "torch": _version("torch"),
                "pinocchio": _version("pin"),
            },
        },
        "results": [asdict(c) for c in cases],
    }

    table = _format_table(cases)
    markdown = "# RoboCore Kinematics Benchmark\n\n" + table + "\n"
    print(table)

    if not args.no_write:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = Path(args.json) if args.json else out_dir / "kinematics.json"
        md_path = Path(args.markdown) if args.markdown else out_dir / "kinematics.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        print(f"\nWrote {json_path} and {md_path}")

    failed = [c for c in cases if c.status == "error" or (c.operation in {"fk", "jacobian"} and c.max_abs_error is not None and c.max_abs_error > 1e-8)]
    failed.extend(c for c in cases if c.operation == "ik" and c.success_count is not None and c.success_count != c.total_count)
    return 1 if failed else 0
