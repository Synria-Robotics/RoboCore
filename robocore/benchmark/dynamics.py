"""Unified dynamics benchmark CLI."""

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

from robocore import dynamics
from robocore.configs import resolve_description_path
from robocore.modeling import RobotModel


@dataclass
class DynamicsBenchmarkCase:
    operation: str
    backend: str
    mode: str
    ms_per_call: float | None
    speedup_vs_cpp: float | None
    max_abs_error: float | None
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


def _max_abs(ref: Any, other: Any) -> float:
    return float(np.max(np.abs(np.asarray(ref, dtype=np.float64) - np.asarray(other, dtype=np.float64))))


def _pin_context(model: RobotModel, urdf: Path):
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
    return (pin, pin_model, pin_data, q_indices, v_indices), "ok"


def _pin_qva(ctx, q, v=None, a=None):
    pin, pin_model, _pin_data, q_indices, v_indices = ctx
    q_full = pin.neutral(pin_model).copy()
    v_full = np.zeros(pin_model.nv, dtype=np.float64)
    a_full = np.zeros(pin_model.nv, dtype=np.float64)
    for i, qidx in enumerate(q_indices):
        q_full[qidx] = float(q[i])
    if v is not None:
        for i, vidx in enumerate(v_indices):
            v_full[vidx] = float(v[i])
    if a is not None:
        for i, vidx in enumerate(v_indices):
            a_full[vidx] = float(a[i])
    return q_full, v_full, a_full


def _pin_ops(ctx, q, v, a, tau):
    pin, pin_model, pin_data, _q_indices, v_indices = ctx
    q_full, v_full, a_full = _pin_qva(ctx, q, v, a)
    tau_full = np.zeros(pin_model.nv, dtype=np.float64)
    for i, vidx in enumerate(v_indices):
        tau_full[vidx] = float(tau[i])

    id_out = np.asarray(pin.rnea(pin_model, pin_data, q_full, v_full, a_full), dtype=np.float64)[v_indices]
    M_full = np.asarray(pin.crba(pin_model, pin_data, q_full), dtype=np.float64)
    M = M_full[np.ix_(v_indices, v_indices)]
    M = 0.5 * (M + M.T)
    g = np.asarray(pin.computeGeneralizedGravity(pin_model, pin_data, q_full), dtype=np.float64)[v_indices]
    nle = np.asarray(pin.nonLinearEffects(pin_model, pin_data, q_full, v_full), dtype=np.float64)[v_indices]
    fd = np.linalg.solve(M, tau - nle)
    return {"id": id_out, "mass_matrix": M, "gravity": g, "nle": nle, "fd": fd}


def _format_table(cases: list[DynamicsBenchmarkCase]) -> str:
    lines = [
        "| Operation | Backend | Mode | ms/call | C++ speedup | Error | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        backend = f"**{case.backend}**" if case.backend == "cpp" else case.backend
        ms = "-" if case.ms_per_call is None else f"{case.ms_per_call:.3f}"
        speed = "-" if case.speedup_vs_cpp is None else f"{case.speedup_vs_cpp:.1f}x"
        err = "-" if case.max_abs_error is None else f"{case.max_abs_error:.2e}"
        lines.append(f"| {case.operation} | {backend} | {case.mode} | {ms} | {speed} | {err} | {case.status} |")
    return "\n".join(lines)


def _format_metadata(metadata: dict[str, Any]) -> str:
    versions = metadata["versions"]
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Python | {metadata['python']} |",
        f"| Platform | {metadata['platform']} |",
        f"| Processor | {metadata['processor']} |",
        f"| Robot | {metadata['robot']} |",
        f"| Chain | {metadata['base_link']} -> {metadata['end_link']} ({metadata['chain_dof']} DOF) |",
        f"| Samples | {metadata['samples']} |",
        f"| Seed | {metadata['seed']} |",
        f"| Repeats | {metadata['repeats']} |",
        f"| Inner single loops | {metadata['inner_single']} |",
        f"| Inner batch loops | {metadata['inner_batch']} |",
        f"| RoboCore | {versions.get('robocore')} |",
        f"| NumPy | {versions.get('numpy')} |",
        f"| Pinocchio | {versions.get('pinocchio')} |",
    ]
    return "\n".join(lines)


def run_dynamics_benchmark(args: Any) -> int:
    backends = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    urdf = Path(args.urdf or _default_urdf(args.robot))
    model = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    rng = np.random.default_rng(args.seed)
    n = model.num_chain_dof
    q = rng.uniform(-0.4, 0.4, n)
    v = rng.uniform(-0.1, 0.1, n)
    a = rng.uniform(-0.1, 0.1, n)
    tau = rng.uniform(-0.5, 0.5, n)
    q_batch = rng.uniform(-0.4, 0.4, (args.samples, n))
    v_batch = rng.uniform(-0.1, 0.1, (args.samples, n))
    a_batch = rng.uniform(-0.1, 0.1, (args.samples, n))
    tau_batch = rng.uniform(-0.5, 0.5, (args.samples, n))

    refs = {
        "id": dynamics.inverse_dynamics(model, q, v, a, backend="numpy"),
        "mass_matrix": dynamics.mass_matrix(model, q, backend="numpy"),
        "gravity": dynamics.gravity(model, q, backend="numpy"),
        "nle": dynamics.nonlinear_effects(model, q, v, backend="numpy"),
        "fd": dynamics.forward_dynamics(model, q, v, tau, backend="numpy"),
    }

    pin_ctx, pin_msg = _pin_context(model, urdf)
    timings: dict[tuple[str, str], float] = {}
    cases: list[DynamicsBenchmarkCase] = []

    def add(operation: str, backend: str, mode: str, ms: float | None, err: float | None, status: str, message: str):
        cpp_ms = ms if backend == "cpp" else timings.get((operation, mode))
        speed = None if ms is None or cpp_ms is None or cpp_ms == 0 else ms / cpp_ms
        if backend == "cpp" and ms is not None:
            timings[(operation, mode)] = ms
        cases.append(DynamicsBenchmarkCase(operation, backend, mode, ms, speed, err, status, message))

    for backend in backends:
        if backend in {"cpp", "numpy"}:
            try:
                ops = {
                    "id": lambda: dynamics.inverse_dynamics(model, q, v, a, backend=backend),
                    "mass_matrix": lambda: dynamics.mass_matrix(model, q, backend=backend),
                    "gravity": lambda: dynamics.gravity(model, q, backend=backend),
                    "nle": lambda: dynamics.nonlinear_effects(model, q, v, backend=backend),
                    "fd": lambda: dynamics.forward_dynamics(model, q, v, tau, backend=backend),
                }
                for name, fn in ops.items():
                    out = fn()
                    ms = _bench(fn, repeats=args.repeats, inner_loops=args.inner_single)
                    add(name, backend, "single", ms, _max_abs(refs[name], out), "ok", "Matched NumPy reference.")
                if backend == "cpp":
                    batch_ops = {
                        "id": lambda: dynamics.inverse_dynamics(model, q_batch, v_batch, a_batch, backend="cpp"),
                        "mass_matrix": lambda: dynamics.mass_matrix(model, q_batch, backend="cpp"),
                        "fd": lambda: dynamics.forward_dynamics(model, q_batch, v_batch, tau_batch, backend="cpp"),
                    }
                    for name, fn in batch_ops.items():
                        ms = _bench(fn, repeats=args.repeats, inner_loops=args.inner_batch)
                        add(name, backend, "batch", ms, None, "ok", "C++ batch path executed.")
            except Exception as exc:
                add("all", backend, "all", None, None, "error", str(exc))
        elif backend == "pinocchio":
            if pin_ctx is None:
                add("all", backend, "all", None, None, "skipped", pin_msg)
                continue
            pin_ops = _pin_ops(pin_ctx, q, v, a, tau)
            for name, out in pin_ops.items():
                ms = _bench(lambda nm=name: _pin_ops(pin_ctx, q, v, a, tau)[nm], repeats=args.repeats, inner_loops=args.inner_single)
                add(name, backend, "single", ms, _max_abs(refs[name], out), "ok", "Pinocchio projected-chain comparison.")
        else:
            add("all", backend, "all", None, None, "skipped", f"unknown backend: {backend}")

    metadata = {
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
        "seed": args.seed,
        "repeats": args.repeats,
        "inner_single": args.inner_single,
        "inner_batch": args.inner_batch,
        "versions": {
            "robocore": _version("synria-robocore"),
            "numpy": np.__version__,
            "pinocchio": _version("pin"),
        },
    }
    payload = {
        "metadata": metadata,
        "results": [asdict(case) for case in cases],
    }
    table = _format_table(cases)
    markdown = (
        "# RoboCore Dynamics Benchmark\n\n"
        "Fixed-base rigid-body dynamics CPU benchmark. Speedup is normalized to "
        "the RoboCore C++/Eigen backend for the same operation and mode.\n\n"
        "## Policy\n\n"
        f"{_format_metadata(metadata)}\n\n"
        "## Results\n\n"
        f"{table}\n"
    )
    print(table)

    if not args.no_write:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = Path(args.json) if args.json else out_dir / "dynamics.json"
        md_path = Path(args.markdown) if args.markdown else out_dir / "dynamics.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        print(f"\nWrote {json_path} and {md_path}")

    failed = [case for case in cases if case.status == "error"]
    failed.extend(case for case in cases if case.max_abs_error is not None and case.max_abs_error > args.atol)
    return 1 if failed else 0
