"""IK success-rate benchmark for Alicia-M v1.2 on FK-reachable poses (C++ backend).

Samples joint configurations inside URDF limits, computes FK targets with the
compiled C++ backend, then runs IK from configurable initial guesses and reports
success probability plus error / status breakdown.

Run::

    python -m robocore.benchmark.alicia_m_v12_ik_success_rate
    python -m robocore.benchmark.alicia_m_v12_ik_success_rate --samples 5000 --batch-size 256

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

import robocore as rc
from robocore.configs import resolve_description_path
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import _normalize_one_ik_result
from robocore.kinematics.ik_utils.ik_solver_cpp import IKSolverCpp
from robocore.modeling import RobotModel
from robocore.transform.utils import rotation_error


DEFAULT_URDF = "synriard://Alicia_M/v1_2/follower/urdf"


@dataclass
class BatchStats:
    total: int
    success: int
    success_rate: float
    pos_err_mean: float
    pos_err_max: float
    ori_err_mean: float
    ori_err_max: float
    iters_mean: float
    fk_closure_pos_max: float
    fk_closure_ori_max: float
    status_counts: dict[str, int]
    elapsed_s: float


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _default_output_dir() -> Path:
    return Path("benchmark-results") / "alicia_m_v12_ik_success"


def _build_q0(
    q_true: np.ndarray,
    *,
    mode: str,
    lo: np.ndarray,
    hi: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build IK initial guesses for one batch.

    :param q_true: (B, n) ground-truth chain joint values.
    :param mode: ``noisy_truth``, ``random``, or ``zero``.
    :param lo: (n,) lower joint limits.
    :param hi: (n,) upper joint limits.
    :param noise_std: Gaussian noise std (rad) for ``noisy_truth``.
    :param rng: NumPy RNG.
    :return: (B, n) initial guesses clipped to limits.
    """
    n = q_true.shape[1]
    if mode == "noisy_truth":
        q0 = q_true + rng.normal(0.0, noise_std, size=q_true.shape)
    elif mode == "random":
        q0 = rng.uniform(lo, hi, size=q_true.shape)
    elif mode == "zero":
        q0 = np.zeros_like(q_true)
    else:
        raise ValueError(f"Unknown q0 mode: {mode!r}")
    return np.clip(q0, lo, hi)


def _fk_closure_errors(
    model: RobotModel,
    targets: np.ndarray,
    q_solved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure FK(T) vs target after IK.

    :param model: robot model.
    :param targets: (B, 4, 4) target poses.
    :param q_solved: (B, n) solved chain joints.
    :return: position and orientation error arrays, shape (B,).
    """
    achieved = forward_kinematics(model, q_solved, return_end=True, backend="cpp")
    achieved = np.asarray(achieved, dtype=np.float64)
    if achieved.ndim == 2:
        achieved = achieved.reshape(1, 4, 4)
    pos_err = np.linalg.norm(achieved[:, :3, 3] - targets[:, :3, 3], axis=1)
    ori_err = np.empty(targets.shape[0], dtype=np.float64)
    for i in range(targets.shape[0]):
        ori_err[i] = np.linalg.norm(
            rotation_error(achieved[i, :3, :3], targets[i, :3, :3])
        )
    return pos_err, ori_err


def _collect_batch_stats(
    results: list[dict[str, Any]],
    *,
    fk_pos: np.ndarray,
    fk_ori: np.ndarray,
    elapsed_s: float,
) -> BatchStats:
    """Aggregate one IK batch into summary statistics."""
    total = len(results)
    success_flags = np.array([bool(r.get("success")) for r in results], dtype=bool)
    success = int(success_flags.sum())
    pos_err = np.array([float(r.get("pos_err", np.inf)) for r in results], dtype=np.float64)
    ori_err = np.array([float(r.get("ori_err", np.inf)) for r in results], dtype=np.float64)
    iters = np.array([int(r.get("iters", 0)) for r in results], dtype=np.int32)

    status_counts: dict[str, int] = {}
    for r in results:
        status = str(r.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return BatchStats(
        total=total,
        success=success,
        success_rate=float(success / total) if total else 0.0,
        pos_err_mean=float(np.mean(pos_err[success_flags])) if success else float("nan"),
        pos_err_max=float(np.max(pos_err[success_flags])) if success else float("nan"),
        ori_err_mean=float(np.mean(ori_err[success_flags])) if success else float("nan"),
        ori_err_max=float(np.max(ori_err[success_flags])) if success else float("nan"),
        iters_mean=float(np.mean(iters[success_flags])) if success else float("nan"),
        fk_closure_pos_max=float(np.max(fk_pos[success_flags])) if success else float("nan"),
        fk_closure_ori_max=float(np.max(fk_ori[success_flags])) if success else float("nan"),
        status_counts=status_counts,
        elapsed_s=elapsed_s,
    )


def _merge_status_counts(acc: dict[str, int], batch: dict[str, int]) -> None:
    for key, value in batch.items():
        acc[key] = acc.get(key, 0) + int(value)


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute Alicia-M v1.2 IK success benchmark."""
    urdf = Path(resolve_description_path(args.urdf or DEFAULT_URDF))

    rc.set_backend("cpp")
    model = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    n = model.num_chain_dof
    lo = np.asarray(model.chain_joint_limit_min, dtype=np.float64)
    hi = np.asarray(model.chain_joint_limit_max, dtype=np.float64)

    ik_solver = IKSolverCpp(
        model,
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
    )

    rng = np.random.default_rng(args.seed)
    total_samples = int(args.samples)
    batch_size = max(1, int(args.batch_size))

    all_status: dict[str, int] = {}
    total_success = 0
    total_count = 0
    pos_err_success: list[float] = []
    ori_err_success: list[float] = []
    iters_success: list[float] = []
    fk_pos_success: list[float] = []
    fk_ori_success: list[float] = []

    t_all = time.perf_counter()
    processed = 0
    batch_index = 0

    while processed < total_samples:
        current = min(batch_size, total_samples - processed)
        seed_q = int(args.seed) + batch_index * 1_000_003
        q_true = model.random_q_batch(current, seed=seed_q, scale=args.scale).astype(np.float64, copy=False)
        q0 = _build_q0(
            q_true,
            mode=args.q0_mode,
            lo=lo,
            hi=hi,
            noise_std=args.noise,
            rng=rng,
        )

        t0 = time.perf_counter()
        targets = np.asarray(
            forward_kinematics(model, q_true, return_end=True, backend="cpp"),
            dtype=np.float64,
        )
        if targets.ndim == 2:
            targets = targets.reshape(1, 4, 4)

        ik_kw = dict(
            method="dls",
            use_analytic_jacobian=True,
            adaptive_damping=True,
            adaptive_step=False,
            max_step_norm=0.5,
        )
        raw = ik_solver.solve(targets, q0, **ik_kw)
        elapsed = time.perf_counter() - t0

        if isinstance(raw, dict):
            results = []
            for i in range(current):
                row = {k: (v[i] if isinstance(v, list) and len(v) == current else v) for k, v in raw.items()}
                results.append(row)
        else:
            results = raw

        results = [
            _normalize_one_ik_result(r, model=model, max_iters=args.max_iters)
            for r in results
        ]

        q_solved = np.stack([np.asarray(r["q"], dtype=np.float64).ravel() for r in results], axis=0)
        fk_pos, fk_ori = _fk_closure_errors(model, targets, q_solved)
        batch_stats = _collect_batch_stats(results, fk_pos=fk_pos, fk_ori=fk_ori, elapsed_s=elapsed)

        total_success += batch_stats.success
        total_count += batch_stats.total
        _merge_status_counts(all_status, batch_stats.status_counts)

        mask = np.array([bool(r.get("success")) for r in results], dtype=bool)
        if mask.any():
            pos_err_success.extend(float(r["pos_err"]) for r, ok in zip(results, mask) if ok)
            ori_err_success.extend(float(r["ori_err"]) for r, ok in zip(results, mask) if ok)
            iters_success.extend(float(r["iters"]) for r, ok in zip(results, mask) if ok)
            fk_pos_success.extend(float(v) for v, ok in zip(fk_pos, mask) if ok)
            fk_ori_success.extend(float(v) for v, ok in zip(fk_ori, mask) if ok)

        processed += current
        batch_index += 1
        if args.verbose:
            print(
                f"  batch {batch_index}: {batch_stats.success}/{batch_stats.total} "
                f"({batch_stats.success_rate * 100:.1f}%) in {batch_stats.elapsed_s:.3f}s"
            )

    elapsed_all = time.perf_counter() - t_all
    success_rate = total_success / total_count if total_count else 0.0

    summary = {
        "robot": "Alicia_M_v1_2_follower",
        "urdf": str(urdf),
        "base_link": args.base_link,
        "end_link": args.end_link,
        "chain_dof": n,
        "backend": "cpp",
        "samples": total_count,
        "batch_size": batch_size,
        "joint_scale": args.scale,
        "q0_mode": args.q0_mode,
        "q0_noise_std": args.noise,
        "max_iters": args.max_iters,
        "pos_tol": args.pos_tol,
        "ori_tol": args.ori_tol,
        "seed": args.seed,
        "success_count": total_success,
        "success_rate": success_rate,
        "success_rate_percent": success_rate * 100.0,
        "elapsed_s": elapsed_all,
        "ms_per_sample": (elapsed_all / total_count * 1e3) if total_count else None,
        "status_counts": all_status,
        "successful_cases": {
            "pos_err_mean_m": float(np.mean(pos_err_success)) if pos_err_success else None,
            "pos_err_max_m": float(np.max(pos_err_success)) if pos_err_success else None,
            "ori_err_mean_rad": float(np.mean(ori_err_success)) if ori_err_success else None,
            "ori_err_max_rad": float(np.max(ori_err_success)) if ori_err_success else None,
            "iters_mean": float(np.mean(iters_success)) if iters_success else None,
            "fk_closure_pos_max_m": float(np.max(fk_pos_success)) if fk_pos_success else None,
            "fk_closure_ori_max_rad": float(np.max(fk_ori_success)) if fk_ori_success else None,
        },
    }

    metadata_block = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            "robocore": _version("synria-robocore"),
            "numpy": np.__version__,
        },
    }

    report_md = _format_markdown(summary, metadata_block)
    print(report_md)

    if not args.no_write:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "alicia_m_v12_ik_success.json"
        md_path = out_dir / "alicia_m_v12_ik_success.md"
        payload = {"metadata": metadata_block, "summary": summary}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(report_md, encoding="utf-8")
        print(f"\nWrote {json_path} and {md_path}")

    return 0 if success_rate >= args.min_success_rate else 1


def _format_markdown(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Render human-readable benchmark report."""
    ok = summary["successful_cases"]
    lines = [
        "# Alicia-M v1.2 IK Success Rate (C++ backend)",
        "",
        "## Configuration",
        "",
        f"- Robot: `{summary['robot']}`",
        f"- URDF: `{summary['urdf']}`",
        f"- Chain: `{summary['base_link']}` → `{summary['end_link']}` ({summary['chain_dof']} DOF)",
        f"- Backend: `{summary['backend']}`",
        f"- Samples: {summary['samples']} (batch size {summary['batch_size']})",
        f"- Joint sampling scale: {summary['joint_scale']} (1.0 = full URDF limits)",
        f"- Initial guess: `{summary['q0_mode']}`",
        f"- q0 noise std: {summary['q0_noise_std']} rad",
        f"- IK tolerances: pos={summary['pos_tol']} m, ori={summary['ori_tol']} rad",
        f"- max_iters: {summary['max_iters']}",
        f"- Seed: {summary['seed']}",
        "",
        "## Result",
        "",
        f"- **Success rate: {summary['success_rate_percent']:.2f}%** "
        f"({summary['success_count']}/{summary['samples']})",
        f"- Total time: {summary['elapsed_s']:.3f} s",
        f"- Throughput: {summary['ms_per_sample']:.3f} ms/sample",
        "",
        "## Status breakdown",
        "",
        "| Status | Count | Fraction |",
        "| --- | ---: | ---: |",
    ]
    total = summary["samples"]
    for status, count in sorted(summary["status_counts"].items()):
        frac = count / total if total else 0.0
        lines.append(f"| {status} | {count} | {frac * 100:.2f}% |")

    lines.extend(
        [
            "",
            "## Successful cases (error stats)",
            "",
            f"- Position error: mean={ok['pos_err_mean_m']}, max={ok['pos_err_max_m']} m",
            f"- Orientation error: mean={ok['ori_err_mean_rad']}, max={ok['ori_err_max_rad']} rad",
            f"- Iterations: mean={ok['iters_mean']}",
            f"- FK closure: max |Δp|={ok['fk_closure_pos_max_m']} m, "
            f"max |Δori|={ok['fk_closure_ori_max_rad']} rad",
            "",
            "## Environment",
            "",
            f"- Created: {metadata['created_at']}",
            f"- Python: {metadata['python']}",
            f"- Platform: {metadata['platform']}",
            f"- RoboCore: {metadata['versions'].get('robocore')}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark IK success rate on FK-reachable poses for Alicia-M v1.2 (C++ backend).",
    )
    parser.add_argument("--urdf", default=None, help=f"URDF path or URI (default: {DEFAULT_URDF})")
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="link6", help="6-DOF arm tip (default: link6)")
    parser.add_argument("--samples", type=int, default=2000, help="Number of joint-space samples")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for FK/IK")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Joint range scale for random_q_batch (1.0 = full limits)",
    )
    parser.add_argument(
        "--q0-mode",
        choices=("noisy_truth", "random", "zero"),
        default="noisy_truth",
        help="Initial guess strategy for IK",
    )
    parser.add_argument("--noise", type=float, default=0.15, help="q0 noise std (rad) for noisy_truth")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--pos-tol", type=float, default=1e-3)
    parser.add_argument("--ori-tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--no-write", action="store_true", help="Print report only")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch progress")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.0,
        help="Exit code 1 if success rate is below this threshold (0–1)",
    )
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0.0 < args.scale <= 1.0:
        parser.error("--scale must be in (0, 1]")

    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
