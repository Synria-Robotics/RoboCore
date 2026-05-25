"""Command line entry point for ``python -m robocore.benchmark``."""

from __future__ import annotations

import argparse

from robocore.benchmark.kinematics import run_kinematics_benchmark
from robocore.benchmark.dynamics import run_dynamics_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robocore.benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    kin = subparsers.add_parser("kinematics", help="Run FK/Jacobian/IK backend benchmarks")
    kin.add_argument("--robot", default="alicia_d", choices=["alicia_d"])
    kin.add_argument("--urdf", default=None)
    kin.add_argument("--base-link", default="base_link")
    kin.add_argument("--end-link", default="link6")
    kin.add_argument("--backends", default="cpp,numpy,torch,pinocchio")
    kin.add_argument("--device", default="cpu")
    kin.add_argument("--samples", type=int, default=100)
    kin.add_argument("--ik-samples", type=int, default=20)
    kin.add_argument("--repeats", type=int, default=3)
    kin.add_argument("--inner-single", type=int, default=200)
    kin.add_argument("--inner-batch", type=int, default=20)
    kin.add_argument("--ik-noise", type=float, default=0.0)
    kin.add_argument("--seed", type=int, default=0)
    kin.add_argument("--output-dir", default="benchmark-results")
    kin.add_argument("--json", default=None, help="Optional explicit JSON output path")
    kin.add_argument("--markdown", default=None, help="Optional explicit Markdown output path")
    kin.add_argument("--no-write", action="store_true", help="Print only; do not write result files")

    dyn = subparsers.add_parser("dynamics", help="Run rigid-body dynamics backend benchmarks")
    dyn.add_argument("--robot", default="alicia_d", choices=["alicia_d"])
    dyn.add_argument("--urdf", default=None)
    dyn.add_argument("--base-link", default="base_link")
    dyn.add_argument("--end-link", default="tool0")
    dyn.add_argument("--backends", default="cpp,numpy,pinocchio")
    dyn.add_argument("--samples", type=int, default=100)
    dyn.add_argument("--repeats", type=int, default=3)
    dyn.add_argument("--inner-single", type=int, default=200)
    dyn.add_argument("--inner-batch", type=int, default=20)
    dyn.add_argument("--seed", type=int, default=0)
    dyn.add_argument("--atol", type=float, default=1e-8)
    dyn.add_argument("--output-dir", default="benchmark-results")
    dyn.add_argument("--json", default=None, help="Optional explicit JSON output path")
    dyn.add_argument("--markdown", default=None, help="Optional explicit Markdown output path")
    dyn.add_argument("--no-write", action="store_true", help="Print only; do not write result files")

    args = parser.parse_args(argv)
    if args.command == "kinematics":
        return run_kinematics_benchmark(args)
    if args.command == "dynamics":
        return run_dynamics_benchmark(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
