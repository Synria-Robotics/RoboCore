"""Benchmark IK: NumPy vs PyTorch vs Eigen/pybind DLS (Alicia-D URDF).

Targets are **reachable poses**: ``RobotModel.random_q_batch`` samples each chain joint inside limits
(scaled range, same as ``random_pose_batch`` / ``03b_demo_ik_parallel.py``), then FK gives ``T_target``.
The old ``standard_normal`` ``q_*`` could lie **outside** URDF limits, so IK often stalled while FK still
returned a pose — that produced spurious ``success=False`` / large closure errors.

Initial guess ``q0`` is ``q_star`` + Gaussian noise, **clipped** to chain limits. Layout like ``benchmark_fk_alicia_d_backends``:
batch=1 and a small multi-sample batch (default 10) for timing. Default ``--inner-*`` / ``--repeats`` are **lower than the FK benchmark** because IK (especially NumPy batch) is much slower — if the script seems to hang after the correctness block, it is usually still in the timing loop; use ``--skip-timing`` or reduce ``--inner-batch``.

Correctness is **FK closure** only (each backend: one batch=1 solve + one batch-B solve):
``FKSolverNumPy`` checks ``FK(q)`` vs ``T_target``. That answers “is the implementation plausible?”

The **timing** section repeats ``solve()`` many more times to estimate stable **ms/call** (noise from OS /
Python). It is optional: ``--skip-timing`` if you only need the closure table. Parameters
``--inner-single``, ``--inner-batch``, ``--repeats`` apply only to timing, not to correctness.

Run from repo root or RoboCore after building native extensions::

    pip install pybind11
    # Eigen3 headers (e.g. brew install eigen), then:
    pip install -e .

    python examples/kinematics/benchmark_ik_alicia_d_backends.py

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import robocore as rc
from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy
from robocore.kinematics.ik_utils.ik_solver_numpy import IKSolverNumPy
from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
from robocore.kinematics.ik_utils.ik_solver_cpp import IKSolverCpp
from robocore.modeling import RobotModel
from robocore.configs import resolve_description_path
from robocore.transform.utils import rotation_error


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
    device_str = str(device)
    if device_str.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device_str)


def _q_row_to_numpy(q) -> np.ndarray:
    """Flatten one IK ``q`` row to 1D numpy."""
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    return np.asarray(q, dtype=np.float64).ravel()


def _ik_result_q_matrix(result: dict) -> np.ndarray:
    """Stack batch IK ``result['q']`` to (B, n)."""
    q = result["q"]
    if hasattr(q, "detach"):
        return q.detach().cpu().numpy().astype(np.float64)
    rows = [_q_row_to_numpy(row) for row in q]
    return np.stack(rows, axis=0)


def _ik_result_success_array(result: dict) -> np.ndarray:
    success = result["success"]
    if hasattr(success, "detach"):
        success = success.detach().cpu().numpy()
    return np.asarray(success, dtype=bool)


def _fk_closure_errors(fk: FKSolverNumPy, T_target: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FK achieved pose vs target; same FK for all backends.

    :param T_target: (4,4) or (B,4,4)
    :param Q: (n,) or (B,n)
    :return: pos_err (B,), ori_err (B,) in meters / radians axis-angle norm
    """
    T_target = np.asarray(T_target, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if T_target.ndim == 2:
        T_target = T_target.reshape(1, 4, 4)
        Q = Q.reshape(1, -1)
    T_hat = fk.solve(Q, return_end_only=True)
    if T_hat.ndim == 2:
        T_hat = T_hat.reshape(1, 4, 4)
    B = T_target.shape[0]
    dp = np.linalg.norm(T_hat[:, :3, 3] - T_target[:, :3, 3], axis=1)
    dori = np.empty(B, dtype=np.float64)
    for i in range(B):
        dori[i] = np.linalg.norm(rotation_error(T_hat[i, :3, :3], T_target[i, :3, :3]))
    return dp, dori


def _print_fk_closure_block(
    label: str,
    dp: np.ndarray,
    dori: np.ndarray,
    *,
    pos_tol: float,
    ori_tol: float,
) -> None:
    """Print max/mean FK closure errors and counts within solver tolerances."""
    ok = (dp < pos_tol) & (dori < ori_tol)
    print(
        f"  {label}: max|Δp|={np.max(dp):.3e} mean|Δp|={np.mean(dp):.3e}  "
        f"max|Δori|={np.max(dori):.3e} mean|Δori|={np.mean(dori):.3e}  "
        f"within tol (pos<{pos_tol}, ori<{ori_tol}): {int(ok.sum())}/{len(ok)}"
    )


_TIMING_EPILOG = """
Correctness vs timing
  Correctness: a few IK solves to print FK closure (already enough to see max/mean error).
  Timing:      many repeated solve() calls so ms/call is not dominated by timer noise.

Timing parameters (see also _bench() in this file)
  inner-single  In ONE timing trial for batch=1: call solve(T_4x4, q0) this many times in a row.
  inner-batch   In ONE timing trial for batch=B: call solve(T_Bx4x4, q0_B) this many times in a row.
  repeats       Run that many independent trials; keep the fastest trial duration (reduces jitter).
  Printed ms/call = (best trial seconds) / inner-*.

Legacy FK-style values (inner-single=2000, inner-batch=200, repeats=7) are fine for FK but
often make this IK script run for minutes; this script uses smaller IK defaults unless you override.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_TIMING_EPILOG,
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to Alicia-D URDF (default: synriard://Alicia_D/v5_6/gripper_100mm/urdf)",
    )
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="link6", help="6-DOF arm tip (before gripper).")
    parser.add_argument("--batch", type=int, default=100, help="IK batch size for multi-sample path (default 10).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timing repeats per backend (best of); FK benchmark uses 7 — IK defaults lower for runtime.",
    )
    parser.add_argument(
        "--inner-single",
        type=int,
        default=400,
        help="IK calls per timing trial (batch=1). IK is slower than FK; FK script uses 2000.",
    )
    parser.add_argument(
        "--inner-batch",
        type=int,
        default=30,
        help="IK calls per timing trial (each call solves a full batch of size --batch). Default << FK's 200.",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Only run FK-closure checks; skip warmup and ms/call benchmark.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (e.g. cpu, cuda).")
    parser.add_argument(
        "--no-pk-pin",
        action="store_true",
        help="Skip pytorch_kinematics + pinocchio timing (optional deps).",
    )
    parser.add_argument("--noise", type=float, default=0.15, help="Gaussian noise std on q0 (rad), then clip to limits.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for limit-scaled q_star (RobotModel.random_q_batch / random_pose_batch style).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Joint range scale for random_q_batch (0–1, default 0.5 = middle 50%% of each limit span).",
    )
    args = parser.parse_args()

    if args.batch < 1:
        print("--batch must be >= 1", file=sys.stderr)
        sys.exit(1)

    urdf = args.urdf or _default_urdf()
    if "://" in str(urdf):
        urdf = resolve_description_path(urdf)
    urdf = Path(urdf)

    rc.set_backend("numpy")
    model_np = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    fk_np = FKSolverNumPy(model_np)
    ik_np = IKSolverNumPy(model_np)
    pos_tol = float(ik_np.pos_tol)
    ori_tol = float(ik_np.ori_tol)

    rc.set_backend("torch", device=args.device)
    model_th = RobotModel(str(urdf), base_link=args.base_link, end_link=args.end_link)
    ik_th = IKSolverTorch(model_th, device=args.device)

    n = model_np.num_chain_dof
    B = args.batch
    lo = model_np.chain_joint_limit_min
    hi = model_np.chain_joint_limit_max
    noise_rng = np.random.default_rng(int(args.seed) + 911)
    # Separate seeds so batch=1 and batch=B are independent draws (like the old benchmark).
    seed_1 = int(args.seed) + 10_000_003
    seed_b = int(args.seed)

    q_star1 = np.asarray(model_np.random_q(seed=seed_1, scale=args.scale), dtype=np.float64)
    T_np1 = np.asarray(model_np.random_pose(seed=seed_1, scale=args.scale), dtype=np.float64)
    q_star_b = model_np.random_q_batch(B, seed=seed_b, scale=args.scale).astype(np.float64, copy=False)
    T_batch = np.asarray(
        model_np.random_pose_batch(B, seed=seed_b, scale=args.scale),
        dtype=np.float64,
    )

    q0_1 = np.clip(q_star1 + noise_rng.standard_normal(n) * args.noise, lo, hi)
    q0_b = np.clip(q_star_b + noise_rng.standard_normal((B, n)) * args.noise, lo, hi)

    ik_kw = dict(
        method="dls",
        use_analytic_jacobian=True,
        adaptive_damping=True,
        adaptive_step=False,
        max_step_norm=0.5,
    )

    active_backend = None

    def _switch_backend(backend: str) -> None:
        nonlocal active_backend
        if active_backend == backend:
            return
        if backend == "torch":
            rc.set_backend("torch", device=args.device)
        else:
            rc.set_backend(backend)
        active_backend = backend

    def _np_solve(T, q0):
        _switch_backend("numpy")
        return ik_np.solve(T, q0, **ik_kw)

    def _th_solve(T, q0):
        _switch_backend("torch")
        return ik_th.solve(
            T,
            q0,
            method="dls",
            use_analytic_jacobian=True,
            adaptive_damping=True,
            adaptive_step=False,
            max_step_norm=0.5,
        )

    def _cpp_solve(T, q0):
        _switch_backend("cpp")
        return ik_cpp.solve(T, q0, **ik_kw)

    r_np1 = _np_solve(T_np1, q0_1)
    r_th1 = _th_solve(T_np1, q0_1)
    q_np1_flat = _q_row_to_numpy(r_np1["q"])
    q_th1_flat = _q_row_to_numpy(r_th1["q"])

    r_np_b = _np_solve(T_batch, q0_b)
    r_th_b = _th_solve(T_batch, q0_b)
    Q_np_b = _ik_result_q_matrix(r_np_b)
    Q_th_b = _ik_result_q_matrix(r_th_b)

    ik_cpp = IKSolverCpp(model_np)
    r_cpp1 = _cpp_solve(T_np1, q0_1)
    r_cpp_b = _cpp_solve(T_batch, q0_b)
    q_cpp1_flat = _q_row_to_numpy(r_cpp1["q"])
    Q_cpp_b = _ik_result_q_matrix(r_cpp_b)

    # --- Correctness: FK closure only (reference FK = numpy) ---
    print("=" * 72)
    print("Correctness (FK closure: FKSolverNumPy(FK(q_ik)) vs T_target; IK tolerances from numpy solver)")
    print("=" * 72)
    print(
        f"chain DOF={n}  noise_std={args.noise}  batch={B}  "
        f"seed={args.seed}  scale={args.scale}  "
        f"(targets: RobotModel.random_pose / random_pose_batch; q0 = clip(q_star + noise))"
    )
    print(f"q0 (batch=1): {np.array2string(q0_1, precision=6, suppress_small=True)}")
    print()
    print(
        f"batch=1: IK flags — numpy success={r_np1['success']}  iters={r_np1['iters']}  "
        f"reported pos_err={r_np1['pos_err']:.4e}  ori_err={r_np1['ori_err']:.4e}"
    )
    dp1_np, dor1_np = _fk_closure_errors(fk_np, T_np1, q_np1_flat)
    dp1_th, dor1_th = _fk_closure_errors(fk_np, T_np1, q_th1_flat)
    _print_fk_closure_block("FK numpy q=numpy IK", dp1_np, dor1_np, pos_tol=pos_tol, ori_tol=ori_tol)
    _print_fk_closure_block("FK numpy q=torch IK", dp1_th, dor1_th, pos_tol=pos_tol, ori_tol=ori_tol)
    dp1_cpp, dor1_cpp = _fk_closure_errors(fk_np, T_np1, q_cpp1_flat)
    _print_fk_closure_block("FK numpy q=cpp IK", dp1_cpp, dor1_cpp, pos_tol=pos_tol, ori_tol=ori_tol)
    print()
    succ_np = _ik_result_success_array(r_np_b)
    succ_th = _ik_result_success_array(r_th_b)
    line = f"batch={B}: IK success count — numpy={int(succ_np.sum())}/{B}  torch={int(succ_th.sum())}/{B}"
    succ_cpp = _ik_result_success_array(r_cpp_b)
    line += f"  cpp={int(succ_cpp.sum())}/{B}"
    print(line)
    dp_b_np, dor_b_np = _fk_closure_errors(fk_np, T_batch, Q_np_b)
    dp_b_th, dor_b_th = _fk_closure_errors(fk_np, T_batch, Q_th_b)
    _print_fk_closure_block("FK numpy q=numpy IK", dp_b_np, dor_b_np, pos_tol=pos_tol, ori_tol=ori_tol)
    _print_fk_closure_block("FK numpy q=torch IK", dp_b_th, dor_b_th, pos_tol=pos_tol, ori_tol=ori_tol)
    dp_b_cpp, dor_b_cpp = _fk_closure_errors(fk_np, T_batch, Q_cpp_b)
    _print_fk_closure_block("FK numpy q=cpp IK", dp_b_cpp, dor_b_cpp, pos_tol=pos_tol, ori_tol=ori_tol)

    ikx_pk_pin = None
    if not args.no_pk_pin:
        import benchmark_alicia_pk_pin as _bpk

        ok_deps, dep_msg = _bpk.pk_pin_status()
        if ok_deps:
            try:
                import torch
                from pytorch_kinematics.transforms import Transform3d

                fkctx = _bpk.AliciaPkPinFKJac.build(model_np, urdf, args.base_link, args.end_link, args.device)
                ikx_pk_pin = _bpk.AliciaPkPinIK.build(
                    fkctx,
                    pos_tol=pos_tol,
                    ori_tol=ori_tol,
                    max_iters=int(ik_np.max_iters),
                    damping=1e-3,
                    step_size=0.2,
                    num_retries=1,
                )
                ikx_pk_pin.set_T_pin_batch(q_star_b)
                T_pk1, T_pkb = ikx_pk_pin.pk_targets_from_q_star(q_star1, q_star_b)

                sol_pk1 = ikx_pk_pin.ik_pk.solve(Transform3d(matrix=T_pk1))
                q_pk1 = sol_pk1.solutions[0, 0, :].detach().cpu().numpy()
                T_pin1 = _bpk.pin_target_pose_from_chain_q(
                    fkctx.pin_model,
                    fkctx.pin_data,
                    q_star1,
                    fkctx.pin_q_indices,
                    fkctx.end_frame_id,
                    fkctx.end_joint_id,
                )
                r_pin1 = _bpk.solve_ik_pinocchio(
                    fkctx.pin_model,
                    fkctx.pin_data,
                    T_pin1,
                    q0_1,
                    fkctx.pin_q_indices,
                    fkctx.pin_v_indices,
                    ikx_pk_pin.pin_ik_end_joint_id,
                    fkctx.end_frame_id,
                    pos_tol,
                    ori_tol,
                    int(ik_np.max_iters),
                    1e-3,
                    0.2,
                    robot_model=model_np,
                )
                q_pin1 = np.asarray(r_pin1["q"], dtype=np.float64)
                q_t = torch.as_tensor(q_pk1, dtype=fkctx.dtype, device=fkctx.device).unsqueeze(0)
                ret_pk = fkctx.chain.forward_kinematics(q_t, end_only=False)
                T_ach_pk = ret_pk[fkctx.end_link].get_matrix()[0].detach().cpu().numpy()
                T_tgt_pk = T_pk1[0].detach().cpu().numpy()
                dpos_pk = np.linalg.norm(T_ach_pk[:3, 3] - T_tgt_pk[:3, 3])
                Rerr_pk = T_tgt_pk[:3, :3] @ T_ach_pk[:3, :3].T
                tr = np.trace(Rerr_pk)
                theta_pk = float(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
                T_ach_pin = _bpk.pin_ee_homogeneous(
                    fkctx.pin_model,
                    fkctx.pin_data,
                    _bpk.pin_q_full_from_chain(fkctx.pin_model, q_pin1, fkctx.pin_q_indices),
                    fkctx.end_frame_id,
                    fkctx.end_joint_id,
                )
                dpos_pin = np.linalg.norm(T_ach_pin[:3, 3] - T_pin1[:3, 3])
                Rerr_pin = T_pin1[:3, :3] @ T_ach_pin[:3, :3].T
                trp = np.trace(Rerr_pin)
                theta_pin = float(np.arccos(np.clip((trp - 1.0) / 2.0, -1.0, 1.0)))
                print()
                print(
                    f"batch=1 (PK/Pin own targets): pytorch_kinematics converged={bool(sol_pk1.converged[0, 0].item())}  "
                    f"|Δp| vs PK target={dpos_pk:.3e}  axis-angle~={theta_pk:.3e} rad"
                )
                print(
                    f"batch=1: pinocchio success={r_pin1['success']}  iters={r_pin1['iters']}  "
                    f"|Δp| vs Pin target={dpos_pin:.3e}  axis-angle~={theta_pin:.3e} rad"
                )
            except Exception as ex:
                print(f"(PK/Pin IK sanity check skipped: {ex})", flush=True)
                ikx_pk_pin = None
        else:
            print(f"(PK/Pin optional deps missing: {dep_msg})", flush=True)

    print("=" * 72)
    print()

    if args.skip_timing:
        print("(skip-timing: no warmup / ms/call table)", flush=True)
        return

    print(
        "Timing: warmup + benchmark (IK is slow; use --skip-timing or smaller --inner-* if this takes too long)...",
        flush=True,
    )

    # Warmup
    for _ in range(20):
        _np_solve(T_np1, q0_1)
        _th_solve(T_np1, q0_1)
    _cuda_sync(args.device)
    for _ in range(20):
        _cpp_solve(T_np1, q0_1)

    t_np_1 = _bench(lambda: _np_solve(T_np1, q0_1), repeats=args.repeats, inner_loops=args.inner_single)
    t_th_1 = _bench(
        lambda: _th_solve(T_np1, q0_1),
        repeats=args.repeats,
        inner_loops=args.inner_single,
        sync=lambda: _cuda_sync(args.device),
    )
    t_cpp_1 = _bench(
        lambda: _cpp_solve(T_np1, q0_1), repeats=args.repeats, inner_loops=args.inner_single
    )

    for _ in range(5):
        _np_solve(T_batch, q0_b)
        _th_solve(T_batch, q0_b)
    _cuda_sync(args.device)
    for _ in range(5):
        _cpp_solve(T_batch, q0_b)

    t_np_b = _bench(
        lambda: _np_solve(T_batch, q0_b), repeats=args.repeats, inner_loops=args.inner_batch
    )
    t_th_b = _bench(
        lambda: _th_solve(T_batch, q0_b),
        repeats=args.repeats,
        inner_loops=args.inner_batch,
        sync=lambda: _cuda_sync(args.device),
    )
    t_cpp_b = _bench(
        lambda: _cpp_solve(T_batch, q0_b), repeats=args.repeats, inner_loops=args.inner_batch
    )

    t_pk_1 = t_pk_b = t_pin_1 = t_pin_b = None
    inner_pin_b = args.inner_batch
    if ikx_pk_pin is not None and ikx_pk_pin.last_T_pkb is not None:
        import benchmark_alicia_pk_pin as _bpp

        T_pin1_ref = _bpp.pin_target_pose_from_chain_q(
            ikx_pk_pin.fkctx.pin_model,
            ikx_pk_pin.fkctx.pin_data,
            q_star1,
            ikx_pk_pin.fkctx.pin_q_indices,
            ikx_pk_pin.fkctx.end_frame_id,
            ikx_pk_pin.fkctx.end_joint_id,
        )
        for _ in range(10):
            ikx_pk_pin.pk_solve1(ikx_pk_pin.last_T_pk1)
            ikx_pk_pin.pin_solve1(T_pin1_ref, q0_1)
        for _ in range(5):
            ikx_pk_pin.pk_solve_batch(ikx_pk_pin.last_T_pkb)
            ikx_pk_pin.pin_solve_batch(q0_b)
        ikx_pk_pin.sync_torch()
        t_pk_1 = _bench(
            lambda: ikx_pk_pin.pk_solve1(ikx_pk_pin.last_T_pk1),
            repeats=args.repeats,
            inner_loops=args.inner_single,
        )
        t_pin_1 = _bench(
            lambda: ikx_pk_pin.pin_solve1(T_pin1_ref, q0_1),
            repeats=args.repeats,
            inner_loops=args.inner_single,
        )
        t_pk_b = _bench(
            lambda: ikx_pk_pin.pk_solve_batch(ikx_pk_pin.last_T_pkb),
            repeats=args.repeats,
            inner_loops=args.inner_batch,
        )
        # Pin batch = B sequential Python CLIK loops per call — keep inner_loops small vs RoboCore batch.
        inner_pin_b = max(1, min(args.inner_batch, 4))
        t_pin_b = _bench(
            lambda: ikx_pk_pin.pin_solve_batch(q0_b),
            repeats=args.repeats,
            inner_loops=inner_pin_b,
        )
        ikx_pk_pin.sync_torch()

    def ms_per_call(t_sec: float, inner: int) -> float:
        return t_sec / inner * 1e3

    col_w = 22
    batch_col = f"batch={B} (ms/call)"
    print(f"Model: {Path(urdf).name}  chain DOF={n}  end_link={args.end_link}")
    print(f"Torch device: {args.device}")
    print()
    print(f"{'backend':<22} {'batch=1 (ms/call)':<{col_w}} {batch_col:<26}")
    print("-" * (22 + col_w + 26 + 2))
    print(f"{'numpy':<22} {ms_per_call(t_np_1, args.inner_single):<{col_w}.6f} {ms_per_call(t_np_b, args.inner_batch):<26.6f}")
    print(f"{'torch (RoboCore)':<22} {ms_per_call(t_th_1, args.inner_single):<{col_w}.6f} {ms_per_call(t_th_b, args.inner_batch):<26.6f}")
    print(
        f"{'cpp+eigen':<22} {ms_per_call(t_cpp_1, args.inner_single):<{col_w}.6f} "
        f"{ms_per_call(t_cpp_b, args.inner_batch):<26.6f}"
    )
    if t_pk_1 is not None:
        print(
            f"{'pytorch_kinematics':<22} {ms_per_call(t_pk_1, args.inner_single):<{col_w}.6f} "
            f"{ms_per_call(t_pk_b, args.inner_batch):<26.6f}"
        )
        print(
            f"{'pinocchio (CLIK)':<22} {ms_per_call(t_pin_1, args.inner_single):<{col_w}.6f} "
            f"{ms_per_call(t_pin_b, args.inner_batch):<26.6f}"
        )

    ms_b_np = ms_per_call(t_np_b, args.inner_batch)
    ms_b_th = ms_per_call(t_th_b, args.inner_batch)
    note = (
        f"Note: batch={B} column is ONE solve(T, q0) that runs {B} IK instances inside. "
        f"Amortized ms/target ≈ numpy {ms_b_np / B:.3f}, torch {ms_b_th / B:.3f}"
    )
    note += f", cpp {ms_per_call(t_cpp_b, args.inner_batch) / B:.3f}"
    if t_pk_b is not None:
        note += f", pytorch_kinematics {ms_per_call(t_pk_b, args.inner_batch) / B:.3f}"
        note += f", pinocchio {ms_per_call(t_pin_b, inner_pin_b) / B:.3f}"
    note += (
        ". C++ batch is ``solve_batch``: B independent IK solves in C++ with one pybind call (still ~linear in B). "
        "NumPy/Torch batch IK vectorizes FK/Jacobian across active rows each iteration — not one big matmul like batched FK — "
        "so their batch column is not ~100× the batch=1 column. "
        "pytorch_kinematics uses batched ``PseudoInverseIK``; pinocchio column is B sequential CLIK solves (Python loop); "
        "pinocchio batch timing uses a smaller inner-loop count than RoboCore because each call is very expensive."
    )
    print()
    print(note)


if __name__ == "__main__":
    main()
