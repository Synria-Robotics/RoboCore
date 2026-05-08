"""Inverse Kinematics Parallel Demo

This demo demonstrates parallel/batch inverse kinematics. Targets are either:

- **random-pose** (default): ``RobotModel.random_pose_batch`` — FK of ``random_q_batch``
  inside scaled joint limits (reachable poses, usually easier for IK from random inits).
- **fk-joints**: same joint lists as ``01b_demo_fk_parallel.py`` → FK targets (still
  reachable, but hard poses + single random guess often yield low ``success`` counts).

Then NumPy / Torch / C++ batch IK is timed and cross-checked (mirrors 03a_demo_ik.py).

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import argparse
import time

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.utils.backend import to_numpy
from robocore.transform.utils import rotation_error


def _targets_from_joint_batch(robot_model, q_batch):
    """Build (B, 4, 4) target poses via FK (NumPy backend).

    :param robot_model: RobotModel instance
    :param q_batch: (B, n_dof) joint configurations
    :return: Target homogeneous transforms, shape (B, 4, 4)
    """
    rc.set_backend('numpy')
    q_batch = np.asarray(q_batch, dtype=np.float64)
    if q_batch.ndim == 1:
        q_batch = q_batch.reshape(1, -1)
    T = forward_kinematics(robot_model, q_batch, return_end=True)
    Tn = to_numpy(T)
    if Tn.ndim == 2:
        Tn = Tn[np.newaxis, ...]
    return Tn


def compute_ik_batch(robot_model, backend, target_poses, ik_kwargs):
    """Run batch IK for given backend.

    :param robot_model: RobotModel instance
    :param backend: Backend name ('numpy', 'torch', or 'cpp')
    :param target_poses: Array (B, 4, 4)
    :param ik_kwargs: Keyword arguments for inverse_kinematics
    :return: Dictionary with list of per-sample result dicts and wall-clock time
    """
    rc.set_backend(backend)
    start_time = time.time()
    results = inverse_kinematics(robot_model, target_poses, **ik_kwargs)
    elapsed_time = time.time() - start_time
    if not isinstance(results, list):
        results = [results]
    return {
        'results': results,
        'time': elapsed_time,
    }


def _max_q_diff(results_a, results_b):
    """Max L2 norm of q difference over batch where both samples succeeded."""
    max_d = 0.0
    for ra, rb in zip(results_a, results_b):
        if not (ra.get('success') and rb.get('success')):
            continue
        qa = np.asarray(to_numpy(ra['q']), dtype=np.float64).ravel()
        qb = np.asarray(to_numpy(rb['q']), dtype=np.float64).ravel()
        max_d = max(max_d, float(np.linalg.norm(qa - qb)))
    return max_d


def _fk_closure_stats(robot_model, target_poses, results_list, pos_tol, ori_tol):
    """FK(NumPy) achieved pose vs IK target; successful samples only.

    :param robot_model: RobotModel instance
    :param target_poses: (B, 4, 4) targets used for IK
    :param results_list: One backend's list of IK result dicts
    :param pos_tol: Position tolerance for within_tol count (m)
    :param ori_tol: Orientation tolerance for within_tol count (rad)
    :return: dict with max/mean errors, within_tol count, or None if none succeeded
    """
    idx = [i for i, r in enumerate(results_list) if r.get('success')]
    if not idx:
        return None
    rc.set_backend('numpy')
    Q = np.stack(
        [np.asarray(to_numpy(results_list[i]['q']), dtype=np.float64).ravel() for i in idx],
        axis=0,
    )
    T_tgt = np.asarray(target_poses[idx], dtype=np.float64)
    T_hat = to_numpy(forward_kinematics(robot_model, Q, return_end=True))
    if T_hat.ndim == 2:
        T_hat = T_hat.reshape(1, 4, 4)
    dp = np.linalg.norm(T_hat[:, :3, 3] - T_tgt[:, :3, 3], axis=1)
    dori = np.empty(len(idx), dtype=np.float64)
    for j in range(len(idx)):
        dori[j] = float(np.linalg.norm(rotation_error(T_hat[j, :3, :3], T_tgt[j, :3, :3])))
    within_tol = int(np.sum((dp < pos_tol) & (dori < ori_tol)))
    return {
        'max_dp': float(np.max(dp)),
        'mean_dp': float(np.mean(dp)),
        'max_dori': float(np.max(dori)),
        'mean_dori': float(np.mean(dori)),
        'n': len(idx),
        'within_tol': within_tol,
    }


def _print_closure_check_03a_style(r_np, r_torch, r_cpp, cpp_available, *, target_index, num_batch):
    """Print IK residuals and joint vectors like ``03a_demo_ik.py`` (single-target layout).

    :param r_np: NumPy IK result dict for one target
    :param r_torch: Torch IK result dict for one target
    :param r_cpp: C++ IK result dict or None
    :param cpp_available: Whether C++ backend was run
    :param target_index: 0-based index printed for context
    :param num_batch: Batch size B
    """
    beauty_print(
        f"Closure check — IK report (same layout as 03a_demo_ik.py), target {target_index + 1}/{num_batch}"
    )
    q_np = to_numpy(r_np['q'])
    q_torch = to_numpy(r_torch['q'])
    q_cpp = to_numpy(r_cpp['q']) if cpp_available and r_cpp is not None else None

    beauty_print("IK Solution:")
    if cpp_available and r_cpp is not None:
        print(
            f"  Success:  NumPy={r_np['success']}, Torch={r_torch['success']}, C++={r_cpp['success']}"
        )
        print(
            f"  Iterations:  NumPy={r_np['iters']}, Torch={r_torch['iters']}, C++={r_cpp['iters']}"
        )
        print(
            f"  Position Error:  NumPy={r_np['pos_err']:.6e} m, Torch={r_torch['pos_err']:.6e} m, "
            f"C++={r_cpp['pos_err']:.6e} m"
        )
        print(
            f"  Orientation Error:  NumPy={r_np['ori_err']:.6e} rad, Torch={r_torch['ori_err']:.6e} rad, "
            f"C++={r_cpp['ori_err']:.6e} rad"
        )
        en, et, ec = r_np.get('err_norm'), r_torch.get('err_norm'), r_cpp.get('err_norm')
        te_parts = []
        if en is not None:
            te_parts.append(f"NumPy={en:.6e}")
        if et is not None:
            te_parts.append(f"Torch={et:.6e}")
        if ec is not None:
            te_parts.append(f"C++={ec:.6e}")
        if te_parts:
            print(f"  Total Error:  {', '.join(te_parts)}")
    else:
        print(f"  Success:  NumPy={r_np['success']}, Torch={r_torch['success']}, C++=N/A")
        print(f"  Iterations:  NumPy={r_np['iters']}, Torch={r_torch['iters']}, C++=N/A")
        print(
            f"  Position Error:  NumPy={r_np['pos_err']:.6e} m, Torch={r_torch['pos_err']:.6e} m, C++=N/A"
        )
        print(
            f"  Orientation Error:  NumPy={r_np['ori_err']:.6e} rad, Torch={r_torch['ori_err']:.6e} rad, "
            f"C++=N/A"
        )
        en, et = r_np.get('err_norm'), r_torch.get('err_norm')
        te_parts = []
        if en is not None:
            te_parts.append(f"NumPy={en:.6e}")
        if et is not None:
            te_parts.append(f"Torch={et:.6e}")
        if te_parts:
            print(f"  Total Error:  {', '.join(te_parts)}")

    beauty_print("Solved Joint Angles (radians):")
    print(f"  NumPy:  {beauty_print_array(q_np)}")
    print(f"  Torch:  {beauty_print_array(q_torch)}")
    if q_cpp is not None:
        print(f"  C++:    {beauty_print_array(q_cpp)}")
        print(
            f"  np vs torch: {np.linalg.norm(q_np - q_torch):.6e}   "
            f"np vs cpp: {np.linalg.norm(q_np - q_cpp):.6e}"
        )
    else:
        print(f"  np vs torch: {np.linalg.norm(q_np - q_torch):.6e}")

    beauty_print("Solved Joint Angles (degrees):")
    print(f"  NumPy:  {beauty_print_array(np.rad2deg(q_np))}")
    print(f"  Torch:  {beauty_print_array(np.rad2deg(q_torch))}")
    if q_cpp is not None:
        print(f"  C++:    {beauty_print_array(np.rad2deg(q_cpp))}")


def _print_fk_closure_summary(label, stats, pos_tol, ori_tol):
    """Print FK vs target closure; counts samples within IK tolerances.

    :param label: Backend label for the line
    :param stats: dict from _fk_closure_stats or None
    :param pos_tol: Position tolerance (m)
    :param ori_tol: Orientation tolerance (rad)
    """
    if stats is None:
        print(f"  {label}: (no successful IK in batch)")
        return
    print(
        f"  {label}: max|Δp|={stats['max_dp']:.3e} mean|Δp|={stats['mean_dp']:.3e}  "
        f"max|Δori|={stats['max_dori']:.3e} mean|Δori|={stats['mean_dori']:.3e}  "
        f"within tol (pos<{pos_tol}, ori<{ori_tol}): {stats['within_tol']}/{stats['n']}"
    )


def main(args):
    robot_model = RobotModel(str(args.model_path), base_link=args.base_link, end_link=args.end_link)
    if args.verbose:
        robot_model.summary(show_chain=True)
        robot_model.print_tree(show_fixed=True)

    if args.target_source == 'random-pose':
        num_batch = args.batch_size
        beauty_print(
            f"Batch IK: B={num_batch} targets from random_pose_batch "
            f"(seed={args.seed}, pose_scale={args.pose_scale})"
        )
        rc.set_backend('numpy')
        q_full = robot_model.random_q_batch(num_batch, seed=args.seed, scale=args.pose_scale)
        target_poses = to_numpy(
            forward_kinematics(robot_model, q_full, return_end=True)
        )
        if target_poses.ndim == 2:
            target_poses = target_poses[np.newaxis, ...]
        ci = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
        joint_configs = [np.asarray(q_full[i, ci], dtype=np.float64) for i in range(num_batch)]
    else:
        joint_configs = args.joint_angles
        num_batch = len(joint_configs)
        q_batch = np.asarray(joint_configs, dtype=np.float64)
        beauty_print(f"Batch IK: B={num_batch} target pose(s) from FK of listed joint configs")
        target_poses = _targets_from_joint_batch(robot_model, q_batch)

    ik_kwargs = dict(
        q0=None,
        method=args.method,
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
        use_analytic_jacobian=True,
        num_initial_guesses=args.num_inits,
        initial_guess_strategy=args.init_strategy,
        initial_guess_scale=args.init_scale,
        random_seed=args.seed,
    )

    results_np = compute_ik_batch(robot_model, 'numpy', target_poses, ik_kwargs)
    results_torch = compute_ik_batch(robot_model, 'torch', target_poses, ik_kwargs)
    run_cpp = str(args.method).lower() == 'dls'
    results_cpp = (
        compute_ik_batch(robot_model, 'cpp', target_poses, ik_kwargs)
        if run_cpp else None
    )

    rn = results_np['results']
    rt = results_torch['results']
    r_cpp_list = results_cpp['results'] if results_cpp is not None else None

    pos_tol = ik_kwargs['pos_tol']
    ori_tol = ik_kwargs['ori_tol']

    _print_closure_check_03a_style(
        rn[0],
        rt[0],
        r_cpp_list[0] if r_cpp_list is not None else None,
        r_cpp_list is not None,
        target_index=0,
        num_batch=num_batch,
    )

    beauty_print("FK closure (NumPy FK(q_ik) vs T_target, successful IK samples only)")
    _print_fk_closure_summary(
        "q from NumPy IK",
        _fk_closure_stats(robot_model, target_poses, rn, pos_tol, ori_tol),
        pos_tol,
        ori_tol,
    )
    _print_fk_closure_summary(
        "q from Torch IK",
        _fk_closure_stats(robot_model, target_poses, rt, pos_tol, ori_tol),
        pos_tol,
        ori_tol,
    )
    if r_cpp_list is not None:
        _print_fk_closure_summary(
            "q from C++ IK",
            _fk_closure_stats(robot_model, target_poses, r_cpp_list, pos_tol, ori_tol),
            pos_tol,
            ori_tol,
        )

    succ_np = sum(1 for r in rn if r['success'])
    succ_t = sum(1 for r in rt if r['success'])
    succ_c = sum(1 for r in r_cpp_list if r['success']) if r_cpp_list is not None else None
    beauty_print("Batch success counts:")
    if r_cpp_list is not None:
        print(f"  NumPy: {succ_np}/{num_batch}   Torch: {succ_t}/{num_batch}   C++: {succ_c}/{num_batch}")
    else:
        print(f"  NumPy: {succ_np}/{num_batch}   Torch: {succ_t}/{num_batch}   C++: (skipped, cpp requires method=dls)")

    qdiff_nt = _max_q_diff(rn, rt)
    beauty_print("Solved q agreement (max L2 over batch where both succeeded):")
    print(f"  np vs torch: {qdiff_nt:.6e}", end='')
    if r_cpp_list is not None:
        qdiff_nc = _max_q_diff(rn, r_cpp_list)
        print(f"   np vs cpp: {qdiff_nc:.6e}")
    else:
        print()

    beauty_print("Batch computation time (wall-clock, full batch):")
    print(f"  NumPy:  {results_np['time']:.6f} seconds")
    print(f"  Torch:  {results_torch['time']:.6f} seconds")
    if results_cpp is not None:
        print(f"  C++:    {results_cpp['time']:.6f} seconds")
    if num_batch > 0:
        print(f"  NumPy avg per target: {results_np['time'] / num_batch:.6f} s")
        print(f"  Torch avg per target: {results_torch['time'] / num_batch:.6f} s")
        if results_cpp is not None:
            print(f"  C++ avg per target:   {results_cpp['time'] / num_batch:.6f} s")

    beauty_print("Computation Time:")
    print(f"  NumPy:  {results_np['time'] * 1000:.4f} ms")
    print(f"  Torch:  {results_torch['time'] * 1000:.4f} ms")
    if results_cpp is not None:
        print(f"  C++:    {results_cpp['time'] * 1000:.4f} ms")
        tnp = max(results_np['time'], 1e-15)
        print(f"  torch/np: {results_torch['time'] / tnp:.2f}x   cpp/np: {results_cpp['time'] / tnp:.2f}x")
    else:
        tnp = max(results_np['time'], 1e-15)
        print(f"  torch/np: {results_torch['time'] / tnp:.2f}x")

    if args.show_details:
        beauty_print("Per-target summary (NumPy backend)", type="module", centered=True)
        for i in range(num_batch):
            r = rn[i]
            q_ref = joint_configs[i]
            beauty_print(f"Configuration {i + 1}:", type="info")
            print(f"  Reference joint angles (FK source): {beauty_print_array(np.array(q_ref))}")
            print(f"  IK success: {r['success']}")
            if r['success']:
                print(f"  Solved q: {beauty_print_array(np.asarray(to_numpy(r['q']), dtype=np.float64))}")
                print(f"  Position error: {r.get('pos_err', 0.0):.6e} m")
                print(f"  Orientation error: {r.get('ori_err', 0.0):.6e} rad")
                print(f"  Iterations: {r.get('iters', 0)}")
            if args.show_targets:
                Ti = to_numpy(target_poses[i])
                print(f"  Target T[:3,3] (m): {beauty_print_array(Ti[:3, 3])}")


if __name__ == "__main__":
    from synriard import get_model_path

    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(
        description="Inverse kinematics batch demo — NumPy / Torch / C++ timing (see also 03a_demo_ik.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python 03b_demo_ik_parallel.py

        python 03b_demo_ik_parallel.py --target-source fk-joints --joint-angles \\
            0.1 0.2 -0.3 0.0 0.5 -0.2 \\
            0.2 0.3 -0.4 0.1 0.6 -0.3

        python 03b_demo_ik_parallel.py --batch-size 8 --pose-scale 0.8 --seed 0

        python 03b_demo_ik_parallel.py --show-details --show-targets
        """
    )
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
    parser.add_argument(
        '--target-source',
        type=str,
        default='random-pose',
        choices=['random-pose', 'fk-joints'],
        help='How batch targets are built (default: random-pose via random_pose_batch)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Batch size when --target-source random-pose (default: 5)',
    )
    parser.add_argument(
        '--pose-scale',
        type=float,
        default=0.8,
        help='Joint limit scale for random_q_batch inside random_pose_batch (0..1, default: 0.8)',
    )
    parser.add_argument('--joint-angles', type=float, nargs='+',
                        default=[0.1, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.4, 0.2, -0.3, 0.0, 0.5, -0.2,
                                 0.1, 0.5, -0.3, 0.0, 0.0, 0.2,
                                 0.5, 0.1, -0.9, 0.0, 0.2, -0.2,
                                 0.1, 0.2, -0.3, 0.7, 0.5, -0.2],
                        help='Used when --target-source fk-joints: joint angles (rad), layout as 01b_demo_fk_parallel.py')
    parser.add_argument('--num-joints', type=int, default=6,
                        help='Number of joints per configuration (default: 6)')
    parser.add_argument('--num-inits', type=int, default=5,
                        help='Number of initial guesses per target (default: 1)')
    parser.add_argument('--init-strategy', type=str, default='random',
                        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
                        help='Initial guess strategy (default: random)')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Scale for joint limits when generating guesses (default: 1.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for initial guesses (default: None)')
    parser.add_argument('--method', type=str, default='dls',
                        choices=['dls', 'pinv', 'transpose'],
                        help='IK method (default: dls); C++ backend requires dls')
    parser.add_argument('--verbose', action='store_true',
                        help='Show robot model summary and tree')
    parser.add_argument('--show-details', action='store_true',
                        help='Show per-target NumPy IK summary')
    parser.add_argument('--show-targets', action='store_true',
                        help='With --show-details, also print target position from FK')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch', 'cpp'],
                        help='Legacy option (ignored — NumPy, Torch, and C++ are all tested)')
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.target_source == 'fk-joints':
        num_joints = args.num_joints
        joint_angles_flat = args.joint_angles
        if len(joint_angles_flat) % num_joints != 0:
            raise ValueError(
                f"Total number of joint angles ({len(joint_angles_flat)}) must be divisible by "
                f"num-joints ({num_joints}) when using --target-source fk-joints"
            )
        n_batch = len(joint_angles_flat) // num_joints
        args.joint_angles = [
            joint_angles_flat[i * num_joints:(i + 1) * num_joints]
            for i in range(n_batch)
        ]
    else:
        args.joint_angles = None

    main(args)
