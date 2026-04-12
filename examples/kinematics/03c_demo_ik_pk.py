"""Inverse Kinematics validation and comparison with Pytorch Kinematics and Pinocchio

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
import argparse
import time
import numpy as np
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.ik import PseudoInverseIK
import pinocchio
from numpy.linalg import norm, solve

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy


def _fmt_mean(values):
    """Format mean of ``values`` or ``n/a`` if empty (avoids numpy empty-slice warnings)."""
    return f"{np.mean(values):.6e}" if values else "n/a"


def _fmt_success_rate(success_flags):
    """Format success fraction * 100 or ``n/a`` if no samples."""
    return f"{np.mean(success_flags) * 100:.1f}%" if success_flags else "n/a"


def pin_q_full_from_chain(pin_model, q_chain, pin_q_indices):
    """Build Pinocchio ``q_full`` from chain coordinates.

    :param q_chain: values in chain order matching ``pin_q_indices``
    :return: full configuration vector
    """
    q_full = pinocchio.neutral(pin_model).copy()
    q_chain = np.asarray(q_chain, dtype=np.float64).ravel()
    for j, pin_idx in enumerate(pin_q_indices):
        if pin_idx < len(q_full):
            q_full[pin_idx] = float(q_chain[j])
    return q_full


def pin_ee_homogeneous(pin_model, pin_data, q_full, end_frame_id, end_joint_id):
    """End-effector 4x4 in Pinocchio world frame after FK at ``q_full``.

    :return: (4, 4) array
    """
    pinocchio.forwardKinematics(pin_model, pin_data, q_full)
    if end_frame_id is not None:
        pinocchio.updateFramePlacements(pin_model, pin_data)
        return np.asarray(pin_data.oMf[end_frame_id].homogeneous, dtype=np.float64)
    return np.asarray(pin_data.oMi[end_joint_id].homogeneous, dtype=np.float64)


def pin_target_pose_from_chain_q(pin_model, pin_data, q_chain, pin_q_indices, end_frame_id, end_joint_id):
    """Pin FK target for IK: same convention as ``solve_ik_pinocchio`` (``oMf`` / ``oMi``).

    RoboCore FK can differ from Pin world/body conventions; Pin IK must chase Pin FK of the same ``q``.

    :param q_chain: chain joint vector aligned with ``pin_q_indices``
    :return: (4, 4) homogeneous matrix
    """
    q_full = pin_q_full_from_chain(pin_model, q_chain, pin_q_indices)
    return pin_ee_homogeneous(pin_model, pin_data, q_full, end_frame_id, end_joint_id)


def rc_ik_solve(rc_model, target_pose, ik_cpp, args, random_seed):
    """RoboCore IK; ``ik_cpp`` uses one ``IKSolverCpp`` instance.

    :param rc_model: RobotModel instance
    :param target_pose: (4, 4) target
    :param ik_cpp: IKSolverCpp or None for ``inverse_kinematics``
    :param args: namespace with max_iters, pos_tol, ori_tol, num_retries
    :param random_seed: seed for initial guesses
    :return: IK result dict
    """
    if ik_cpp is None:
        return inverse_kinematics(
            rc_model,
            target_pose,
            q0=None,
            method='dls',
            max_iters=args.max_iters,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=random_seed,
        )

    from robocore.kinematics.ik_utils.initial_guess import generate_initial_guesses

    tgt = np.asarray(target_pose, dtype=np.float64)
    n_dof = rc_model.num_chain_dof
    guesses = generate_initial_guesses(
        rc_model,
        args.num_retries,
        strategy='random',
        seed=random_seed,
        scale=1.0,
        base_q0=np.zeros(n_dof),
    )
    solve_kw = dict(
        method='dls',
        use_analytic_jacobian=True,
        target_link=None,
        row_mask=None,
        nullspace_gain=0.0,
        joint_centering=True,
        joint_center_gain=0.2,
        joint_center_weights=None,
    )

    def run_once(q_init):
        return ik_cpp.solve(tgt, np.asarray(q_init, dtype=np.float64), **solve_kw)

    if len(guesses) == 1:
        res = dict(run_once(guesses[0]))
    else:
        candidates = [dict(run_once(guesses[i])) for i in range(len(guesses))]
        succ = [c for c in candidates if c.get('success')]
        if succ:
            res = min(succ, key=lambda c: c.get('err_norm', float('inf')))
        else:
            res = min(candidates, key=lambda c: c.get('err_norm', float('inf')))
    res.setdefault('backend', 'cpp')
    return res


def solve_ik_pinocchio(
    pin_model,
    pin_data,
    target_pose,
    q_init,
    pin_q_indices,
    pin_v_indices,
    end_joint_id,
    end_frame_id,
    pos_tol,
    ori_tol,
    max_iters,
    damping,
    step_size,
    robot_model=None,
):
    """Pinocchio CLIK with joint limits and adaptive damping.

    :param target_pose: (4, 4) desired ``oMf`` / ``oMi`` in Pinocchio convention
    :param q_init: actuated joints only, chain order
    :return: result dict with success, q, iters, pos_err, ori_err
    """
    q_full = pin_q_full_from_chain(pin_model, q_init, pin_q_indices)

    joint_limits = None
    if robot_model is not None:
        chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
        joint_limits = [
            (robot_model.joint_list[idx].limit_lower, robot_model.joint_list[idx].limit_upper)
            for idx in chain_indices
        ]

    oMdes = pinocchio.SE3(target_pose[:3, :3], target_pose[:3, 3])
    min_damping, max_damping = 1e-4, 5e-2
    base_damping = max(damping, min_damping)

    def fk_err(qf):
        pinocchio.forwardKinematics(pin_model, pin_data, qf)
        if end_frame_id is not None:
            pinocchio.updateFramePlacements(pin_model, pin_data)
            iMd = pin_data.oMf[end_frame_id].actInv(oMdes)
        else:
            iMd = pin_data.oMi[end_joint_id].actInv(oMdes)
        e = pinocchio.log6(iMd).vector
        return e, iMd

    for _ in range(max_iters):
        err, iMd = fk_err(q_full)
        pos_err, ori_err = norm(err[:3]), norm(err[3:])
        if pos_err < pos_tol and ori_err < ori_tol:
            return {
                'success': True,
                'q': np.array([q_full[idx] for idx in pin_q_indices]),
                'iters': _ + 1,
                'pos_err': pos_err,
                'ori_err': ori_err,
            }

        if end_frame_id is not None:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = pinocchio.getFrameJacobian(
                pin_model, pin_data, end_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        else:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = pinocchio.getJointJacobian(
                pin_model, pin_data, end_joint_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

        J = J_full[:, pin_v_indices] if len(pin_v_indices) > 0 else J_full
        Jlog = pinocchio.Jlog6(iMd.inverse())
        J_se3 = -Jlog @ J

        try:
            _u, s, _vt = np.linalg.svd(J_se3, full_matrices=False)
            cond_num = s[0] / (s[-1] + 1e-10)
            if cond_num > 1e6:
                adaptive_damping = max_damping
            elif cond_num > 1e4:
                adaptive_damping = min_damping + (max_damping - min_damping) * (cond_num - 1e4) / (1e6 - 1e4)
            else:
                adaptive_damping = min_damping
        except Exception:
            adaptive_damping = base_damping

        JJt = J_se3 @ J_se3.T + adaptive_damping * np.eye(6)
        v = -J_se3.T @ solve(JJt, err)
        v_norm = norm(v)
        if v_norm > 0.5:
            v *= 0.5 / v_norm

        v_full = np.zeros(pin_model.nv)
        for i_v, pin_v_idx in enumerate(pin_v_indices):
            if pin_v_idx < len(v_full):
                v_full[pin_v_idx] = v[i_v]

        q_full_new = pinocchio.integrate(pin_model, q_full, v_full * step_size)
        if joint_limits is not None:
            for j, (lo, hi) in enumerate(joint_limits):
                pin_idx = pin_q_indices[j]
                if pin_idx >= len(q_full_new):
                    continue
                if lo is not None:
                    q_full_new[pin_idx] = max(q_full_new[pin_idx], lo)
                if hi is not None:
                    q_full_new[pin_idx] = min(q_full_new[pin_idx], hi)
        q_full = q_full_new

    err, _ = fk_err(q_full)
    pos_err, ori_err = norm(err[:3]), norm(err[3:])
    return {
        'success': False,
        'q': np.array([q_full[idx] for idx in pin_q_indices]),
        'iters': max_iters,
        'pos_err': pos_err,
        'ori_err': ori_err,
    }


def solve_ik_pinocchio_multistart(
    pin_model,
    pin_data,
    target_pose,
    pin_q_indices,
    pin_v_indices,
    end_joint_id,
    end_frame_id,
    pos_tol,
    ori_tol,
    max_iters,
    damping,
    step_size,
    robot_model,
    num_retries,
    random_seed,
    warm_start_q=None,
):
    """Multistart Pin IK; ``warm_start_q`` becomes first guess (see ``generate_initial_guesses``).

    :param warm_start_q: chain-space configuration used as ``base_q0`` for guess generation
    :return: first success or lowest-cost failure
    """
    from robocore.kinematics.ik_utils.initial_guess import generate_initial_guesses

    n_dof = robot_model.num_chain_dof
    base_q0 = np.asarray(warm_start_q, dtype=np.float64).ravel() if warm_start_q is not None else np.zeros(n_dof)
    guesses = generate_initial_guesses(
        robot_model,
        num_retries,
        strategy='random',
        seed=random_seed,
        scale=1.0,
        base_q0=base_q0,
    )
    best, best_cost = None, float('inf')
    for q_init in guesses:
        res = solve_ik_pinocchio(
            pin_model,
            pin_data,
            target_pose,
            q_init,
            pin_q_indices,
            pin_v_indices,
            end_joint_id,
            end_frame_id,
            pos_tol,
            ori_tol,
            max_iters,
            damping,
            step_size,
            robot_model=robot_model,
        )
        if res.get('success'):
            return res
        cost = float(res['pos_err'] ** 2 + res['ori_err'] ** 2)
        if cost < best_cost:
            best_cost, best = cost, res
    return best


def main(args):
    model_path = str(args.model_path)
    end_link = args.end_link

    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name=args.base_link)
    n_dof = len(chain.get_joint_parameter_names())

    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=end_link)
    rc.set_backend(args.backend, device=args.device)

    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()

    chain_indices = rc_model._get_joint_indices(rc_model.base_link, rc_model.end_link)
    actuated_joint_names = [rc_model.joint_list[idx].name for idx in chain_indices]
    pin_q_indices, pin_v_indices = [], []
    for joint_name in actuated_joint_names:
        if pin_model.existJointName(joint_name):
            jid = pin_model.getJointId(joint_name)
            pin_q_indices.append(pin_model.idx_qs[jid] if jid < len(pin_model.idx_qs) else jid)
            pin_v_indices.append(pin_model.idx_vs[jid] if jid < len(pin_model.idx_vs) else jid)

    end_frame_id = end_joint_id = None
    if pin_model.existFrame(end_link):
        end_frame_id = pin_model.getFrameId(end_link)
    elif pin_model.existJointName(end_link):
        end_joint_id = pin_model.getJointId(end_link)
    else:
        beauty_print(f"Warning: Could not find {end_link} in pinocchio model. Using last joint.", type="warning")
        end_joint_id = len(pin_model.joints) - 1
    pin_joint_for_ik = end_joint_id if end_frame_id is None else None

    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    joint_limits = torch.tensor(rc_model.chain_joint_limit, dtype=dtype, device=device)

    ik_cpp = None
    if args.backend == 'cpp':
        from robocore.kinematics.ik_utils.ik_solver_cpp import IKSolverCpp

        ik_cpp = IKSolverCpp(
            rc_model,
            max_iters=args.max_iters,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
        )

    rc_label = 'RoboCore C++' if ik_cpp is not None else f'RoboCore ({args.backend})'
    beauty_print(
        f"Inverse Kinematics Comparison: PyTorch Kinematics vs Pinocchio vs {rc_label} ({n_dof} DOF)",
        type="module",
    )

    ik_solver_pk = PseudoInverseIK(
        chain,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        max_iterations=args.max_iters,
        lr=args.step_size,
        regularlization=args.damping,
        num_retries=args.num_retries,
        joint_limits=joint_limits,
    )

    q_demo = np.asarray(rc_model.random_q(seed=args.seed, scale=args.scale), dtype=np.float64)
    target_pose = np.asarray(to_numpy(forward_kinematics(rc_model, q_demo, return_end=True)), dtype=np.float64)
    target_pose_pin = pin_target_pose_from_chain_q(
        pin_model, pin_data, q_demo, pin_q_indices, end_frame_id, end_joint_id
    )

    def pin_ms(tgt, seed, warm_q):
        return solve_ik_pinocchio_multistart(
            pin_model,
            pin_data,
            tgt,
            pin_q_indices,
            pin_v_indices,
            pin_joint_for_ik,
            end_frame_id,
            args.pos_tol,
            args.ori_tol,
            args.max_iters,
            args.damping,
            args.step_size,
            rc_model,
            args.num_retries,
            seed,
            warm_start_q=warm_q,
        )

    beauty_print("[1] Inverse Kinematics Computation", type="module", centered=False)

    target_transform = Transform3d(matrix=torch.tensor(target_pose, dtype=dtype, device=device))
    sol_pk = ik_solver_pk.solve(target_transform)
    q_pk_result = sol_pk.solutions[0, 0, :].cpu().numpy()
    converged = sol_pk.converged[0, 0].item()
    pos_err_pk = sol_pk.err_pos[0, 0].item()
    ori_err_pk = sol_pk.err_rot[0, 0].item()
    ik_result_pk = {
        'success': converged,
        'q': q_pk_result,
        'iters': sol_pk.iterations,
        'pos_err': pos_err_pk,
        'ori_err': ori_err_pk,
        'err_norm': np.sqrt(pos_err_pk**2 + ori_err_pk**2),
    }

    if args.q_init is None:
        ik_result_pin = pin_ms(target_pose_pin, args.seed, q_demo)
    else:
        ik_result_pin = solve_ik_pinocchio(
            pin_model,
            pin_data,
            target_pose_pin,
            np.asarray(args.q_init, dtype=np.float64),
            pin_q_indices,
            pin_v_indices,
            pin_joint_for_ik,
            end_frame_id,
            args.pos_tol,
            args.ori_tol,
            args.max_iters,
            args.damping,
            args.step_size,
            robot_model=rc_model,
        )

    ik_result_rc = rc_ik_solve(rc_model, target_pose, ik_cpp, args, args.seed)

    beauty_print("IK Solution (PyTorch Kinematics):")
    print(f"  Success: {ik_result_pk['success']}")
    print(f"  Iterations: {ik_result_pk['iters']}")
    print(f"  Position Error: {ik_result_pk['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_pk['ori_err']:.6e} rad")
    print(f"  Total Error: {ik_result_pk['err_norm']:.6e}")

    beauty_print("IK Solution (Pinocchio):")
    print(f"  Success: {ik_result_pin['success']}")
    print(f"  Iterations: {ik_result_pin['iters']}")
    print(f"  Position Error: {ik_result_pin['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_pin['ori_err']:.6e} rad")

    beauty_print(f"IK Solution ({rc_label}):")
    print(f"  Success: {ik_result_rc['success']}")
    print(f"  Iterations: {ik_result_rc['iters']}")
    print(f"  Position Error: {ik_result_rc['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_rc['ori_err']:.6e} rad")
    if 'err_norm' in ik_result_rc:
        print(f"  Total Error: {ik_result_rc['err_norm']:.6e}")

    q_pk, q_pin, q_rc = ik_result_pk['q'], ik_result_pin['q'], ik_result_rc['q']
    beauty_print("Solved Joint Angles (PyTorch Kinematics, radians):")
    print(f"  q_ik = {beauty_print_array(q_pk)}")
    beauty_print("Solved Joint Angles (Pinocchio, radians):")
    print(f"  q_ik = {beauty_print_array(q_pin)}")
    beauty_print(f"Solved Joint Angles ({rc_label}, radians):")
    print(f"  q_ik = {beauty_print_array(q_rc)}")

    beauty_print(f"Joint Angle Comparison (PyTorch Kinematics vs {rc_label}):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_pk - q_rc)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_pk - q_rc):.6e} rad")
    beauty_print(f"Joint Angle Comparison (Pinocchio vs {rc_label}):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_pin - q_rc)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_pin - q_rc):.6e} rad")
    beauty_print("Joint Angle Comparison (PyTorch Kinematics vs Pinocchio):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_pk - q_pin)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_pk - q_pin):.6e} rad")

    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 50

    def benchmark_pk():
        tt = Transform3d(matrix=torch.tensor(target_pose, dtype=dtype, device=device))
        sol = ik_solver_pk.solve(tt)
        return {
            'success': sol.converged[0, 0].item(),
            'q': sol.solutions[0, 0, :].cpu().numpy(),
            'iters': sol.iterations,
            'pos_err': sol.err_pos[0, 0].item(),
            'ori_err': sol.err_rot[0, 0].item(),
        }

    def benchmark_pin():
        # Cold start (zeros + noisy guesses) for timing comparable to PK/RC
        return pin_ms(target_pose_pin, args.seed, None)

    def benchmark_rc():
        return rc_ik_solve(rc_model, target_pose, ik_cpp, args, args.seed)

    def benchmark(func):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000

    time_pk = benchmark(benchmark_pk)
    time_pin = benchmark(benchmark_pin)
    time_rc = benchmark(benchmark_rc)

    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"Pinocchio:           {time_pin:.4f} ms")
    beauty_print(f"{(rc_label + ':'):<22} {time_rc:.4f} ms")
    if time_rc > 0:
        beauty_print(
            f"Speedup (PK vs {rc_label}):  {time_pk / time_rc:.2f}x",
            type="success" if time_pk > time_rc else "info",
        )
        beauty_print(
            f"Speedup (Pin vs {rc_label}): {time_pin / time_rc:.2f}x",
            type="success" if time_pin > time_rc else "info",
        )

    beauty_print(
        f"[3] Success rate and error comparison across {args.samples} random configurations",
        type="module",
        centered=False,
    )
    success_pk, success_pin, success_rc = [], [], []
    pos_errs_pk, pos_errs_pin, pos_errs_rc = [], [], []
    ori_errs_pk, ori_errs_pin, ori_errs_rc = [], [], []
    q_diffs_pk_rc, q_diffs_pin_rc, q_diffs_pk_pin = [], [], []

    for i in range(args.samples):
        q_i = np.asarray(rc_model.random_q(seed=args.seed + i, scale=args.scale), dtype=np.float64)
        tgt_rc = np.asarray(to_numpy(forward_kinematics(rc_model, q_i, return_end=True)), dtype=np.float64)
        tgt_pin = pin_target_pose_from_chain_q(pin_model, pin_data, q_i, pin_q_indices, end_frame_id, end_joint_id)
        sol = ik_solver_pk.solve(Transform3d(matrix=torch.tensor(tgt_rc, dtype=dtype, device=device)))
        mask = sol.converged[0, :].cpu().numpy()
        ri = int(np.where(mask)[0][0]) if np.any(mask) else 0
        ik_pk_rand = {
            'success': sol.converged[0, ri].item(),
            'q': sol.solutions[0, ri, :].cpu().numpy(),
            'iters': sol.iterations,
            'pos_err': sol.err_pos[0, ri].item(),
            'ori_err': sol.err_rot[0, ri].item(),
        }
        ik_pin_rand = pin_ms(tgt_pin, args.seed + i, q_i)
        ik_rc_rand = rc_ik_solve(rc_model, tgt_rc, ik_cpp, args, args.seed + i)

        success_pk.append(ik_pk_rand['success'])
        success_pin.append(ik_pin_rand['success'])
        success_rc.append(ik_rc_rand['success'])
        pos_errs_pk.append(ik_pk_rand['pos_err'])
        pos_errs_pin.append(ik_pin_rand['pos_err'])
        pos_errs_rc.append(ik_rc_rand['pos_err'])
        ori_errs_pk.append(ik_pk_rand['ori_err'])
        ori_errs_pin.append(ik_pin_rand['ori_err'])
        ori_errs_rc.append(ik_rc_rand['ori_err'])

        if ik_pk_rand['success'] and ik_rc_rand['success']:
            q_diffs_pk_rc.append(np.linalg.norm(ik_pk_rand['q'] - ik_rc_rand['q']))
        if ik_pin_rand['success'] and ik_rc_rand['success']:
            q_diffs_pin_rc.append(np.linalg.norm(ik_pin_rand['q'] - ik_rc_rand['q']))
        if ik_pk_rand['success'] and ik_pin_rand['success']:
            q_diffs_pk_pin.append(np.linalg.norm(ik_pk_rand['q'] - ik_pin_rand['q']))

    beauty_print("Success rate:")
    beauty_print(f"  PyTorch Kinematics: {_fmt_success_rate(success_pk)}")
    beauty_print(f"  Pinocchio:          {_fmt_success_rate(success_pin)}")
    beauty_print(f"  {rc_label}: {_fmt_success_rate(success_rc)}")

    pos_ok_pk = [e for e, s in zip(pos_errs_pk, success_pk) if s]
    pos_ok_pin = [e for e, s in zip(pos_errs_pin, success_pin) if s]
    pos_ok_rc = [e for e, s in zip(pos_errs_rc, success_rc) if s]
    ori_ok_pk = [e for e, s in zip(ori_errs_pk, success_pk) if s]
    ori_ok_pin = [e for e, s in zip(ori_errs_pin, success_pin) if s]
    ori_ok_rc = [e for e, s in zip(ori_errs_rc, success_rc) if s]

    beauty_print("Position error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {_fmt_mean(pos_ok_pk)} m")
    beauty_print(f"  Pinocchio          - Mean: {_fmt_mean(pos_ok_pin)} m")
    beauty_print(f"  {rc_label} - Mean: {_fmt_mean(pos_ok_rc)} m")
    beauty_print("Orientation error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {_fmt_mean(ori_ok_pk)} rad")
    beauty_print(f"  Pinocchio          - Mean: {_fmt_mean(ori_ok_pin)} rad")
    beauty_print(f"  {rc_label} - Mean: {_fmt_mean(ori_ok_rc)} rad")

    for label, diffs in (
        (f"PK vs {rc_label}", q_diffs_pk_rc),
        (f"Pin vs {rc_label}", q_diffs_pin_rc),
        ("PK vs Pin", q_diffs_pk_pin),
    ):
        if diffs:
            beauty_print(f"Joint angle difference statistics - {label} (both successful):")
            beauty_print(f"  Mean:   {np.mean(diffs):.6e} rad")
            beauty_print(f"  Median: {np.median(diffs):.6e} rad")
            beauty_print(f"  Max:    {np.max(diffs):.6e} rad")
            beauty_print(f"  Min:    {np.min(diffs):.6e} rad")

    beauty_print("✓ Inverse kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard

    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="IK: PyTorch Kinematics vs Pinocchio vs RoboCore")
    parser.add_argument('--model-path', type=str, default=model_path, help='URDF path')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='link6', help='End-effector link name')
    parser.add_argument(
        '--q-init', type=float, nargs='+', default=None,
        help='Pinocchio initial guess (chain DOF); default multistart uses FK seed q',
    )
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-4, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (rad)')
    parser.add_argument('--damping', type=float, default=1e-4, help='DLS base damping (adapted internally)')
    parser.add_argument('--step-size', type=float, default=0.2, help='CLIK step size')
    parser.add_argument(
        '--backend',
        type=str,
        default='cpp',
        choices=['numpy', 'torch', 'cpp'],
        help='RoboCore backend (default: cpp)',
    )
    parser.add_argument('--device', default='cpu', help='PyTorch device')
    parser.add_argument('--samples', type=int, default=50, help='Random targets in section [3]')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-retries', type=int, default=5, help='Multistart / PK retries')
    parser.add_argument('--scale', type=float, default=0.8, help='Joint sampling scale inside limits')
    main(parser.parse_args())
