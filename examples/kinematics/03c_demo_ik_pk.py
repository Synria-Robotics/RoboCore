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
from pytorch_kinematics.chain import SerialChain
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
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_quaternion


def solve_ik_pinocchio(pin_model, pin_data, target_pose, q_init, pin_q_indices, pin_v_indices,
                       end_joint_id, end_frame_id, pos_tol, ori_tol, max_iters, damping, step_size,
                       robot_model=None):
    """Solve IK using Pinocchio CLIK method with joint limits and adaptive damping.
    
    :param pin_model: Pinocchio model
    :param pin_data: Pinocchio data
    :param target_pose: Target pose (4x4 matrix)
    :param q_init: Initial joint configuration (actuated joints only)
    :param pin_q_indices: Mapping from actuated joints to pinocchio q indices
    :param pin_v_indices: Mapping from actuated joints to pinocchio v indices
    :param end_joint_id: End joint ID (if frame not found)
    :param end_frame_id: End frame ID (if available)
    :param pos_tol: Position tolerance
    :param ori_tol: Orientation tolerance
    :param max_iters: Maximum iterations
    :param damping: Base damping factor (will be adapted)
    :param step_size: Step size
    :param robot_model: RobotModel instance for joint limits (optional)
    :return: Dictionary with solution results
    """
    # Map initial configuration to full pinocchio configuration
    q_full = pinocchio.neutral(pin_model).copy()
    for i, pin_idx in enumerate(pin_q_indices):
        if pin_idx < len(q_full):
            q_full[pin_idx] = q_init[i]

    # Store initial configuration for joint limit checking
    q_initial = q_full.copy()

    # Get joint limits if robot_model is provided
    joint_limits = None
    if robot_model is not None:
        chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
        joint_limits = []
        for idx in chain_indices:
            js = robot_model.joint_list[idx]
            joint_limits.append((js.limit_lower, js.limit_upper))

    oMdes = pinocchio.SE3(target_pose[:3, :3], target_pose[:3, 3])

    # Adaptive damping parameters (matching RoboCore)
    min_damping = 1e-4
    max_damping = 5e-2
    base_damping = max(damping, min_damping)  # Use at least min_damping

    for i in range(max_iters):
        pinocchio.forwardKinematics(pin_model, pin_data, q_full)

        if end_frame_id is not None:
            pinocchio.updateFramePlacements(pin_model, pin_data)
            iMd = pin_data.oMf[end_frame_id].actInv(oMdes)
        elif end_joint_id is not None:
            iMd = pin_data.oMi[end_joint_id].actInv(oMdes)
        else:
            raise ValueError("Could not find frame or joint")

        err = pinocchio.log6(iMd).vector

        # Check convergence
        pos_err = norm(err[:3])
        ori_err = norm(err[3:])
        if pos_err < pos_tol and ori_err < ori_tol:
            # Extract actuated joint values
            q_result = np.array([q_full[idx] for idx in pin_q_indices])
            return {
                'success': True,
                'q': q_result,
                'iters': i + 1,
                'pos_err': pos_err,
                'ori_err': ori_err,
            }

        # Compute Jacobian
        if end_frame_id is not None:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = pinocchio.getFrameJacobian(pin_model, pin_data, end_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        elif end_joint_id is not None:
            pinocchio.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = pinocchio.getJointJacobian(pin_model, pin_data, end_joint_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        else:
            raise ValueError("Could not find frame or joint")

        # Extract columns for actuated joints
        J = J_full[:, pin_v_indices] if len(pin_v_indices) > 0 else J_full

        # Compute Jlog6 and transform Jacobian to SE(3) tangent space
        Jlog = pinocchio.Jlog6(iMd.inverse())
        J_se3 = -Jlog @ J

        # Adaptive damping based on condition number and error
        # Compute condition number using SVD
        try:
            U, s, Vt = np.linalg.svd(J_se3, full_matrices=False)
            cond_num = s[0] / (s[-1] + 1e-10)
            # Adaptive damping: larger damping for ill-conditioned Jacobians
            if cond_num > 1e6:
                adaptive_damping = max_damping
            elif cond_num > 1e4:
                adaptive_damping = min_damping + (max_damping - min_damping) * (cond_num - 1e4) / (1e6 - 1e4)
            else:
                adaptive_damping = min_damping
        except:
            adaptive_damping = base_damping

        # Damped least squares with adaptive damping
        JJt = J_se3 @ J_se3.T
        JJt += adaptive_damping * np.eye(6)
        v = -J_se3.T @ solve(JJt, err)

        # Limit step size to prevent large jumps
        v_norm = norm(v)
        max_v_norm = 0.5  # Maximum velocity norm
        if v_norm > max_v_norm:
            v = v * (max_v_norm / v_norm)

        # Map velocity to full configuration space
        v_full = np.zeros(pin_model.nv)
        for i_v, pin_v_idx in enumerate(pin_v_indices):
            if pin_v_idx < len(v_full):
                v_full[pin_v_idx] = v[i_v]

        # Update configuration on manifold
        q_full_new = pinocchio.integrate(pin_model, q_full, v_full * step_size)

        # Apply joint limits if available
        if joint_limits is not None:
            for j, (limit_lower, limit_upper) in enumerate(joint_limits):
                pin_idx = pin_q_indices[j]
                if pin_idx < len(q_full_new):
                    if limit_lower is not None:
                        q_full_new[pin_idx] = max(q_full_new[pin_idx], limit_lower)
                    if limit_upper is not None:
                        q_full_new[pin_idx] = min(q_full_new[pin_idx], limit_upper)

        q_full = q_full_new

    # Final error check
    pinocchio.forwardKinematics(pin_model, pin_data, q_full)
    if end_frame_id is not None:
        pinocchio.updateFramePlacements(pin_model, pin_data)
        iMd = pin_data.oMf[end_frame_id].actInv(oMdes)
    elif end_joint_id is not None:
        iMd = pin_data.oMi[end_joint_id].actInv(oMdes)
    else:
        raise ValueError("Could not find frame or joint")
    err = pinocchio.log6(iMd).vector
    pos_err = norm(err[:3])
    ori_err = norm(err[3:])

    # Extract actuated joint values
    q_result = np.array([q_full[idx] for idx in pin_q_indices])

    return {
        'success': False,
        'q': q_result,
        'iters': max_iters,
        'pos_err': pos_err,
        'ori_err': ori_err,
    }


def main(args):
    # Load models
    model_path = str(args.model_path)
    end_link = args.end_link

    # PyTorch Kinematics
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link)
    n_dof = len(chain.get_joint_parameter_names())

    # RoboCore
    rc_model = RobotModel(model_path, base_link=args.base_link, end_link=end_link)
    rc.set_backend('torch', device=args.device)

    # Pinocchio
    pin_model = pinocchio.buildModelFromUrdf(model_path)
    pin_data = pin_model.createData()

    # Map RoboCore actuated joints to Pinocchio joints
    chain_indices = rc_model._get_joint_indices(rc_model.base_link, rc_model.end_link)
    actuated_joint_names = [rc_model.joint_list[idx].name for idx in chain_indices]
    pin_q_indices = []
    pin_v_indices = []
    for joint_name in actuated_joint_names:
        if pin_model.existJointName(joint_name):
            joint_id = pin_model.getJointId(joint_name)
            if joint_id < len(pin_model.idx_qs):
                pin_q_indices.append(pin_model.idx_qs[joint_id])
            else:
                pin_q_indices.append(joint_id)
            if joint_id < len(pin_model.idx_vs):
                pin_v_indices.append(pin_model.idx_vs[joint_id])
            else:
                pin_v_indices.append(joint_id)

    # Find end link frame ID
    end_frame_id = None
    end_joint_id = None
    if pin_model.existFrame(end_link):
        end_frame_id = pin_model.getFrameId(end_link)
    elif pin_model.existJointName(end_link):
        end_joint_id = pin_model.getJointId(end_link)
    else:
        beauty_print(f"Warning: Could not find {end_link} in pinocchio model. Using last joint.", type="warning")
        end_joint_id = len(pin_model.joints) - 1

    beauty_print(f"Inverse Kinematics Comparison: PyTorch Kinematics vs Pinocchio vs RoboCore ({n_dof} DOF)", type="module")

    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    # Get chain joint limits (already processed in robot_model, None values handled)
    joint_limits = torch.tensor(rc_model.chain_joint_limit, dtype=dtype, device=device)

    # Build target pose from input
    target_pose = rc_model.random_pose(seed=args.seed, scale=args.scale)
    beauty_print("[1] Inverse Kinematics Computation", type="module", centered=False)

    # Solve IK with PyTorch Kinematics using PseudoInverseIK
    # Create IK solver with matching parameters and joint limits
    ik_solver_pk = PseudoInverseIK(
        chain,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        max_iterations=args.max_iters,
        lr=args.step_size,  # learning rate
        regularlization=args.damping,  # lambda^2
        num_retries=args.num_retries,
        joint_limits=joint_limits,
    )
    

    # Create target pose as Transform3d
    target_transform = Transform3d(matrix=torch.tensor(target_pose, dtype=dtype, device=device))  # (1, 4, 4)
    sol_pk = ik_solver_pk.solve(target_transform)
    
    # Extract result from IKSolution
    # sol_pk.solutions shape: (M, num_retries, dof) = (1, 1, n_dof)
    # sol_pk.converged shape: (M, num_retries) = (1, 1)
    q_pk_result = sol_pk.solutions[0, 0, :].cpu().numpy()  # Get first (and only) solution
    converged = sol_pk.converged[0, 0].item()
    pos_err_pk = sol_pk.err_pos[0, 0].item()
    ori_err_pk = sol_pk.err_rot[0, 0].item()

    ik_result_pk = {
        'success': converged,
        'q': q_pk_result,
        'iters': sol_pk.iterations,
        'pos_err': pos_err_pk,
        'ori_err': ori_err_pk,
        'err_norm': np.sqrt(pos_err_pk**2 + ori_err_pk**2)
    }

    # Solve IK with Pinocchio
    if args.q_init is None:
        q_init_pin = np.array([pinocchio.neutral(pin_model)[idx] for idx in pin_q_indices])
    else:
        q_init_pin = np.array(args.q_init)
    ik_result_pin = solve_ik_pinocchio(
        pin_model, pin_data, target_pose, q_init_pin, pin_q_indices, pin_v_indices,
        end_joint_id if end_frame_id is None else None,
        end_frame_id,
        args.pos_tol, args.ori_tol, args.max_iters,
        args.damping, args.step_size,
        robot_model=rc_model
    )

    # Solve IK with RoboCore
    # Note: RoboCore applies joint limits from URDF, while pytorch_kinematics does not.
    # Using default adaptive parameters for better convergence with joint limits.
    ik_result_rc = inverse_kinematics(
        rc_model,
        target_pose,
        method='dls',
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        num_initial_guesses=args.num_retries,
        initial_guess_strategy='random',
        initial_guess_scale=1.0,
        random_seed=args.seed,
    )

    beauty_print(f"IK Solution (PyTorch Kinematics):")
    print(f"  Success: {ik_result_pk['success']}")
    print(f"  Iterations: {ik_result_pk['iters']}")
    print(f"  Position Error: {ik_result_pk['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_pk['ori_err']:.6e} rad")
    print(f"  Total Error: {ik_result_pk['err_norm']:.6e}")

    beauty_print(f"IK Solution (Pinocchio):")
    print(f"  Success: {ik_result_pin['success']}")
    print(f"  Iterations: {ik_result_pin['iters']}")
    print(f"  Position Error: {ik_result_pin['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_pin['ori_err']:.6e} rad")

    beauty_print(f"IK Solution (RoboCore):")
    print(f"  Success: {ik_result_rc['success']}")
    print(f"  Iterations: {ik_result_rc['iters']}")
    print(f"  Position Error: {ik_result_rc['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_rc['ori_err']:.6e} rad")
    if 'err_norm' in ik_result_rc:
        print(f"  Total Error: {ik_result_rc['err_norm']:.6e}")

    # Compare solutions
    q_pk = ik_result_pk['q']
    q_pin = ik_result_pin['q']
    q_rc = ik_result_rc['q']
    q_diff_pk_rc = q_pk - q_rc
    q_diff_pin_rc = q_pin - q_rc
    q_diff_pk_pin = q_pk - q_pin

    beauty_print(f"Solved Joint Angles (PyTorch Kinematics, radians):")
    print(f"  q_ik = {beauty_print_array(q_pk)}")
    beauty_print(f"Solved Joint Angles (Pinocchio, radians):")
    print(f"  q_ik = {beauty_print_array(q_pin)}")
    beauty_print(f"Solved Joint Angles (RoboCore, radians):")
    print(f"  q_ik = {beauty_print_array(q_rc)}")

    beauty_print("Joint Angle Comparison (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff_pk_rc)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff_pk_rc):.6e} rad")

    beauty_print("Joint Angle Comparison (Pinocchio vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff_pin_rc)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff_pin_rc):.6e} rad")

    beauty_print("Joint Angle Comparison (PyTorch Kinematics vs Pinocchio):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff_pk_pin)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff_pk_pin):.6e} rad")

    # Performance comparison
    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 50

    def benchmark_pk():
        ik_solver_pk = PseudoInverseIK(
            chain,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits,
        )
        target_transform = Transform3d(matrix=torch.tensor(target_pose, dtype=dtype, device=device))  # (1, 4, 4)
        sol_pk = ik_solver_pk.solve(target_transform)
        q_result = sol_pk.solutions[0, 0, :].cpu().numpy()
        converged = sol_pk.converged[0, 0].item()
        return {
            'success': converged,
            'q': q_result,
            'iters': sol_pk.iterations,
            'pos_err': sol_pk.err_pos[0, 0].item(),
            'ori_err': sol_pk.err_rot[0, 0].item(),
        }

    def benchmark_pin():
        q_init = np.array([pinocchio.neutral(pin_model)[idx] for idx in pin_q_indices])
        return solve_ik_pinocchio(
            pin_model, pin_data, target_pose, q_init, pin_q_indices, pin_v_indices,
            end_joint_id if end_frame_id is None else None,
            end_frame_id,
            args.pos_tol, args.ori_tol, args.max_iters,
            args.damping, args.step_size,
            robot_model=rc_model
        )

    def benchmark_rc():
        result = inverse_kinematics(
            rc_model, target_pose, q0=None,
            method='dls', max_iters=args.max_iters,
            pos_tol=args.pos_tol, ori_tol=args.ori_tol,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
        )
        return result

    def benchmark(func):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = func()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000

    time_pk = benchmark(benchmark_pk)
    time_pin = benchmark(benchmark_pin)
    time_rc = benchmark(benchmark_rc)

    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"Pinocchio:           {time_pin:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup_pk_rc = time_pk / time_rc if time_rc > 0 else 0
    speedup_pin_rc = time_pin / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup (PK vs RC):  {speedup_pk_rc:.2f}x", type="success" if speedup_pk_rc > 1 else "info")
    beauty_print(f"Speedup (Pin vs RC): {speedup_pin_rc:.2f}x", type="success" if speedup_pin_rc > 1 else "info")

    # Success rate and error comparison across random configurations
    beauty_print(f"[3] Success rate and error comparison across {args.samples} random configurations", type="module", centered=False)
    success_pk = []
    success_pin = []
    success_rc = []
    pos_errs_pk = []
    pos_errs_pin = []
    pos_errs_rc = []
    ori_errs_pk = []
    ori_errs_pin = []
    ori_errs_rc = []
    q_diffs_pk_rc = []
    q_diffs_pin_rc = []
    q_diffs_pk_pin = []

    for i in range(args.samples):
        # Generate random target pose
        target_pose_rand = rc_model.random_pose(seed=args.seed, scale=args.scale)
        # PyTorch Kinematics IK with multiple initial guesses
        ik_solver_pk_rand = PseudoInverseIK(
            chain,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits,
        )
        target_transform_rand = Transform3d(matrix=torch.tensor(target_pose_rand, dtype=dtype, device=device))  # (1, 4, 4)
        sol_pk_rand = ik_solver_pk_rand.solve(target_transform_rand)

        # Select best retry: first converged, or first one if none converged
        # sol_pk_rand.converged shape: [1, num_retries]
        converged_mask = sol_pk_rand.converged[0, :].cpu().numpy()  # [num_retries]
        if np.any(converged_mask):
            best_retry_idx = np.where(converged_mask)[0][0]  # First converged
        else:
            best_retry_idx = 0  # Use first retry if none converged

        q_pk_rand = sol_pk_rand.solutions[0, best_retry_idx, :].cpu().numpy()
        converged_pk_rand = sol_pk_rand.converged[0, best_retry_idx].item()
        ik_pk_rand = {
            'success': converged_pk_rand,
            'q': q_pk_rand,
            'iters': sol_pk_rand.iterations,
            'pos_err': sol_pk_rand.err_pos[0, best_retry_idx].item(),
            'ori_err': sol_pk_rand.err_rot[0, best_retry_idx].item(),
        }

        # Pinocchio IK
        q_init_pin_rand = np.array([pinocchio.neutral(pin_model)[idx] for idx in pin_q_indices])
        ik_pin_rand = solve_ik_pinocchio(
            pin_model, pin_data, target_pose_rand, q_init_pin_rand, pin_q_indices, pin_v_indices,
            end_joint_id if end_frame_id is None else None,
            end_frame_id,
            args.pos_tol, args.ori_tol, args.max_iters,
            args.damping, args.step_size,
            robot_model=rc_model
        )

        # RoboCore IK - multiple initial guesses are handled automatically by inverse_kinematics()
        ik_rc_rand = inverse_kinematics(
            rc_model, target_pose_rand, q0=None,
            method='dls', max_iters=args.max_iters,
            pos_tol=args.pos_tol, ori_tol=args.ori_tol,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
        )

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
            q_diff_rand = ik_pk_rand['q'] - ik_rc_rand['q']
            q_diffs_pk_rc.append(np.linalg.norm(q_diff_rand))
        if ik_pin_rand['success'] and ik_rc_rand['success']:
            q_diff_rand = ik_pin_rand['q'] - ik_rc_rand['q']
            q_diffs_pin_rc.append(np.linalg.norm(q_diff_rand))
        if ik_pk_rand['success'] and ik_pin_rand['success']:
            q_diff_rand = ik_pk_rand['q'] - ik_pin_rand['q']
            q_diffs_pk_pin.append(np.linalg.norm(q_diff_rand))

    beauty_print(f"Success rate:")
    beauty_print(f"  PyTorch Kinematics: {np.mean(success_pk)*100:.1f}%")
    beauty_print(f"  Pinocchio:          {np.mean(success_pin)*100:.1f}%")
    beauty_print(f"  RoboCore:           {np.mean(success_rc)*100:.1f}%")

    beauty_print(f"Position error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(pos_errs_pk, success_pk) if s]):.6e} m")
    beauty_print(f"  Pinocchio          - Mean: {np.mean([e for e, s in zip(pos_errs_pin, success_pin) if s]):.6e} m")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(pos_errs_rc, success_rc) if s]):.6e} m")

    beauty_print(f"Orientation error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(ori_errs_pk, success_pk) if s]):.6e} rad")
    beauty_print(f"  Pinocchio          - Mean: {np.mean([e for e, s in zip(ori_errs_pin, success_pin) if s]):.6e} rad")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(ori_errs_rc, success_rc) if s]):.6e} rad")

    if q_diffs_pk_rc:
        beauty_print(f"Joint angle difference statistics - PK vs RC (both successful):")
        beauty_print(f"  Mean:   {np.mean(q_diffs_pk_rc):.6e} rad")
        beauty_print(f"  Median: {np.median(q_diffs_pk_rc):.6e} rad")
        beauty_print(f"  Max:    {np.max(q_diffs_pk_rc):.6e} rad")
        beauty_print(f"  Min:    {np.min(q_diffs_pk_rc):.6e} rad")

    if q_diffs_pin_rc:
        beauty_print(f"Joint angle difference statistics - Pin vs RC (both successful):")
        beauty_print(f"  Mean:   {np.mean(q_diffs_pin_rc):.6e} rad")
        beauty_print(f"  Median: {np.median(q_diffs_pin_rc):.6e} rad")
        beauty_print(f"  Max:    {np.max(q_diffs_pin_rc):.6e} rad")
        beauty_print(f"  Min:    {np.min(q_diffs_pin_rc):.6e} rad")

    if q_diffs_pk_pin:
        beauty_print(f"Joint angle difference statistics - PK vs Pin (both successful):")
        beauty_print(f"  Mean:   {np.mean(q_diffs_pk_pin):.6e} rad")
        beauty_print(f"  Median: {np.median(q_diffs_pk_pin):.6e} rad")
        beauty_print(f"  Max:    {np.max(q_diffs_pk_pin):.6e} rad")
        beauty_print(f"  Min:    {np.min(q_diffs_pk_pin):.6e} rad")

    beauty_print("✓ Inverse kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Inverse Kinematics validation with Pytorch Kinematics and Pinocchio")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='base_link', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+',
                        default=[0.16993, 0.01740, 0.20530, 0.041461, 0.828399, 0.083471, 0.552331],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--q-init', type=float, nargs='+', default=None,
                        help='Initial guess (if None, uses random)')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-4, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (rad)')
    parser.add_argument('--damping', type=float, default=1e-4, help='Base regularization factor for DLS (will be adapted, default: 1e-4)')
    parser.add_argument('--step-size', type=float, default=0.2, help='Learning rate (default: 0.2 to match PyTorch Kinematics lr=0.2)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=50, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-retries', type=int, default=5, help='Number of initial guesses to try (default: 5)')
    parser.add_argument('--num-configs', type=int, default=100, help='Number of random joint configurations to generate (default: 100)')
    parser.add_argument('--scale', type=float, default=0.8, help='Scaling factor for the joint range (default: 0.5)')
    args = parser.parse_args()

    main(args)
