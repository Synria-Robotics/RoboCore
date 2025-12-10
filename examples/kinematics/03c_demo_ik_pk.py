"""Inverse Kinematics validation and comparison with Pytorch Kinematics

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

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_quaternion


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

    beauty_print(f"Inverse Kinematics Comparison: PyTorch Kinematics vs RoboCore ({n_dof} DOF)", type="module")

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64
    chain = chain.to(dtype=dtype, device=device)
    joint_limits = torch.tensor([[js.limit_lower, js.limit_upper] for js in rc_model._chain_actuated], dtype=dtype, device=device)

    # Build target pose from input
    num_configs = args.num_configs
    joint_configs = rc_model.random_q(rng, scale=args.scale)
    target_poses = forward_kinematics(rc_model, joint_configs, return_end=True)
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
    target_transform = Transform3d(matrix=torch.tensor(target_poses, dtype=dtype, device=device))  # (1, 4, 4)
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

    # Solve IK with RoboCore
    # Note: RoboCore applies joint limits from URDF, while pytorch_kinematics does not.
    # Using default adaptive parameters for better convergence with joint limits.
    ik_result_rc = inverse_kinematics(
        rc_model,
        target_poses,
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

    beauty_print(f"IK Solution (RoboCore):")
    print(f"  Success: {ik_result_rc['success']}")
    print(f"  Iterations: {ik_result_rc['iters']}")
    print(f"  Position Error: {ik_result_rc['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result_rc['ori_err']:.6e} rad")
    if 'err_norm' in ik_result_rc:
        print(f"  Total Error: {ik_result_rc['err_norm']:.6e}")

    # Compare solutions
    q_pk = ik_result_pk['q']
    q_rc = ik_result_rc['q']
    q_diff = q_pk - q_rc

    beauty_print(f"Solved Joint Angles (PyTorch Kinematics, radians):")
    print(f"  q_ik = {beauty_print_array(q_pk)}")
    beauty_print(f"Solved Joint Angles (RoboCore, radians):")
    print(f"  q_ik = {beauty_print_array(q_rc)}")

    beauty_print("Joint Angle Comparison (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff):.6e} rad")

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
        target_transform = Transform3d(matrix=torch.tensor(target_poses, dtype=dtype, device=device))  # (1, 4, 4)
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

    def benchmark_rc():
        result = inverse_kinematics(
            rc_model, target_poses, q0=None,
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
    time_rc = benchmark(benchmark_rc)

    beauty_print(f"PyTorch Kinematics:  {time_pk:.4f} ms")
    beauty_print(f"RoboCore:            {time_rc:.4f} ms")
    speedup = time_pk / time_rc if time_rc > 0 else 0
    beauty_print(f"Speedup:             {speedup:.2f}x", type="success" if speedup > 1 else "info")

    # Success rate and error comparison across random configurations
    beauty_print(f"[3] Success rate and error comparison across {args.samples} random configurations", type="module", centered=False)
    success_pk = []
    success_rc = []
    pos_errs_pk = []
    pos_errs_rc = []
    ori_errs_pk = []
    ori_errs_rc = []
    q_diffs = []

    for i in range(args.samples):
        # Generate random target pose
        joint_configs_rand = rc_model.random_q(rng, scale=args.scale)
        target_poses_rand = forward_kinematics(rc_model, joint_configs_rand, return_end=True)
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
        target_transform_rand = Transform3d(matrix=torch.tensor(target_poses_rand, dtype=dtype, device=device))  # (1, 4, 4)
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

        # RoboCore IK - multiple initial guesses are handled automatically by inverse_kinematics()
        ik_rc_rand = inverse_kinematics(
            rc_model, target_poses_rand, q0=None,
            method='dls', max_iters=args.max_iters,
            pos_tol=args.pos_tol, ori_tol=args.ori_tol,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
        )

        success_pk.append(ik_pk_rand['success'])
        success_rc.append(ik_rc_rand['success'])
        pos_errs_pk.append(ik_pk_rand['pos_err'])
        pos_errs_rc.append(ik_rc_rand['pos_err'])
        ori_errs_pk.append(ik_pk_rand['ori_err'])
        ori_errs_rc.append(ik_rc_rand['ori_err'])

        if ik_pk_rand['success'] and ik_rc_rand['success']:
            q_diff_rand = ik_pk_rand['q'] - ik_rc_rand['q']
            q_diffs.append(np.linalg.norm(q_diff_rand))

    beauty_print(f"Success rate:")
    beauty_print(f"  PyTorch Kinematics: {np.mean(success_pk)*100:.1f}%")
    beauty_print(f"  RoboCore:           {np.mean(success_rc)*100:.1f}%")

    beauty_print(f"Position error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(pos_errs_pk, success_pk) if s]):.6e} m")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(pos_errs_rc, success_rc) if s]):.6e} m")

    beauty_print(f"Orientation error statistics (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(ori_errs_pk, success_pk) if s]):.6e} rad")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(ori_errs_rc, success_rc) if s]):.6e} rad")

    if q_diffs:
        beauty_print(f"Joint angle difference statistics (both successful):")
        beauty_print(f"  Mean:   {np.mean(q_diffs):.6e} rad")
        beauty_print(f"  Median: {np.median(q_diffs):.6e} rad")
        beauty_print(f"  Max:    {np.max(q_diffs):.6e} rad")
        beauty_print(f"  Min:    {np.min(q_diffs):.6e} rad")

    beauty_print("✓ Inverse kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")

    parser = argparse.ArgumentParser(description="Inverse Kinematics validation with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--base-link', type=str, default='world', help='Base link name')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+',
                        default=[0.17006, 0.01704, 0.20533, 0.042114, 0.828366, 0.083037, 0.552396],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--q-init', type=float, nargs='+', default=None,
                        help='Initial guess (if None, uses random)')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-4, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (rad)')
    parser.add_argument('--damping', type=float, default=1e-9, help='Regularization factor for DLS (lambda^2, matches PyTorch Kinematics regularlization=1e-9)')
    parser.add_argument('--step-size', type=float, default=0.2, help='Learning rate (default: 0.2 to match PyTorch Kinematics lr=0.2)')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=50, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of initial guesses to try (default: 10)')
    parser.add_argument('--num-configs', type=int, default=100, help='Number of random joint configurations to generate (default: 100)')
    parser.add_argument('--scale', type=float, default=0.8, help='Scaling factor for the joint range (default: 0.5)')
    args = parser.parse_args()

    main(args)
