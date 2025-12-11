"""Bimanual Inverse Kinematics validation and comparison with Pytorch Kinematics

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
from robocore.kinematics.bimanual import bimanual_inverse_kinematics, bimanual_forward_kinematics
from robocore.utils.beauty_logger import beauty_print, beauty_print_array
from robocore.utils.backend import to_numpy
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_quaternion


def main(args):
    # Load models
    model_path = str(args.model_path)
    
    # PyTorch Kinematics - build chains for both arms
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    chain_left = pk.build_serial_chain_from_urdf(urdf_bytes, args.left_end_link, root_link_name=args.left_base_link)
    chain_right = pk.build_serial_chain_from_urdf(urdf_bytes, args.right_end_link, root_link_name=args.right_base_link)
    n_dof_left = len(chain_left.get_joint_parameter_names())
    n_dof_right = len(chain_right.get_joint_parameter_names())
    
    # RoboCore
    left_model = RobotModel(model_path, base_link=args.left_base_link, end_link=args.left_end_link)
    right_model = RobotModel(model_path, base_link=args.right_base_link, end_link=args.right_end_link)
    rc.set_backend('torch', device=args.device)

    beauty_print(f"Bimanual Inverse Kinematics Comparison: PyTorch Kinematics vs RoboCore", type="module")
    beauty_print(f"Left arm: {n_dof_left} DOF, Right arm: {n_dof_right} DOF", type="info")
    beauty_print(f"Coordination mode: {args.coordination}", type="info")
    beauty_print(f"PyTorch device: {args.device}", type="info")
    beauty_print("Note: PyTorch Kinematics solves each arm independently. RoboCore supports coordination modes.", type="info")

    device = torch.device(args.device)
    dtype = torch.float64
    chain_left = chain_left.to(dtype=dtype, device=device)
    chain_right = chain_right.to(dtype=dtype, device=device)
    joint_limits_left = torch.tensor([[js.limit_lower, js.limit_upper] for js in left_model._chain_actuated], dtype=dtype, device=device)
    joint_limits_right = torch.tensor([[js.limit_lower, js.limit_upper] for js in right_model._chain_actuated], dtype=dtype, device=device)

    # Build target poses from input
    joint_configs_left = left_model.random_q(seed=args.seed, scale=args.scale)
    joint_configs_right = right_model.random_q(seed=args.seed, scale=args.scale)
    fk_result = bimanual_forward_kinematics(
        left_model, right_model,
        joint_configs_left, joint_configs_right,
        return_end=True, mode='indep'
    )
    target_left = to_numpy(fk_result['left'])
    target_right = to_numpy(fk_result['right'])
    beauty_print("[1] Inverse Kinematics Computation", type="module", centered=False)

    # Solve IK with PyTorch Kinematics (independent for each arm)
    ik_solver_pk_left = PseudoInverseIK(
        chain_left,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        max_iterations=args.max_iters,
        lr=args.step_size,
        regularlization=args.damping,
        num_retries=args.num_retries,
        joint_limits=joint_limits_left,
    )
    
    ik_solver_pk_right = PseudoInverseIK(
        chain_right,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        max_iterations=args.max_iters,
        lr=args.step_size,
        regularlization=args.damping,
        num_retries=args.num_retries,
        joint_limits=joint_limits_right,
    )
    
    target_transform_left = Transform3d(matrix=torch.tensor(target_left, dtype=dtype, device=device))
    target_transform_right = Transform3d(matrix=torch.tensor(target_right, dtype=dtype, device=device))
    
    sol_pk_left = ik_solver_pk_left.solve(target_transform_left)
    sol_pk_right = ik_solver_pk_right.solve(target_transform_right)
    
    # Select best retry: first converged, or first one if none converged
    converged_mask_left = sol_pk_left.converged[0, :].cpu().numpy()
    converged_mask_right = sol_pk_right.converged[0, :].cpu().numpy()
    best_retry_idx_left = np.where(converged_mask_left)[0][0] if np.any(converged_mask_left) else 0
    best_retry_idx_right = np.where(converged_mask_right)[0][0] if np.any(converged_mask_right) else 0

    q_pk_left = sol_pk_left.solutions[0, best_retry_idx_left, :].cpu().numpy()
    q_pk_right = sol_pk_right.solutions[0, best_retry_idx_right, :].cpu().numpy()
    converged_pk_left = sol_pk_left.converged[0, best_retry_idx_left].item()
    converged_pk_right = sol_pk_right.converged[0, best_retry_idx_right].item()
    
    ik_result_pk = {
        'success_left': converged_pk_left,
        'success_right': converged_pk_right,
        'q_left': q_pk_left,
        'q_right': q_pk_right,
        'iters_left': sol_pk_left.iterations,
        'iters_right': sol_pk_right.iterations,
        'pos_err_left': sol_pk_left.err_pos[0, best_retry_idx_left].item(),
        'pos_err_right': sol_pk_right.err_pos[0, best_retry_idx_right].item(),
        'ori_err_left': sol_pk_left.err_rot[0, best_retry_idx_left].item(),
        'ori_err_right': sol_pk_right.err_rot[0, best_retry_idx_right].item(),
    }

    # Solve IK with RoboCore
    ik_result_rc = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=target_left,
        target_right=target_right,
        method='dls',
        coordination=args.coordination,
        num_initial_guesses=args.num_retries,
        initial_guess_strategy='random',
        initial_guess_scale=1.0,
        random_seed=args.seed,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        max_iters=args.max_iters,
        torch_device=device,
        torch_dtype=dtype,
        use_analytic_jacobian=True,
    )

    beauty_print(f"IK Solution (PyTorch Kinematics):")
    print(f"  Success Left: {ik_result_pk['success_left']}")
    print(f"  Success Right: {ik_result_pk['success_right']}")
    print(f"  Iterations Left: {ik_result_pk['iters_left']}")
    print(f"  Iterations Right: {ik_result_pk['iters_right']}")
    print(f"  Position Error Left: {ik_result_pk['pos_err_left']:.6e} m")
    print(f"  Position Error Right: {ik_result_pk['pos_err_right']:.6e} m")
    print(f"  Orientation Error Left: {ik_result_pk['ori_err_left']:.6e} rad")
    print(f"  Orientation Error Right: {ik_result_pk['ori_err_right']:.6e} rad")

    # Extract error information from res_left and res_right
    res_left = ik_result_rc['res_left']
    res_right = ik_result_rc['res_right']
    
    # Handle case where res_left/res_right might be dict or list
    if isinstance(res_left, list) and len(res_left) > 0:
        res_left = res_left[0]
    if isinstance(res_right, list) and len(res_right) > 0:
        res_right = res_right[0]
    
    # Get iters (use max of both arms if available)
    iters_left = res_left['iters'] if isinstance(res_left, dict) else 0
    iters_right = res_right['iters'] if isinstance(res_right, dict) else 0
    iters_rc = max(iters_left, iters_right)
    
    pos_err_left_rc = res_left['pos_err'] if isinstance(res_left, dict) else 0.0
    pos_err_right_rc = res_right['pos_err'] if isinstance(res_right, dict) else 0.0
    ori_err_left_rc = res_left['ori_err'] if isinstance(res_left, dict) else 0.0
    ori_err_right_rc = res_right['ori_err'] if isinstance(res_right, dict) else 0.0

    beauty_print(f"IK Solution (RoboCore):")
    print(f"  Success Left: {ik_result_rc['success_left']}")
    print(f"  Success Right: {ik_result_rc['success_right']}")
    print(f"  Iterations: {iters_rc}")
    print(f"  Position Error Left: {pos_err_left_rc:.6e} m")
    print(f"  Position Error Right: {pos_err_right_rc:.6e} m")
    print(f"  Orientation Error Left: {ori_err_left_rc:.6e} rad")
    print(f"  Orientation Error Right: {ori_err_right_rc:.6e} rad")

    # Compare solutions
    q_pk_left = ik_result_pk['q_left']
    q_pk_right = ik_result_pk['q_right']
    q_rc_left = ik_result_rc['q_left']
    q_rc_right = ik_result_rc['q_right']
    
    q_diff_left = q_pk_left - q_rc_left
    q_diff_right = q_pk_right - q_rc_right

    beauty_print(f"Solved Joint Angles - Left (PyTorch Kinematics, radians):")
    print(f"  q_left = {beauty_print_array(q_pk_left)}")
    beauty_print(f"Solved Joint Angles - Left (RoboCore, radians):")
    print(f"  q_left = {beauty_print_array(q_rc_left)}")
    
    beauty_print(f"Solved Joint Angles - Right (PyTorch Kinematics, radians):")
    print(f"  q_right = {beauty_print_array(q_pk_right)}")
    beauty_print(f"Solved Joint Angles - Right (RoboCore, radians):")
    print(f"  q_right = {beauty_print_array(q_rc_right)}")

    beauty_print("Joint Angle Comparison - Left (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff_left)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff_left):.6e} rad")
    
    beauty_print("Joint Angle Comparison - Right (PyTorch Kinematics vs RoboCore):")
    beauty_print(f"  Max difference:        {np.max(np.abs(q_diff_right)):.6e} rad")
    beauty_print(f"  Euclidean norm:        {np.linalg.norm(q_diff_right):.6e} rad")

    # Performance comparison
    beauty_print("[2] Performance comparison", type="module", centered=False)
    n_runs = 50

    def benchmark_pk():
        ik_solver_pk_left = PseudoInverseIK(
            chain_left,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_left,
        )
        ik_solver_pk_right = PseudoInverseIK(
            chain_right,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_right,
        )
        target_transform_left = Transform3d(matrix=torch.tensor(target_left, dtype=dtype, device=device))
        target_transform_right = Transform3d(matrix=torch.tensor(target_right, dtype=dtype, device=device))
        sol_left = ik_solver_pk_left.solve(target_transform_left)
        sol_right = ik_solver_pk_right.solve(target_transform_right)
        return {
            'q_left': sol_left.solutions[0, 0, :].cpu().numpy(),
            'q_right': sol_right.solutions[0, 0, :].cpu().numpy(),
        }

    def benchmark_rc():
        return bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=target_left,
            target_right=target_right,
            method='dls',
            coordination=args.coordination,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            max_iters=args.max_iters,
            torch_device=device,
            torch_dtype=dtype,
            use_analytic_jacobian=True,
        )

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
    success_pk_left = []
    success_pk_right = []
    success_rc_left = []
    success_rc_right = []
    pos_errs_pk_left = []
    pos_errs_pk_right = []
    pos_errs_rc_left = []
    pos_errs_rc_right = []
    ori_errs_pk_left = []
    ori_errs_pk_right = []
    ori_errs_rc_left = []
    ori_errs_rc_right = []

    for i in range(args.samples):
        # Generate random target poses
        joint_configs_left_rand = left_model.random_q(seed=args.seed, scale=args.scale)
        joint_configs_right_rand = right_model.random_q(seed=args.seed, scale=args.scale)
        fk_result_rand = bimanual_forward_kinematics(
            left_model, right_model,
            joint_configs_left_rand, joint_configs_right_rand,
            return_end=True, mode='indep',
        )
        target_left_rand = to_numpy(fk_result_rand['left'])
        target_right_rand = to_numpy(fk_result_rand['right'])

        # PyTorch Kinematics IK with multiple initial guesses
        ik_solver_pk_left_rand = PseudoInverseIK(
            chain_left,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_left,
        )
        ik_solver_pk_right_rand = PseudoInverseIK(
            chain_right,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            num_retries=args.num_retries,
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_right,
        )
        target_transform_left_rand = Transform3d(matrix=torch.tensor(target_left_rand, dtype=dtype, device=device))
        target_transform_right_rand = Transform3d(matrix=torch.tensor(target_right_rand, dtype=dtype, device=device))
        
        sol_pk_left_rand = ik_solver_pk_left_rand.solve(target_transform_left_rand)
        sol_pk_right_rand = ik_solver_pk_right_rand.solve(target_transform_right_rand)

        # Select best retry: first converged, or first one if none converged
        converged_mask_left = sol_pk_left_rand.converged[0, :].cpu().numpy()
        converged_mask_right = sol_pk_right_rand.converged[0, :].cpu().numpy()
        best_retry_idx_left = np.where(converged_mask_left)[0][0] if np.any(converged_mask_left) else 0
        best_retry_idx_right = np.where(converged_mask_right)[0][0] if np.any(converged_mask_right) else 0
        
        ik_pk_rand = {
            'success_left': sol_pk_left_rand.converged[0, best_retry_idx_left].item(),
            'success_right': sol_pk_right_rand.converged[0, best_retry_idx_right].item(),
            'pos_err_left': sol_pk_left_rand.err_pos[0, best_retry_idx_left].item(),
            'pos_err_right': sol_pk_right_rand.err_pos[0, best_retry_idx_right].item(),
            'ori_err_left': sol_pk_left_rand.err_rot[0, best_retry_idx_left].item(),
            'ori_err_right': sol_pk_right_rand.err_rot[0, best_retry_idx_right].item(),
        }

        # RoboCore IK - multiple initial guesses are handled automatically by bimanual_inverse_kinematics()
        ik_rc_rand = bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=target_left_rand,
            target_right=target_right_rand,
            method='dls',
            coordination=args.coordination,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            max_iters=args.max_iters,
            torch_device=device,
            torch_dtype=dtype,
            use_analytic_jacobian=True,
        )

        success_pk_left.append(ik_pk_rand['success_left'])
        success_pk_right.append(ik_pk_rand['success_right'])
        success_rc_left.append(ik_rc_rand['success_left'])
        success_rc_right.append(ik_rc_rand['success_right'])
        pos_errs_pk_left.append(ik_pk_rand['pos_err_left'])
        pos_errs_pk_right.append(ik_pk_rand['pos_err_right'])
        
        # Extract error information from res_left and res_right for RoboCore
        res_left_rand = ik_rc_rand['res_left']
        res_right_rand = ik_rc_rand['res_right']
        if isinstance(res_left_rand, list) and len(res_left_rand) > 0:
            res_left_rand = res_left_rand[0]
        if isinstance(res_right_rand, list) and len(res_right_rand) > 0:
            res_right_rand = res_right_rand[0]

        pos_err_left_rc_rand = res_left_rand['pos_err']
        pos_err_right_rc_rand = res_right_rand['pos_err']
        ori_err_left_rc_rand = res_left_rand['ori_err']
        ori_err_right_rc_rand = res_right_rand['ori_err']
        
        pos_errs_rc_left.append(pos_err_left_rc_rand)
        pos_errs_rc_right.append(pos_err_right_rc_rand)
        ori_errs_pk_left.append(ik_pk_rand['ori_err_left'])
        ori_errs_pk_right.append(ik_pk_rand['ori_err_right'])
        ori_errs_rc_left.append(ori_err_left_rc_rand)
        ori_errs_rc_right.append(ori_err_right_rc_rand)

    beauty_print(f"Success rate - Left arm:")
    beauty_print(f"  PyTorch Kinematics: {np.mean(success_pk_left)*100:.1f}%")
    beauty_print(f"  RoboCore:           {np.mean(success_rc_left)*100:.1f}%")
    
    beauty_print(f"Success rate - Right arm:")
    beauty_print(f"  PyTorch Kinematics: {np.mean(success_pk_right)*100:.1f}%")
    beauty_print(f"  RoboCore:           {np.mean(success_rc_right)*100:.1f}%")

    beauty_print(f"Position error statistics - Left arm (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(pos_errs_pk_left, success_pk_left) if s]):.6e} m")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(pos_errs_rc_left, success_rc_left) if s]):.6e} m")
    
    beauty_print(f"Position error statistics - Right arm (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(pos_errs_pk_right, success_pk_right) if s]):.6e} m")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(pos_errs_rc_right, success_rc_right) if s]):.6e} m")
    
    beauty_print(f"Orientation error statistics - Left arm (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(ori_errs_pk_left, success_pk_left) if s]):.6e} rad")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(ori_errs_rc_left, success_rc_left) if s]):.6e} rad")
    
    beauty_print(f"Orientation error statistics - Right arm (successful cases):")
    beauty_print(f"  PyTorch Kinematics - Mean: {np.mean([e for e, s in zip(ori_errs_pk_right, success_pk_right) if s]):.6e} rad")
    beauty_print(f"  RoboCore           - Mean: {np.mean([e for e, s in zip(ori_errs_rc_right, success_rc_right) if s]):.6e} rad")

    beauty_print("✓ Bimanual inverse kinematics validation complete", type="success")


if __name__ == '__main__':
    import synriard
    # Bessica is a dual-arm robot
    # Note: PyTorch Kinematics requires URDF format
    model_path = synriard.get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="urdf")

    parser = argparse.ArgumentParser(description="Bimanual Inverse Kinematics validation with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to URDF file (default: Bessica-D)')
    parser.add_argument('--left-base-link', type=str, default='base_link', help='Left arm base link name')
    parser.add_argument('--left-end-link', type=str, default='left_arm_link7', help='Left arm end-effector link name')
    parser.add_argument('--right-base-link', type=str, default='base_link', help='Right arm base link name')
    parser.add_argument('--right-end-link', type=str, default='right_arm_link7', help='Right arm end-effector link name')
    parser.add_argument('--target-left', type=float, nargs='+',
                        default=[0.05717, -0.35161, 0.45995, -0.504640, 0.483414, 0.385570, 0.602483],
                        help='Target left end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--target-right', type=float, nargs='+',
                        default=[0.05715, 0.12706, 0.46730, 0.385071, 0.387550, -0.485394, 0.682582],
                        help='Target right end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--coordination', type=str, default='indep',
                        choices=['indep', 'relative_pose', 'relative_pos', 'relative_ori', 'mirror'],
                        help='Coordination mode. Note: PyTorch Kinematics only supports indep mode directly.')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations (for PyTorch Kinematics)')
    parser.add_argument('--pos-tol', type=float, default=1e-4, help='Position tolerance (m, for PyTorch Kinematics)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (rad, for PyTorch Kinematics)')
    parser.add_argument('--damping', type=float, default=1e-9, help='Regularization factor for DLS (lambda^2)')
    parser.add_argument('--step-size', type=float, default=0.2, help='Learning rate')
    parser.add_argument('--device', default='cpu', help='PyTorch device (cpu, cuda)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of initial guesses to try (default: 1)')
    parser.add_argument('--scale', type=float, default=0.8, help='Scaling factor for the joint range (default: 0.8)')
    args = parser.parse_args()

    main(args)
