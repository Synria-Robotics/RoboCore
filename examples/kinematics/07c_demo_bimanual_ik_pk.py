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

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import robocore as rc
from robocore.modeling import RobotModel
from robocore.kinematics.bimanual import bimanual_inverse_kinematics
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

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64
    chain_left = chain_left.to(dtype=dtype, device=device)
    chain_right = chain_right.to(dtype=dtype, device=device)

    # Build target poses from input
    T_target_left = torch.eye(4, dtype=dtype, device=device)
    T_target_left[:3, 3] = torch.tensor(args.target_left[:3], dtype=dtype, device=device)
    quat_target_left_xyzw = np.array(args.target_left[3:])  # [x, y, z, w]
    R_target_left = torch.tensor(quaternion_to_matrix(quat_target_left_xyzw), dtype=dtype, device=device)
    T_target_left[:3, :3] = R_target_left
    
    T_target_right = torch.eye(4, dtype=dtype, device=device)
    T_target_right[:3, 3] = torch.tensor(args.target_right[:3], dtype=dtype, device=device)
    quat_target_right_xyzw = np.array(args.target_right[3:])  # [x, y, z, w]
    R_target_right = torch.tensor(quaternion_to_matrix(quat_target_right_xyzw), dtype=dtype, device=device)
    T_target_right[:3, :3] = R_target_right
    
    # Generate initial guesses for PyTorch Kinematics (RoboCore uses initial_guess_strategy)
    q0_left = torch.tensor(left_model.random_q(rng=rng), dtype=dtype, device=device)
    q0_right = torch.tensor(right_model.random_q(rng=rng), dtype=dtype, device=device)

    # Get joint limits from RoboCore models
    joint_limits_left_list = []
    for js in left_model._chain_actuated:
        joint_limits_left_list.append([js.limit_lower, js.limit_upper])
    joint_limits_left = torch.tensor(joint_limits_left_list, dtype=dtype, device=device).T  # [2, n_dof_left]
    
    joint_limits_right_list = []
    for js in right_model._chain_actuated:
        joint_limits_right_list.append([js.limit_lower, js.limit_upper])
    joint_limits_right = torch.tensor(joint_limits_right_list, dtype=dtype, device=device).T  # [2, n_dof_right]

    beauty_print("[1] Inverse Kinematics Computation", type="module", centered=False)
    beauty_print(f"Initial Guess - Left (radians):")
    print(f"  q0_left = {beauty_print_array(q0_left.cpu().numpy())}")
    beauty_print(f"Initial Guess - Right (radians):")
    print(f"  q0_right = {beauty_print_array(q0_right.cpu().numpy())}")

    beauty_print(f"Target End-Effector Pose - Left:")
    print(f"  Position: {beauty_print_array(T_target_left[:3, 3].cpu().numpy())}")
    quat_display_left = matrix_to_quaternion(T_target_left[:3, :3].cpu().numpy())
    print(f"  Quaternion (xyzw): {beauty_print_array(quat_display_left, precision=6)}")
    
    beauty_print(f"Target End-Effector Pose - Right:")
    print(f"  Position: {beauty_print_array(T_target_right[:3, 3].cpu().numpy())}")
    quat_display_right = matrix_to_quaternion(T_target_right[:3, :3].cpu().numpy())
    print(f"  Quaternion (xyzw): {beauty_print_array(quat_display_right, precision=6)}")

    # Solve IK with PyTorch Kinematics (independent for each arm)
    ik_solver_pk_left = PseudoInverseIK(
        chain_left,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        retry_configs=q0_left.unsqueeze(0),
        max_iterations=args.max_iters,
        lr=args.step_size,
        regularlization=args.damping,
        joint_limits=joint_limits_left,
    )
    
    ik_solver_pk_right = PseudoInverseIK(
        chain_right,
        pos_tolerance=args.pos_tol,
        rot_tolerance=args.ori_tol,
        retry_configs=q0_right.unsqueeze(0),
        max_iterations=args.max_iters,
        lr=args.step_size,
        regularlization=args.damping,
        joint_limits=joint_limits_right,
    )
    
    target_transform_left = Transform3d(matrix=T_target_left.unsqueeze(0))
    target_transform_right = Transform3d(matrix=T_target_right.unsqueeze(0))
    
    sol_pk_left = ik_solver_pk_left.solve(target_transform_left)
    sol_pk_right = ik_solver_pk_right.solve(target_transform_right)
    
    q_pk_left = sol_pk_left.solutions[0, 0, :].cpu().numpy()
    q_pk_right = sol_pk_right.solutions[0, 0, :].cpu().numpy()
    converged_pk_left = sol_pk_left.converged[0, 0].item()
    converged_pk_right = sol_pk_right.converged[0, 0].item()
    
    ik_result_pk = {
        'success_left': converged_pk_left,
        'success_right': converged_pk_right,
        'q_left': q_pk_left,
        'q_right': q_pk_right,
        'iters_left': sol_pk_left.iterations,
        'iters_right': sol_pk_right.iterations,
        'pos_err_left': sol_pk_left.err_pos[0, 0].item(),
        'pos_err_right': sol_pk_right.err_pos[0, 0].item(),
        'ori_err_left': sol_pk_left.err_rot[0, 0].item(),
        'ori_err_right': sol_pk_right.err_rot[0, 0].item(),
    }

    # Solve IK with RoboCore
    T_target_left_np = T_target_left.cpu().numpy()
    T_target_right_np = T_target_right.cpu().numpy()
    
    ik_result_rc = bimanual_inverse_kinematics(
        left_model, right_model,
        target_left=T_target_left_np,
        target_right=T_target_right_np,
        method='dls',
        coordination=args.coordination,
        torch_device=device,
        torch_dtype=dtype,
        use_analytic_jacobian=True,
        num_initial_guesses=args.num_retries,
        initial_guess_strategy='random',
        initial_guess_scale=1.0,
        random_seed=args.seed,
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
    res_left = ik_result_rc.get('res_left', {})
    res_right = ik_result_rc.get('res_right', {})
    
    # Handle case where res_left/res_right might be dict or list
    if isinstance(res_left, list) and len(res_left) > 0:
        res_left = res_left[0]
    if isinstance(res_right, list) and len(res_right) > 0:
        res_right = res_right[0]
    
    # Get iters (use max of both arms if available)
    iters_left = res_left.get('iters', 0) if isinstance(res_left, dict) else 0
    iters_right = res_right.get('iters', 0) if isinstance(res_right, dict) else 0
    iters_rc = max(iters_left, iters_right)
    
    pos_err_left_rc = res_left.get('pos_err', 0.0) if isinstance(res_left, dict) else 0.0
    pos_err_right_rc = res_right.get('pos_err', 0.0) if isinstance(res_right, dict) else 0.0
    ori_err_left_rc = res_left.get('ori_err', 0.0) if isinstance(res_left, dict) else 0.0
    ori_err_right_rc = res_right.get('ori_err', 0.0) if isinstance(res_right, dict) else 0.0

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
            retry_configs=q0_left.unsqueeze(0),
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_left,
        )
        ik_solver_pk_right = PseudoInverseIK(
            chain_right,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            retry_configs=q0_right.unsqueeze(0),
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_right,
        )
        target_transform_left = Transform3d(matrix=T_target_left.unsqueeze(0))
        target_transform_right = Transform3d(matrix=T_target_right.unsqueeze(0))
        sol_left = ik_solver_pk_left.solve(target_transform_left)
        sol_right = ik_solver_pk_right.solve(target_transform_right)
        return {
            'q_left': sol_left.solutions[0, 0, :].cpu().numpy(),
            'q_right': sol_right.solutions[0, 0, :].cpu().numpy(),
        }

    def benchmark_rc():
        return bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=T_target_left_np,
            target_right=T_target_right_np,
            method='dls',
            coordination=args.coordination,
            torch_device=device,
            torch_dtype=dtype,
            use_analytic_jacobian=True,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
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
        # Generate random target poses using FK
        q_target_left = torch.tensor(left_model.random_q(rng), dtype=dtype, device=device)
        q_target_right = torch.tensor(right_model.random_q(rng), dtype=dtype, device=device)
        
        q_target_left_tensor = q_target_left.unsqueeze(0)
        q_target_right_tensor = q_target_right.unsqueeze(0)
        ret_target_left = chain_left.forward_kinematics(q_target_left_tensor, end_only=False)
        ret_target_right = chain_right.forward_kinematics(q_target_right_tensor, end_only=False)
        tg_target_left = ret_target_left[args.left_end_link]
        tg_target_right = ret_target_right[args.right_end_link]
        T_target_left_rand = tg_target_left.get_matrix()[0]
        T_target_right_rand = tg_target_right.get_matrix()[0]
        T_target_left_rand_np = T_target_left_rand.cpu().numpy()
        T_target_right_rand_np = T_target_right_rand.cpu().numpy()

        # Generate initial guesses
        noise_scale = 0.1
        q_init_left_near = q_target_left + torch.tensor(rng.normal(0, noise_scale, n_dof_left), dtype=dtype, device=device)
        q_init_right_near = q_target_right + torch.tensor(rng.normal(0, noise_scale, n_dof_right), dtype=dtype, device=device)

        # PyTorch Kinematics IK
        ik_solver_pk_left_rand = PseudoInverseIK(
            chain_left,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            retry_configs=q_init_left_near.unsqueeze(0),
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_left,
        )
        ik_solver_pk_right_rand = PseudoInverseIK(
            chain_right,
            pos_tolerance=args.pos_tol,
            rot_tolerance=args.ori_tol,
            retry_configs=q_init_right_near.unsqueeze(0),
            max_iterations=args.max_iters,
            lr=args.step_size,
            regularlization=args.damping,
            joint_limits=joint_limits_right,
        )
        target_transform_left_rand = Transform3d(matrix=T_target_left_rand.unsqueeze(0))
        target_transform_right_rand = Transform3d(matrix=T_target_right_rand.unsqueeze(0))
        
        sol_pk_left_rand = ik_solver_pk_left_rand.solve(target_transform_left_rand)
        sol_pk_right_rand = ik_solver_pk_right_rand.solve(target_transform_right_rand)
        
        ik_pk_rand = {
            'success_left': sol_pk_left_rand.converged[0, 0].item(),
            'success_right': sol_pk_right_rand.converged[0, 0].item(),
            'pos_err_left': sol_pk_left_rand.err_pos[0, 0].item(),
            'pos_err_right': sol_pk_right_rand.err_pos[0, 0].item(),
            'ori_err_left': sol_pk_left_rand.err_rot[0, 0].item(),
            'ori_err_right': sol_pk_right_rand.err_rot[0, 0].item(),
        }

        # RoboCore IK
        ik_rc_rand = bimanual_inverse_kinematics(
            left_model, right_model,
            target_left=T_target_left_rand_np,
            target_right=T_target_right_rand_np,
            method='dls',
            coordination=args.coordination,
            torch_device=device,
            torch_dtype=dtype,
            use_analytic_jacobian=True,
            num_initial_guesses=args.num_retries,
            initial_guess_strategy='random',
            initial_guess_scale=1.0,
            random_seed=args.seed,
        )

        success_pk_left.append(ik_pk_rand['success_left'])
        success_pk_right.append(ik_pk_rand['success_right'])
        success_rc_left.append(ik_rc_rand['success_left'])
        success_rc_right.append(ik_rc_rand['success_right'])
        pos_errs_pk_left.append(ik_pk_rand['pos_err_left'])
        pos_errs_pk_right.append(ik_pk_rand['pos_err_right'])
        
        # Extract error information from res_left and res_right for RoboCore
        res_left_rand = ik_rc_rand.get('res_left', {})
        res_right_rand = ik_rc_rand.get('res_right', {})
        if isinstance(res_left_rand, list) and len(res_left_rand) > 0:
            res_left_rand = res_left_rand[0]
        if isinstance(res_right_rand, list) and len(res_right_rand) > 0:
            res_right_rand = res_right_rand[0]
        
        pos_err_left_rc_rand = res_left_rand.get('pos_err', 0.0) if isinstance(res_left_rand, dict) else 0.0
        pos_err_right_rc_rand = res_right_rand.get('pos_err', 0.0) if isinstance(res_right_rand, dict) else 0.0
        ori_err_left_rc_rand = res_left_rand.get('ori_err', 0.0) if isinstance(res_left_rand, dict) else 0.0
        ori_err_right_rc_rand = res_right_rand.get('ori_err', 0.0) if isinstance(res_right_rand, dict) else 0.0
        
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
    parser.add_argument('--samples', type=int, default=50, help='Number of test configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of initial guesses to try (default: 10)')
    args = parser.parse_args()

    main(args)
