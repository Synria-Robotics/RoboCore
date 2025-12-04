#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

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

import numpy as np
import argparse
from pathlib import Path
from typing import Optional

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian

from robocore.configs import ConfigManager, get_default_config
from omegaconf import OmegaConf
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
import random
import math
from time import perf_counter
from robocore.transform import get_rotation, rotation_distance


def compute_fk_ik_jacobian(
    robot_model: RobotModel,
    joint_angles: np.ndarray,
    ik_max_iters: int = 100,
    ik_pos_tol: float = 1e-4,
    ik_ori_tol: float = 1e-4,
    verbose: bool = True
):
    """
    Compute FK, IK, and Jacobian for specified joint angles.
    
    Parameters
    ----------
    robot_model : RobotModel
        The robot model
    joint_angles : np.ndarray
        Input joint angles (radians)
    ik_max_iters : int
        Maximum IK iterations
    ik_pos_tol : float
        IK position tolerance
    ik_ori_tol : float
        IK orientation tolerance
    verbose : bool
        Whether to print detailed information
    
    Returns
    -------
    dict
        Dictionary containing FK, IK, and Jacobian results
    """
    results = {}
    
    # ========================================
    # 1. Forward Kinematics
    # ========================================
    if verbose:
        beauty_print("Step 1: Forward Kinematics (FK)", type="module", centered=True)
        print(f"Input Joint Angles (radians):")
        print(f"  q = {beauty_print_array(joint_angles)}")
        print(f"Input Joint Angles (degrees):")
        print(f"  q = {beauty_print_array(np.rad2deg(joint_angles))}")
    
    T_fk = forward_kinematics(robot_model, joint_angles, backend='numpy', return_end=True)
    
    # Extract position and orientation
    position_fk = T_fk[:3, 3]
    rotation_fk = T_fk[:3, :3]
    
    # Convert rotation matrix to Euler angles (XYZ) and Quaternion (xyzw)
    from scipy.spatial.transform import Rotation
    rot_obj = Rotation.from_matrix(rotation_fk)
    euler_fk = rot_obj.as_euler('xyz', degrees=False)
    quat_fk = rot_obj.as_quat()  # scipy uses xyzw order by default
    
    results['fk'] = {
        'transform': T_fk,
        'position': position_fk,
        'rotation': rotation_fk,
        'euler_xyz': euler_fk,
        'quaternion_xyzw': quat_fk  # Quaternion in xyzw order
    }
    
    if verbose:
        print(f"End-Effector Position (m):")
        print(f"  p = {beauty_print_array(position_fk)}")
        print(f"End-Effector Orientation (Euler XYZ, radians):")
        print(f"  rpy = {beauty_print_array(euler_fk)}")
        print(f"End-Effector Orientation (Euler XYZ, degrees):")
        print(f"  rpy = {beauty_print_array(np.rad2deg(euler_fk))}")
        print(f"End-Effector Orientation (Quaternion xyzw):")
        print(f"  quat = {beauty_print_array(quat_fk, precision=6)}")
        # Add note about quaternion sign ambiguity
        quat_neg = -quat_fk
        print(f"  Note: q and -q represent the same rotation")
        print(f"  -quat = {beauty_print_array(quat_neg, precision=6)} (equivalent)")
        print(f"Rotation Matrix:")
        print(beauty_print_array(rotation_fk, precision=6))
        print(f"Homogeneous Transformation Matrix:")
        print(beauty_print_array(T_fk, precision=6))
    
    # ========================================
    # 2. Inverse Kinematics (DLS)
    # ========================================
    if verbose:
        beauty_print("Step 2: Inverse Kinematics (IK) - DLS Solver", type="module", centered=True)
        print(f"Target Pose (from FK):")
        print(f"  Position: {beauty_print_array(position_fk)}")
        print(f"  Euler XYZ: {beauty_print_array(euler_fk)} rad")
        print(f"  Quaternion xyzw: {beauty_print_array(quat_fk, precision=6)}")
    
    # Use zero initial guess for better convergence
    q_init = np.zeros(robot_model.num_dof())
    
    if verbose:
        print(f"Initial Guess (radians):")
        print(f"  q_init = {beauty_print_array(q_init)}")
    
    # Solve IK using DLS method
    ik_result = inverse_kinematics(
        robot_model,
        T_fk,
        q_init,
        backend='numpy',
        method='dls',
        max_iters=ik_max_iters,
        pos_tol=ik_pos_tol,
        ori_tol=ik_ori_tol
    )
    
    results['ik'] = ik_result
    
    if verbose:
        print(f"IK Solution:")
        print(f"  Success: {ik_result['success']}")
        print(f"  Iterations: {ik_result['iters']}")
        print(f"  Position Error: {ik_result['pos_err']:.6e} m")
        print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
        print(f"Solved Joint Angles (radians):")
        print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
        print(f"Solved Joint Angles (degrees):")
        print(f"  q_ik = {beauty_print_array(np.rad2deg(ik_result['q']))}")
        
        # Compare with original joint angles
        q_diff = joint_angles - ik_result['q']
        print(f"Joint Angle Difference (original - solved, radians):")
        print(f"  Δq = {beauty_print_array(q_diff)}")
        print(f"  ||Δq|| = {np.linalg.norm(q_diff):.6e} rad")
    
    # Verify IK solution with FK
    T_ik_verify = forward_kinematics(robot_model, ik_result['q'], backend='numpy', return_end=True)
    position_ik = T_ik_verify[:3, 3]
    rotation_ik = T_ik_verify[:3, :3]
    
    # Convert to quaternion (xyzw)
    quat_ik = Rotation.from_matrix(rotation_ik).as_quat()  # xyzw order
    
    position_diff = position_fk - position_ik
    rotation_diff = rotation_fk @ rotation_ik.T
    angle_diff = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1, 1))
    
    results['ik_verification'] = {
        'transform': T_ik_verify,
        'position': position_ik,
        'rotation': rotation_ik,
        'quaternion_xyzw': quat_ik,
        'position_diff': position_diff,
        'angle_diff': angle_diff
    }
    
    if verbose:
        print(f"IK Verification (FK of solved joints):")
        print(f"  Position Error: {np.linalg.norm(position_diff):.6e} m")
        print(f"  Orientation Error: {angle_diff:.6e} rad ({np.rad2deg(angle_diff):.6e} deg)")
    
    # ========================================
    # 3. Jacobian Matrix (Analytical)
    # ========================================
    if verbose:
        beauty_print("Step 3: Jacobian Matrix (Analytical, NumPy)", type="module", centered=True)
        print(f"Computing Jacobian at joint angles:")
        print(f"  q = {beauty_print_array(joint_angles)}")
    
    J = jacobian(robot_model, joint_angles, backend='numpy', method='analytic')
    
    results['jacobian'] = {
        'J': J,
        'shape': J.shape,
        'rank': np.linalg.matrix_rank(J),
        'condition_number': np.linalg.cond(J)
    }
    
    if verbose:
        print(f"Jacobian Matrix (6 × {robot_model.num_dof()}):")
        print(beauty_print_array(J, precision=6))
        print(f"Jacobian Properties:")
        print(f"  Shape: {J.shape}")
        print(f"  Rank: {results['jacobian']['rank']}")
        print(f"  Condition Number: {results['jacobian']['condition_number']:.6e}")
        
        # Compute singular values
        U, s, Vt = np.linalg.svd(J)
        print(f"Singular Values:")
        print(f"  σ = {beauty_print_array(s)}")
        print(f"  σ_min / σ_max = {s[-1] / s[0]:.6e}")
        
        # Check manipulability
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        print(f"Manipulability Index:")
        print(f"  w = sqrt(det(J·J^T)) = {manipulability:.6e}")
    
    # ========================================
    # 4. Jacobian at IK Solution
    # ========================================
    if ik_result['success'] and verbose:
        beauty_print("Step 4: Jacobian at IK Solution", type="module", centered=True)
        print(f"Computing Jacobian at IK solved joints:")
        print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
        
        J_ik = jacobian(robot_model, ik_result['q'], backend='numpy', method='analytic')
        
        results['jacobian_ik'] = {
            'J': J_ik,
            'rank': np.linalg.matrix_rank(J_ik),
            'condition_number': np.linalg.cond(J_ik)
        }
        
        print(f"Jacobian Matrix at IK Solution:")
        print(beauty_print_array(J_ik, precision=6))
        print(f"Jacobian Properties:")
        print(f"  Rank: {results['jacobian_ik']['rank']}")
        print(f"  Condition Number: {results['jacobian_ik']['condition_number']:.6e}")
        
        # Compare Jacobians
        J_diff = J - J_ik
        print(f"Jacobian Difference (original - IK):")
        print(f"  ||ΔJ||_F = {np.linalg.norm(J_diff, 'fro'):.6e}")
        
    return results


def rotation_matrix(T):
    """Extract 3x3 rotation from 4x4 pose."""
    return get_rotation(np.array(T))


def orientation_angle_deg(R1, R2):
    """Compute orientation difference in degrees using new transform API."""
    R1_np = np.array(R1) if not isinstance(R1, np.ndarray) else R1
    R2_np = np.array(R2) if not isinstance(R2, np.ndarray) else R2
    angle_rad = rotation_distance(R1_np, R2_np)
    return math.degrees(angle_rad)


def sample_q(model: RobotModel, scale: float = 0.6):
    """Sample random joint configuration within limits."""
    qs = [0.0] * model.num_dof()
    for js in model._actuated:  # type: ignore[attr-defined]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        qs[js.index] = random.uniform(mid - span, mid + span)
    return qs


def run_closure_validation(robot_model: RobotModel, args):
    """
    Run FK/IK closure validation:
    1. Sample random reachable joint config q*
    2. Compute end-effector pose T* (FK)
    3. Solve IK from zero/random initial guess to recover q*
    4. Compare all IK methods (DLS, Pinv, Transpose) with analytic/numeric Jacobian
    """
    random.seed(42)
    
    beauty_print("FK/IK Closure Validation", type="module", centered=True)
    
    # Sample target configuration
    q_true = sample_q(robot_model)
    fk_true = forward_kinematics(robot_model, q_true, backend='numpy', return_end=True)
    
    beauty_print("Target Configuration", type="module")
    beauty_print(f"  q* = {beauty_print_array(np.array(q_true), 4)}")
    
    beauty_print("Target End-Effector Pose (4x4):", type="module")
    for row in fk_true:
        print(f"  {beauty_print_array(row, 6)}")
    
    # Initial configuration (zero or random)
    q0 = [0.0] * robot_model.num_dof()
    
    # Test all combinations
    methods = ["dls", "pinv", "transpose"]
    jacobian_modes = [(False, "numeric"), (True, "analytic")]
    
    results_table = []
    
    for m in methods:
        for use_ana, tag in jacobian_modes:
            method_name = f"{m.upper()} + {tag.capitalize()} Jacobian"
            beauty_print(f"{method_name}", type="module")
            
            # Create solver (using inverse_kinematics API)
            t0 = perf_counter()
            res = inverse_kinematics(
                robot_model,
                fk_true,
                q0,
                backend='numpy',
                method=m,
                max_iters=120,
                pos_tol=5e-4,
                ori_tol=5e-3,
                use_analytic_jacobian=use_ana,
                use_central_diff=True,
                pos_weight=1.0,
                ori_weight=1.0,
            )
            dt_ms = (perf_counter() - t0) * 1000.0
            
            # Verify solution
            fk_rec = forward_kinematics(robot_model, res['q'], backend='numpy', return_end=True)
            pos_diff = np.linalg.norm(fk_true[:3, 3] - fk_rec[:3, 3])
            ang_deg = orientation_angle_deg(rotation_matrix(fk_true), rotation_matrix(fk_rec))
            
            # Display results
            status_type = "success" if res.get('success') else "warning"
            beauty_print(
                f"  Success: {res.get('success')} | "
                f"Iters: {res.get('iters')} | "
                f"Pos Err: {pos_diff:.2e} m | "
                f"Ori Err: {ang_deg:.2e}° | "
                f"Time: {dt_ms:.2f} ms",
                type=status_type,
            )
            beauty_print(f"  Solution q = {beauty_print_array(np.array(res['q']), 4)}")
            
            results_table.append({
                'method': method_name,
                'success': res.get('success'),
                'iters': res.get('iters'),
                'pos_err': pos_diff,
                'ori_err': ang_deg,
                'time_ms': dt_ms
            })
    
    # Summary table
    beauty_print("Summary Table", type="module")
    print(f"{'Method':<30} | {'Success':<8} | {'Iters':<6} | {'Pos Err (m)':<12} | {'Ori Err (°)':<12} | {'Time (ms)':<10}")
    for r in results_table:
        print(
            f"{r['method']:<30} | "
            f"{str(r['success']):<8} | "
            f"{r['iters']:<6} | "
            f"{r['pos_err']:<12.2e} | "
            f"{r['ori_err']:<12.2e} | "
            f"{r['time_ms']:<10.2f}"
        )
    
    beauty_print("True Target q*", type="module")
    beauty_print(f"  {beauty_print_array(np.array(q_true), 4)}")
    

def main(args):
    # Set random seed
    np.random.seed(args.seed)
    
    robot_model = RobotModel(str(args.urdf), end_link=args.end_link)
    
    # Check if closure validation mode
    if args.closure:
        run_closure_validation(robot_model, args)
        return
    
    # Determine joint angles
    rng = np.random.default_rng(args.seed)
    
    if args.joints is not None:
        joint_angles = np.array(args.joints)
        if len(joint_angles) != robot_model.num_dof():
            print(f"❌ Error: Expected {robot_model.num_dof()} joint angles, got {len(joint_angles)}")
            return
    elif args.joints_deg is not None:
        joint_angles = np.deg2rad(args.joints_deg)
        if len(joint_angles) != robot_model.num_dof():
            print(f"❌ Error: Expected {robot_model.num_dof()} joint angles, got {len(joint_angles)}")
            return
    elif args.random:
        joint_angles = np.array(robot_model.random_q(rng))
        print("🎲 Using random joint angles")
    else:
        # Default: zero configuration
        joint_angles = np.zeros(robot_model.num_dof())
        print("⚙️  Using zero configuration")
    
    # Compute FK, IK, and Jacobian
    results = compute_fk_ik_jacobian(
        robot_model=robot_model,
        joint_angles=joint_angles,
        ik_max_iters=args.ik_iters,
        ik_pos_tol=args.ik_pos_tol,
        ik_ori_tol=args.ik_ori_tol,
        verbose=not args.quiet
    )
    
    # Summary
    if not args.quiet:
        beauty_print("Summary", type="module", centered=True)
        print(f"✓ Forward Kinematics computed successfully")
        print(f"  Position: {beauty_print_array(results['fk']['position'], 3)} m")
        print(f"  Orientation: {beauty_print_array(np.rad2deg(results['fk']['euler_xyz']), 3)} deg")
        print(f"  Quaternion (xyzw): {beauty_print_array(results['fk']['quaternion_xyzw'], 6)}")
        
        print(f"✓ Inverse Kinematics (DLS) {'succeeded' if results['ik']['success'] else 'failed'}")
        if results['ik']['success']:
            print(f"  Iterations: {results['ik']['iters']}")
            print(f"  Position Error: {results['ik']['pos_err']:.6e} m")
            print(f"  Orientation Error: {results['ik']['ori_err']:.6e} rad")
        
        print(f"✓ Jacobian Matrix computed successfully")
        print(f"  Shape: {results['jacobian']['shape']}")
        print(f"  Rank: {results['jacobian']['rank']}")
        print(f"  Condition Number: {results['jacobian']['condition_number']:.6e}")
    return results


if __name__ == '__main__':
    import synriard
    model_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm")

    parser = argparse.ArgumentParser(
        description='Compute FK, IK (DLS), and Jacobian (Analytical) for specified joint angles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration options (NEW)
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Load configuration from YAML file (overrides default values)'
    )
    config_group.add_argument(
        '--override',
        type=str,
        nargs='+',
        default=[],
        help='Override config values (e.g., robot.end_link=tool0 kinematics.ik.solver.max_iterations=200)'
    )
    
    # Robot options
    robot_group = parser.add_argument_group('Robot Options')
    robot_group.add_argument(
        '--urdf',
        type=str,
        default=model_path,
        help='Path to URDF file'
    )
    robot_group.add_argument(
        '--end-link',
        type=str,
        default='tool0',
        # default="left_arm_gripper_left_finger",
        help='End-effector link name (default: tool0)'
    )
    
    # Joint angle options
    joints_group = parser.add_argument_group('Joint Angle Options')
    joints_group.add_argument(
        '--joints',
        type=float,
        nargs='+',
        default=None,
        help='Joint angles in radians (space-separated)'
    )
    joints_group.add_argument(
        '--joints-deg',
        type=float,
        nargs='+',
        default=None,
        help='Joint angles in degrees (space-separated)'
    )
    joints_group.add_argument(
        '--random',
        action='store_true',
        help='Use random joint angles'
    )
    joints_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # IK options
    ik_group = parser.add_argument_group('IK Solver Options')
    ik_group.add_argument(
        '--ik-iters',
        type=int,
        default=100,
        help='Maximum IK iterations'
    )
    ik_group.add_argument(
        '--ik-pos-tol',
        type=float,
        default=1e-4,
        help='IK position tolerance (m)'
    )
    ik_group.add_argument(
        '--ik-ori-tol',
        type=float,
        default=1e-4,
        help='IK orientation tolerance (rad)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    output_group.add_argument(
        '--closure',
        action='store_true',
        help='Run FK/IK closure validation (tests all IK methods with analytic/numeric Jacobian)'
    )

    args = parser.parse_args()
    
    # Handle configuration if specified
    if args.config:
        print(f"✓ Loading configuration from: {args.config}")
        config_manager = ConfigManager(args.config)
        
        # Apply overrides
        if args.override:
            print("  Applying overrides:")
            for override in args.override:
                if '=' in override:
                    key, value = override.split('=', 1)
                    try:
                        value = eval(value)
                    except:
                        pass
                    OmegaConf.update(config_manager.cfg, key, value, merge=True)
                    print(f"    {key} = {value}")
        
        # Override args with config values
        args.urdf = config_manager.cfg.robot.urdf_path
        args.end_link = config_manager.cfg.robot.end_link
        args.ik_iters = config_manager.cfg.kinematics.ik.solver.max_iterations
        args.ik_pos_tol = config_manager.cfg.kinematics.ik.solver.position_tolerance
        args.ik_ori_tol = config_manager.cfg.kinematics.ik.solver.orientation_tolerance
        if config_manager.cfg.seed is not None:
            args.seed = config_manager.cfg.seed
        
    elif args.config:
        print("⚠️  OmegaConf not available. Install with: pip install omegaconf")
        print("    Falling back to command-line arguments.\n")

    main(args)
