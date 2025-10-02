#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FK, IK, and Jacobian Computation Demo (with OmegaConf Support)
================================================================

This script demonstrates:
1. Forward Kinematics (FK): Compute end-effector pose from joint angles
2. Inverse Kinematics (IK): Solve joint angles from end-effector pose using DLS
3. Jacobian: Compute analytical Jacobian matrix at the end-effector pose

Features:
- **NEW: OmegaConf configuration support** (see demo_with_config.py for cleaner example)
- Uses DLS (Damped Least Squares) IK solver
- Uses analytical Jacobian computation (NumPy backend)
- Displays detailed kinematics information
- Verifies IK solution with FK
- Computes manipulability and singularity metrics

Usage Examples:
--------------
# 1. Zero configuration (default)
python demo_fk_ik_jacobian.py

# 2. Specify joint angles in radians
python demo_fk_ik_jacobian.py --joints 0.5 -0.3 1.2 0.8 -0.5 1.5 0.0

# 3. Specify joint angles in degrees
python demo_fk_ik_jacobian.py --joints-deg 30 -15 60 45 -30 90 0

# 4. Use random joint angles
python demo_fk_ik_jacobian.py --random --seed 123

# 5. Custom IK parameters
python demo_fk_ik_jacobian.py --random --ik-iters 200 --ik-pos-tol 1e-5

# 6. Quiet mode (minimal output)
python demo_fk_ik_jacobian.py --quiet --random

# 7. Load configuration from YAML file (NEW)
python demo_fk_ik_jacobian.py --config robocore/configs/default.yaml

# 8. Override specific config values (NEW)
python demo_fk_ik_jacobian.py --config robocore/configs/default.yaml \
    --override robot.end_link=Link7 kinematics.ik.solver.max_iterations=200

NOTE: For a cleaner configuration-focused example, see demo_with_config.py

Author: RoboCore Team
Date: 2025-10-03
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Optional

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian

try:
    from robocore.configs import ConfigManager, get_default_config
    from omegaconf import OmegaConf
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
from robocore.utils.beauty_logger import beauty_print


def print_separator(title: str = "", width: int = 80):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'=' * width}")
        print(f"{title.center(width)}")
        print(f"{'=' * width}")
    else:
        print(f"{'=' * width}")


def format_array(arr: np.ndarray, precision: int = 5) -> str:
    """Format numpy array for pretty printing."""
    arr = np.asarray(arr)  # Convert to numpy array if needed
    if arr.ndim == 1:
        values = ', '.join([f"{x:+.{precision}f}" for x in arr])
        return f"[{values}]"
    elif arr.ndim == 2:
        lines = []
        for row in arr:
            values = '  '.join([f"{x:+.{precision}f}" for x in row])
            lines.append(f"  [{values}]")
        return "[\n" + "\n".join(lines) + "\n]"
    else:
        return str(arr)


def random_q(model, rng, scale=0.5):
    """Generate random joint configuration within limits."""
    q = [0.0] * model.dof()
    for js in model._actuated:
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        q[js.index] = float(rng.uniform(mid - span, mid + span))
    return np.array(q)


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
        print_separator("Step 1: Forward Kinematics (FK)")
        print(f"\nInput Joint Angles (radians):")
        print(f"  q = {format_array(joint_angles)}")
        print(f"\nInput Joint Angles (degrees):")
        print(f"  q = {format_array(np.rad2deg(joint_angles))}")
    
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
        print(f"\nEnd-Effector Position (m):")
        print(f"  p = {format_array(position_fk)}")
        print(f"\nEnd-Effector Orientation (Euler XYZ, radians):")
        print(f"  rpy = {format_array(euler_fk)}")
        print(f"\nEnd-Effector Orientation (Euler XYZ, degrees):")
        print(f"  rpy = {format_array(np.rad2deg(euler_fk))}")
        print(f"\nEnd-Effector Orientation (Quaternion xyzw):")
        print(f"  quat = {format_array(quat_fk, precision=6)}")
        # Add note about quaternion sign ambiguity
        quat_neg = -quat_fk
        print(f"  Note: q and -q represent the same rotation")
        print(f"  -quat = {format_array(quat_neg, precision=6)} (equivalent)")
        print(f"\nRotation Matrix:")
        print(format_array(rotation_fk, precision=6))
        print(f"\nHomogeneous Transformation Matrix:")
        print(format_array(T_fk, precision=6))
    
    # ========================================
    # 2. Inverse Kinematics (DLS)
    # ========================================
    if verbose:
        print_separator("Step 2: Inverse Kinematics (IK) - DLS Solver")
        print(f"\nTarget Pose (from FK):")
        print(f"  Position: {format_array(position_fk)}")
        print(f"  Euler XYZ: {format_array(euler_fk)} rad")
        print(f"  Quaternion xyzw: {format_array(quat_fk, precision=6)}")
    
    # Use a random initial guess
    rng = np.random.default_rng(42)
    q_init = random_q(robot_model, rng)
    
    if verbose:
        print(f"\nInitial Guess (radians):")
        print(f"  q_init = {format_array(q_init)}")
    
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
        print(f"\nIK Solution:")
        print(f"  Success: {ik_result['success']}")
        print(f"  Iterations: {ik_result['iters']}")
        print(f"  Position Error: {ik_result['pos_err']:.6e} m")
        print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
        print(f"\nSolved Joint Angles (radians):")
        print(f"  q_ik = {format_array(ik_result['q'])}")
        print(f"\nSolved Joint Angles (degrees):")
        print(f"  q_ik = {format_array(np.rad2deg(ik_result['q']))}")
        
        # Compare with original joint angles
        q_diff = joint_angles - ik_result['q']
        print(f"\nJoint Angle Difference (original - solved, radians):")
        print(f"  Δq = {format_array(q_diff)}")
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
        print(f"\nIK Verification (FK of solved joints):")
        print(f"  Position Error: {np.linalg.norm(position_diff):.6e} m")
        print(f"  Orientation Error: {angle_diff:.6e} rad ({np.rad2deg(angle_diff):.6e} deg)")
    
    # ========================================
    # 3. Jacobian Matrix (Analytical)
    # ========================================
    if verbose:
        print_separator("Step 3: Jacobian Matrix (Analytical, NumPy)")
        print(f"\nComputing Jacobian at joint angles:")
        print(f"  q = {format_array(joint_angles)}")
    
    J = jacobian(robot_model, joint_angles, backend='numpy', method='analytic')
    
    results['jacobian'] = {
        'J': J,
        'shape': J.shape,
        'rank': np.linalg.matrix_rank(J),
        'condition_number': np.linalg.cond(J)
    }
    
    if verbose:
        print(f"\nJacobian Matrix (6 × {robot_model.dof()}):")
        print(format_array(J, precision=6))
        print(f"\nJacobian Properties:")
        print(f"  Shape: {J.shape}")
        print(f"  Rank: {results['jacobian']['rank']}")
        print(f"  Condition Number: {results['jacobian']['condition_number']:.6e}")
        
        # Compute singular values
        U, s, Vt = np.linalg.svd(J)
        print(f"\nSingular Values:")
        print(f"  σ = {format_array(s)}")
        print(f"  σ_min / σ_max = {s[-1] / s[0]:.6e}")
        
        # Check manipulability
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        print(f"\nManipulability Index:")
        print(f"  w = sqrt(det(J·J^T)) = {manipulability:.6e}")
    
    # ========================================
    # 4. Jacobian at IK Solution
    # ========================================
    if ik_result['success'] and verbose:
        print_separator("Step 4: Jacobian at IK Solution")
        print(f"\nComputing Jacobian at IK solved joints:")
        print(f"  q_ik = {format_array(ik_result['q'])}")
        
        J_ik = jacobian(robot_model, ik_result['q'], backend='numpy', method='analytic')
        
        results['jacobian_ik'] = {
            'J': J_ik,
            'rank': np.linalg.matrix_rank(J_ik),
            'condition_number': np.linalg.cond(J_ik)
        }
        
        print(f"\nJacobian Matrix at IK Solution:")
        print(format_array(J_ik, precision=6))
        print(f"\nJacobian Properties:")
        print(f"  Rank: {results['jacobian_ik']['rank']}")
        print(f"  Condition Number: {results['jacobian_ik']['condition_number']:.6e}")
        
        # Compare Jacobians
        J_diff = J - J_ik
        print(f"\nJacobian Difference (original - IK):")
        print(f"  ||ΔJ||_F = {np.linalg.norm(J_diff, 'fro'):.6e}")
    
    if verbose:
        print_separator()
    
    return results


def main(args):
    # Set random seed
    np.random.seed(args.seed)
    
    # Load robot model
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"❌ Error: URDF file not found: {urdf_path}")
        return
    
    print(f"📦 Loading robot model from: {urdf_path}")
    robot_model = RobotModel(str(urdf_path), end_link=args.end_link)
    print(f"✓ Robot loaded: {robot_model.dof()} DOF, end_link={robot_model.end_link}")
    
    # Determine joint angles
    rng = np.random.default_rng(args.seed)
    
    if args.joints is not None:
        joint_angles = np.array(args.joints)
        if len(joint_angles) != robot_model.dof():
            print(f"❌ Error: Expected {robot_model.dof()} joint angles, got {len(joint_angles)}")
            return
    elif args.joints_deg is not None:
        joint_angles = np.deg2rad(args.joints_deg)
        if len(joint_angles) != robot_model.dof():
            print(f"❌ Error: Expected {robot_model.dof()} joint angles, got {len(joint_angles)}")
            return
    elif args.random:
        joint_angles = random_q(robot_model, rng)
        print("🎲 Using random joint angles")
    else:
        # Default: zero configuration
        joint_angles = np.zeros(robot_model.dof())
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
        print_separator("Summary")
        print(f"\n✓ Forward Kinematics computed successfully")
        print(f"  Position: {format_array(results['fk']['position'], 3)} m")
        print(f"  Orientation: {format_array(np.rad2deg(results['fk']['euler_xyz']), 3)} deg")
        print(f"  Quaternion (xyzw): {format_array(results['fk']['quaternion_xyzw'], 6)}")
        
        print(f"\n✓ Inverse Kinematics (DLS) {'succeeded' if results['ik']['success'] else 'failed'}")
        if results['ik']['success']:
            print(f"  Iterations: {results['ik']['iters']}")
            print(f"  Position Error: {results['ik']['pos_err']:.6e} m")
            print(f"  Orientation Error: {results['ik']['ori_err']:.6e} rad")
        
        print(f"\n✓ Jacobian Matrix computed successfully")
        print(f"  Shape: {results['jacobian']['shape']}")
        print(f"  Rank: {results['jacobian']['rank']}")
        print(f"  Condition Number: {results['jacobian']['condition_number']:.6e}")
        
        print_separator()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute FK, IK (DLS), and Jacobian (Analytical) for specified joint angles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration options (NEW)
    config_group = parser.add_argument_group('Configuration Options (NEW)')
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
        # default='robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf',
        default='robocore/assets/robot/urdf/Bessica-D_v1_0/BessicaDCodver.urdf',
        help='Path to URDF file'
    )
    robot_group.add_argument(
        '--end-link',
        type=str,
        # default='tool0',
        default="left_arm_gripper_left_finger",
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

    args = parser.parse_args()
    
    # Handle configuration if specified
    if args.config and CONFIG_AVAILABLE:
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
        
        print()
    elif args.config and not CONFIG_AVAILABLE:
        print("⚠️  OmegaConf not available. Install with: pip install omegaconf")
        print("    Falling back to command-line arguments.\n")

    main(args)
