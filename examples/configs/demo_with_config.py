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
from omegaconf import OmegaConf

from robocore.configs import ConfigManager, get_default_config
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian


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
    arr = np.asarray(arr)
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
    q = [0.0] * model.num_dof
    # Get chain joint indices for the current chain
    chain_indices = model._get_joint_indices(model.base_link, model.end_link)
    for idx in chain_indices:
        js = model.joint_list[idx]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        q[idx] = float(rng.uniform(mid - span, mid + span))
    return np.array(q)


def compute_kinematics(config: ConfigManager, joint_angles: np.ndarray):
    """
    Compute FK, IK, and Jacobian using configuration.
    
    Parameters
    ----------
    config : ConfigManager
        Configuration manager
    joint_angles : np.ndarray
        Joint angles (radians)
    """
    # Load robot model from config
    robot_cfg = config.cfg.robot
    robot_model = RobotModel(robot_cfg.urdf_path, end_link=robot_cfg.end_link)
    
    print(f"📦 Robot: {Path(robot_cfg.urdf_path).name}")
    print(f"   DOF: {robot_model.num_dof}, End Link: {robot_cfg.end_link}")
    
    # Get config values
    fk_backend = config.cfg.kinematics.fk_backend
    ik_cfg = config.cfg.kinematics.ik
    jac_backend = config.cfg.kinematics.jacobian_backend
    jac_method = config.cfg.kinematics.jacobian_method
    
    print_separator("Forward Kinematics")
    print(f"Joint Angles: {format_array(joint_angles)} rad")
    print(f"Backend: {fk_backend}")
    
    # Set global backend for FK
    import robocore
    robocore.set_backend(fk_backend)
    
    # Compute FK
    T_fk = forward_kinematics(
        robot_model, 
        joint_angles, 
        return_end=True
    )
    
    position = T_fk[:3, 3]
    rotation = T_fk[:3, :3]
    
    # Convert to quaternion
    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    euler_xyz = Rotation.from_matrix(rotation).as_euler('xyz')
    
    print(f"\nEnd-Effector Pose:")
    print(f"  Position: {format_array(position)} m")
    print(f"  Quaternion (xyzw): {format_array(quat_xyzw, 6)}")
    print(f"  Euler XYZ: {format_array(np.rad2deg(euler_xyz))} deg")
    
    # Compute Inverse Kinematics
    print_separator("Inverse Kinematics")
    print(f"Method: {ik_cfg.method}, Backend: {ik_cfg.backend}")
    print(f"Max Iterations: {ik_cfg.solver.max_iterations}")
    print(f"Tolerances: pos={ik_cfg.solver.position_tolerance}, ori={ik_cfg.solver.orientation_tolerance}")
    
    # Random initial guess
    rng = np.random.default_rng(config.cfg.seed or 42)
    q_init = random_q(robot_model, rng)
    
    # Set global backend for IK
    robocore.set_backend(ik_cfg.backend)
    
    ik_result = inverse_kinematics(
        robot_model,
        T_fk,
        q_init,
        method=ik_cfg.method,
        max_iters=ik_cfg.solver.max_iterations,
        pos_tol=ik_cfg.solver.position_tolerance,
        ori_tol=ik_cfg.solver.orientation_tolerance
    )
    
    print(f"\nIK Solution:")
    print(f"  Success: {ik_result['success']}")
    print(f"  Iterations: {ik_result['iters']}")
    print(f"  Position Error: {ik_result['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
    print(f"  Solution: {format_array(ik_result['q'])} rad")
    
    # Compute Jacobian
    print_separator("Jacobian Matrix")
    print(f"Method: {jac_method}, Backend: {jac_backend}")
    
    # Set global backend for Jacobian
    robocore.set_backend(jac_backend)
    
    J = jacobian(
        robot_model, 
        joint_angles, 
        method=jac_method
    )
    
    print(f"\nJacobian Shape: {J.shape}")
    print(f"Rank: {np.linalg.matrix_rank(J)}")
    print(f"Condition Number: {np.linalg.cond(J):.6e}")
    
    # Compute manipulability
    manipulability = np.sqrt(np.linalg.det(J @ J.T))
    print(f"Manipulability: {manipulability:.6e}")
    
    print_separator()
    
    return {
        'fk': {'transform': T_fk, 'position': position, 'quaternion': quat_xyzw},
        'ik': ik_result,
        'jacobian': {'J': J, 'manipulability': manipulability}
    }


def main():
    parser = argparse.ArgumentParser(
        description='FK/IK/Jacobian Demo with OmegaConf Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file (default: use built-in default config)'
    )
    
    # Joint angles
    parser.add_argument(
        '--joints',
        type=float,
        nargs='+',
        default=None,
        help='Joint angles in radians'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random joint angles'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    # Save config
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save current configuration to YAML file'
    )
    
    # Print config
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print current configuration and exit'
    )
    
    # Parse known args and config overrides
    args, unknown = parser.parse_known_args()
    
    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return
        config_manager = ConfigManager(config_path)
        print(f"✓ Loaded config from: {config_path}")
    else:
        config_manager = get_default_config()
        print("✓ Using default configuration")
    
    # Apply command-line config overrides (OmegaConf dot notation)
    if unknown:
        overrides = {}
        for arg in unknown:
            if '=' in arg:
                key, value = arg.split('=', 1)
                # Parse value
                try:
                    # Try to eval as Python literal
                    value = eval(value)
                except:
                    # Keep as string
                    pass
                # Support dot notation
                OmegaConf.update(config_manager.cfg, key, value, merge=True)
                print(f"  Override: {key} = {value}")
    
    # Override seed if specified
    if args.seed is not None:
        config_manager.cfg.seed = args.seed
    
    # Print config if requested
    if args.print_config:
        print("\n" + "="*60)
        print("Current Configuration:")
        print("="*60)
        print(config_manager.to_yaml())
        return
    
    # Save config if requested
    if args.save_config:
        save_path = Path(args.save_config)
        config_manager.save(save_path)
        print(f"✓ Configuration saved to: {save_path}")
    
    # Load robot to determine DOF
    robot_cfg = config_manager.cfg.robot
    robot_model = RobotModel(robot_cfg.urdf_path, end_link=robot_cfg.end_link)
    
    # Determine joint angles
    if args.joints is not None:
        joint_angles = np.array(args.joints)
        if len(joint_angles) != robot_model.num_dof:
            print(f"❌ Expected {robot_model.num_dof} joints, got {len(joint_angles)}")
            return
    elif args.random:
        rng = np.random.default_rng(config_manager.cfg.seed or 42)
        joint_angles = random_q(robot_model, rng)
        print("🎲 Using random joint angles")
    else:
        joint_angles = np.zeros(robot_model.num_dof)
        print("⚙️  Using zero configuration")
    
    # Compute kinematics
    print()
    results = compute_kinematics(config_manager, joint_angles)
    
    # Summary
    print("="*60)
    print("Summary:")
    print(f"  FK: {format_array(results['fk']['position'], 3)} m")
    print(f"  IK: {'✓ Success' if results['ik']['success'] else '✗ Failed'} "
          f"({results['ik']['iters']} iterations)")
    print(f"  Jacobian: rank={np.linalg.matrix_rank(results['jacobian']['J'])}, "
          f"manip={results['jacobian']['manipulability']:.4e}")
    print("="*60)


if __name__ == '__main__':
    main()
