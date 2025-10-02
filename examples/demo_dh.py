"""Denavit-Hartenberg (DH) parameter extraction and validation.

Demonstrates                q.append(float(rng.uniform(lo * 0.6, hi * 0.6)))
        
        beauty_print(f"q = [{', '.join(f'{x:+.3f}' for x in q)}]")
        
        # Compute FKExtraction of standard and modified DH parameters from URDF
2. Comparison of DH-based FK vs native URDF FK
3. Statistical evaluation of approximation quality

Usage:
    python examples/demo_dh.py [--mode extract|compare|quality]
    python examples/demo_dh.py --mode quality --samples 100
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
from robocore import RobotModel
from robocore.transform.dh import (
    extract_dh_parameters,
    forward_kinematics_dh_standard,
    forward_kinematics_dh_modified,
    evaluate_dh_fit,
)
from robocore.utils.beauty_logger import beauty_print


def fmt_dh_row(r):
    """Format DH parameter row for display."""
    return (
        f"{r.joint_name:<10} {r.joint_type:<9} "
        f"α={r.alpha: .5f}  a={r.a: .5f}  "
        f"d={r.d: .5f}  θ={r.theta: .5f}"
    )


def orientation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Compute rotation angle between two rotation matrices (radians)."""
    tr = np.trace(Ra.T @ Rb)
    val = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(val))


def mode_extract(model: RobotModel):
    """Extract and display DH parameters."""
    beauty_print("DH Parameter Extraction", type="module")
    
    res = extract_dh_parameters(model)
    
    beauty_print("\nStandard DH (Approximation)", type="module")
    for row in res.standard:
        print("  " + fmt_dh_row(row))
    
    beauty_print("\nModified DH (Approximation)", type="module")
    for row in res.modified:
        print("  " + fmt_dh_row(row))
    
    beauty_print(f"\nAxis misalignment norm: {res.axis_misalignment_norm:.6e} m", type="info")
    if res.notes:
        beauty_print(f"Notes: {res.notes}", type="warning")


def mode_compare(model: RobotModel, samples: int = 5, seed: int = 42):
    """Detailed comparison of URDF FK vs DH FK."""
    beauty_print(f"DH FK Comparison ({samples} samples)", type="module")
    
    rng = np.random.default_rng(seed)
    res = extract_dh_parameters(model)
    
    for idx in range(samples):
        beauty_print(f"\nSample {idx + 1}", type="module")
        
        # Random configuration
        q = []
        for js in model._actuated:  # type: ignore[attr-defined]
            lo, hi = -1.0, 1.0
            if js.limit:
                if js.limit[0] is not None: lo = js.limit[0]
                if js.limit[1] is not None: hi = js.limit[1]
            q.append(float(rng.uniform(lo * 0.6, hi * 0.6)))
        
        print(f"q = [{', '.join(f'{x:+.3f}' for x in q)}]")
        
        # Compute FK
        T_true = np.array(model.forward_kinematics(q)['end'])
        T_std = forward_kinematics_dh_standard(res.standard, q)
        T_mod = forward_kinematics_dh_modified(res.modified, q)
        
        # Position errors
        p_true = T_true[:3, 3]
        p_std = T_std[:3, 3]
        p_mod = T_mod[:3, 3]
        
        pos_err_std = np.linalg.norm(p_std - p_true)
        pos_err_mod = np.linalg.norm(p_mod - p_true)
        
        # Orientation errors
        R_true = T_true[:3, :3]
        R_std = T_std[:3, :3]
        R_mod = T_mod[:3, :3]
        
        ang_err_std = orientation_angle(R_true, R_std)
        ang_err_mod = orientation_angle(R_true, R_mod)
        
        beauty_print(f"  Standard DH: pos_err={pos_err_std:.3e} m, ang_err={ang_err_std:.3e} rad")
        beauty_print(f"  Modified DH: pos_err={pos_err_mod:.3e} m, ang_err={ang_err_mod:.3e} rad")


def mode_quality(model: RobotModel, samples: int = 80, seed: int = 123):
    """Statistical evaluation of DH approximation quality."""
    beauty_print(f"DH Fit Quality Assessment ({samples} samples)", type="module")
    
    res = extract_dh_parameters(model)
    stats = evaluate_dh_fit(model, res, samples=samples, scale=0.6, seed=seed)
    
    beauty_print(f"\nAxis misalignment: {res.axis_misalignment_norm:.6e} m", type="info")
    if res.notes:
        beauty_print(f"Notes: {res.notes}", type="warning")
    
    beauty_print("\nStandard DH", type="module")
    mean_p, med_p, max_p = stats['standard_pos']
    mean_a, med_a, max_a = stats['standard_ang']
    beauty_print(f"Position error (m):      mean={mean_p:.3e}, median={med_p:.3e}, max={max_p:.3e}")
    beauty_print(f"Orientation error (rad): mean={mean_a:.3e}, median={med_a:.3e}, max={max_a:.3e}")
    
    beauty_print("\nModified DH", type="module")
    mean_p, med_p, max_p = stats['modified_pos']
    mean_a, med_a, max_a = stats['modified_ang']
    beauty_print(f"Position error (m):      mean={mean_p:.3e}, median={med_p:.3e}, max={max_p:.3e}")
    beauty_print(f"Orientation error (rad): mean={mean_a:.3e}, median={med_a:.3e}, max={max_a:.3e}")


def main(args):
    # Load model
    base = Path(__file__).resolve().parents[1]
    urdf = os.path.join(base, "robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf")
    model = RobotModel(str(urdf), end_link="tool0")
    
    if args.mode == 'extract':
        mode_extract(model)
    elif args.mode == 'compare':
        mode_compare(model, samples=args.samples, seed=args.seed)
    elif args.mode == 'quality':
        mode_quality(model, samples=args.samples, seed=args.seed)
    
    beauty_print("\n✓ DH demo complete", type="success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DH parameter demonstration")
    parser.add_argument('--mode', choices=['extract', 'compare', 'quality'],
                        default='extract', help='Demonstration mode')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples (for compare/quality modes)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    main(args)
