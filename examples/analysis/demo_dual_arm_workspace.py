"""Dual-arm workspace analysis demo.

Demonstrates:
- Per-arm workspace computation and caching
- Reachability checking for target poses
- Workspace bounds and volume estimation
- Combined dual-arm workspace visualization

Run:
    python examples/analysis/demo_dual_arm_workspace.py
"""
from __future__ import annotations
import numpy as np
from robotcore.modeling.robot_model import RobotModel
from robotcore.utils.path import get_robocore_path
from robotcore.utils.beauty_logger import beauty_print

URDF_REL = "assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"


def main():
    beauty_print("=== Dual-Arm Workspace Analysis ===", type="module", centered=True)
    
    # Load and spawn chains
    base = RobotModel(get_robocore_path(URDF_REL))
    leaves = base.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper_left_finger' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper_left_finger' in l)
    
    left = base.spawn_chain(left_end)
    right = base.spawn_chain(right_end)
    
    # Compute workspaces
    beauty_print("\n1. Computing left arm workspace...")
    left_ws = left.compute_workspace(num_samples=5000, method='monte_carlo', verbose=True)
    
    beauty_print("\n2. Computing right arm workspace...")
    right_ws = right.compute_workspace(num_samples=5000, method='monte_carlo', verbose=True)
    
    # Get bounds
    left_bounds = left.get_workspace_bounds()
    right_bounds = right.get_workspace_bounds()
    
    beauty_print("\n3. Workspace bounds:")
    print(f"  Left arm:")
    print(f"    X: [{left_bounds['x'][0]:.3f}, {left_bounds['x'][1]:.3f}] m")
    print(f"    Y: [{left_bounds['y'][0]:.3f}, {left_bounds['y'][1]:.3f}] m")
    print(f"    Z: [{left_bounds['z'][0]:.3f}, {left_bounds['z'][1]:.3f}] m")
    print(f"  Right arm:")
    print(f"    X: [{right_bounds['x'][0]:.3f}, {right_bounds['x'][1]:.3f}] m")
    print(f"    Y: [{right_bounds['y'][0]:.3f}, {right_bounds['y'][1]:.3f}] m")
    print(f"    Z: [{right_bounds['z'][0]:.3f}, {right_bounds['z'][1]:.3f}] m")
    
    # Estimate volumes
    try:
        from scipy.spatial import ConvexHull
        left_hull = ConvexHull(left_ws)
        right_hull = ConvexHull(right_ws)
        beauty_print(f"\n4. Workspace volumes (convex hull):")
        print(f"  Left arm:  {left_hull.volume:.4f} m³")
        print(f"  Right arm: {right_hull.volume:.4f} m³")
    except Exception as e:
        beauty_print(f"⚠️ Volume estimation skipped: {e}", type="warning")
    
    # Test reachability
    beauty_print("\n5. Testing reachability for sample points:")
    test_points = [
        np.array([0.3, 0.2, 0.8]),   # front center
        np.array([-0.5, 0.3, 0.6]),  # left side
        np.array([0.5, -0.3, 0.6]),  # right side
        np.array([0.0, 0.0, 1.5]),   # high center
    ]
    
    for i, pt in enumerate(test_points):
        left_reach = left.is_point_reachable(pt, tolerance=0.08)
        right_reach = right.is_point_reachable(pt, tolerance=0.08)
        print(f"  Point {i+1} {pt}: Left={left_reach}, Right={right_reach}")
    
    # Overlap analysis
    beauty_print("\n6. Workspace overlap estimation:")
    # Simple grid-based overlap check
    x_min = max(left_bounds['x'][0], right_bounds['x'][0])
    x_max = min(left_bounds['x'][1], right_bounds['x'][1])
    y_min = max(left_bounds['y'][0], right_bounds['y'][0])
    y_max = min(left_bounds['y'][1], right_bounds['y'][1])
    z_min = max(left_bounds['z'][0], right_bounds['z'][0])
    z_max = min(left_bounds['z'][1], right_bounds['z'][1])
    
    if x_min < x_max and y_min < y_max and z_min < z_max:
        overlap_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        print(f"  Bounding box overlap: {overlap_volume:.4f} m³")
        print(f"  Overlap region: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
    else:
        print("  No bounding box overlap detected")
    
    beauty_print("\n✓ Workspace analysis complete.", type="success")
    beauty_print("💡 Tip: Workspace is cached in model instances. Subsequent calls to is_point_reachable() are fast!")


if __name__ == '__main__':
    main()
