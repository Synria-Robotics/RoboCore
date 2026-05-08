#!/usr/bin/env python3
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import os
import numpy as np
import argparse
from pathlib import Path
import time

from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer


def demo_reachable_workspace(analyzer, num_samples=10000, visualize=False):
    """Demonstrate reachable workspace computation."""
    print("\n" + "="*70)
    print("REACHABLE WORKSPACE ANALYSIS")
    print("="*70)
    
    print(f"\nComputing reachable workspace with {num_samples} samples...")
    print("Method: Monte Carlo sampling")
    
    start_time = time.time()
    points = analyzer.compute_reachable_workspace(
        num_samples=num_samples,
        method='monte_carlo',
        seed=42
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Computation complete in {elapsed:.2f}s")
    print(f"  Reachable points: {len(points)}")
    
    # Get workspace bounds
    bounds = analyzer.get_workspace_bounds(points)
    print(f"\nWorkspace Bounds:")
    print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m  (range: {bounds['x'][1]-bounds['x'][0]:.3f} m)")
    print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m  (range: {bounds['y'][1]-bounds['y'][0]:.3f} m)")
    print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m  (range: {bounds['z'][1]-bounds['z'][0]:.3f} m)")
    
    # Estimate volume
    print(f"\nWorkspace Volume Estimation:")
    
    vol_convex = analyzer.estimate_workspace_volume(points, method='convex_hull')
    print(f"  Convex hull volume: {vol_convex:.6f} m³")
    
    vol_voxel = analyzer.estimate_workspace_volume(points, method='voxel')
    print(f"  Voxel-based volume: {vol_voxel:.6f} m³")
    
    print(f"  Volume ratio (voxel/convex): {vol_voxel/vol_convex:.2%}")
    
    # Check specific points
    print(f"\nReachability Tests:")
    test_points = [
        np.array([0.3, 0.0, 0.3]),
        np.array([0.5, 0.2, 0.4]),
        np.array([1.0, 1.0, 1.0])  # Likely unreachable
    ]
    
    for i, point in enumerate(test_points, 1):
        reachable = analyzer.check_point_in_workspace(point, points, tolerance=0.05)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  Point {i} {point}: {status}")
    
    # Visualize
    if visualize:
        print("\nGenerating visualization...")
        try:
            analyzer.visualize_workspace(points, show_bounds=True, alpha=0.1)
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
    
    return points


def demo_dexterous_workspace(analyzer, num_samples=10000, visualize=False):
    """Demonstrate dexterous workspace analysis."""
    print("\n" + "="*70)
    print("DEXTEROUS WORKSPACE ANALYSIS")
    print("="*70)
    
    print(f"\nComputing dexterous workspace...")
    print(f"Samples: {num_samples}")
    print(f"Required orientations: 8")
    
    start_time = time.time()
    dex_points = analyzer.compute_dexterous_workspace(
        num_samples=num_samples,
        num_orientations=8,
        tolerance=0.02
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Computation complete in {elapsed:.2f}s")
    print(f"  Dexterous points: {len(dex_points)}")
    
    if len(dex_points) > 0:
        bounds = analyzer.get_workspace_bounds(dex_points)
        print(f"\nDexterous Workspace Bounds:")
        print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
        print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
        print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
        
        # Compare with reachable workspace
        if 'reachable_points' in analyzer._workspace_cache:
            reachable_points = analyzer._workspace_cache['reachable_points']
            ratio = len(dex_points) / len(reachable_points)
            print(f"\nDexterous/Reachable ratio: {ratio:.2%}")
            print(f"  ({len(dex_points)}/{len(reachable_points)} points)")
    else:
        print("\n⚠️  No dexterous workspace points found.")
        print("  Try: Increase num_samples or decrease num_orientations")
    
    return dex_points


def demo_workspace_density(analyzer):
    """Demonstrate workspace density analysis."""
    print("\n" + "="*70)
    print("WORKSPACE DENSITY ANALYSIS")
    print("="*70)
    
    if 'reachable_points' not in analyzer._workspace_cache:
        print("\n⚠️  No workspace data available. Run reachable workspace first.")
        return
    
    print("\nComputing density distribution...")
    density, (x, y, z) = analyzer.compute_workspace_density(grid_resolution=15)
    
    print(f"✓ Density grid: {density.shape}")
    print(f"  Total voxels: {density.size}")
    print(f"  Occupied voxels: {np.sum(density > 0)}")
    print(f"  Max density: {np.max(density):.1f} points/voxel")
    
    # Find highest density region
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    max_pos = (x[max_idx[0]], y[max_idx[1]], z[max_idx[2]])
    
    print(f"\nHighest density region:")
    print(f"  Position: ({max_pos[0]:.3f}, {max_pos[1]:.3f}, {max_pos[2]:.3f})")
    print(f"  Density: {density[max_idx]:.1f} points/voxel")
    
    # Density statistics
    occupied_density = density[density > 0]
    if len(occupied_density) > 0:
        print(f"\nDensity statistics (occupied voxels only):")
        print(f"  Mean: {np.mean(occupied_density):.2f}")
        print(f"  Std: {np.std(occupied_density):.2f}")
        print(f"  Median: {np.median(occupied_density):.2f}")


def demo_singularity_free_workspace(analyzer, num_samples=5000):
    """Demonstrate singularity-free workspace."""
    print("\n" + "="*70)
    print("SINGULARITY-FREE WORKSPACE")
    print("="*70)
    
    print(f"\nFinding singularity-free regions...")
    print(f"Samples: {num_samples}")
    print(f"Manipulability threshold: 0.01")
    
    start_time = time.time()
    safe_points = analyzer.find_singularity_free_regions(
        num_samples=num_samples,
        manipulability_threshold=0.01
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Computation complete in {elapsed:.2f}s")
    
    if len(safe_points) > 0:
        print(f"  Safe points: {len(safe_points)}")
        print(f"  Safety ratio: {len(safe_points)/num_samples:.2%}")
        
        bounds = analyzer.get_workspace_bounds(safe_points)
        print(f"\nSafe Workspace Bounds:")
        print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
        print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
        print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
    else:
        print("\n⚠️  No singularity-free points found.")
        print("  Try: Lower manipulability_threshold")
    
    return safe_points


def demo_workspace_statistics(analyzer):
    """Print comprehensive workspace statistics."""
    print("\n" + "="*70)
    print("WORKSPACE STATISTICS SUMMARY")
    print("="*70)
    
    if 'reachable_points' not in analyzer._workspace_cache:
        print("\n⚠️  No workspace data available.")
        return
    
    points = analyzer._workspace_cache['reachable_points']
    bounds = analyzer.get_workspace_bounds(points)
    
    print(f"\nDataset:")
    print(f"  Total points: {len(points)}")
    print(f"  Sampling method: {analyzer._workspace_cache.get('method', 'unknown')}")
    print(f"  Samples requested: {analyzer._workspace_cache.get('num_samples', 'unknown')}")
    
    print(f"\nSpatial Extent:")
    x_range = bounds['x'][1] - bounds['x'][0]
    y_range = bounds['y'][1] - bounds['y'][0]
    z_range = bounds['z'][1] - bounds['z'][0]
    
    print(f"  X range: {x_range:.3f} m")
    print(f"  Y range: {y_range:.3f} m")
    print(f"  Z range: {z_range:.3f} m")
    print(f"  Max extent: {max(x_range, y_range, z_range):.3f} m")
    print(f"  Min extent: {min(x_range, y_range, z_range):.3f} m")
    
    print(f"\nCenter of Workspace:")
    center = np.mean(points, axis=0)
    print(f"  Position: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    print(f"\nPoint Distribution:")
    std = np.std(points, axis=0)
    print(f"  Std X: {std[0]:.3f} m")
    print(f"  Std Y: {std[1]:.3f} m")
    print(f"  Std Z: {std[2]:.3f} m")
    
    # Distance from origin
    distances = np.linalg.norm(points, axis=1)
    print(f"\nDistance from Origin:")
    print(f"  Min: {np.min(distances):.3f} m")
    print(f"  Max: {np.max(distances):.3f} m")
    print(f"  Mean: {np.mean(distances):.3f} m")


def main():
    parser = argparse.ArgumentParser(description='Workspace Analysis Demo')
    parser.add_argument('--robot', type=str, default='bessica',
                       choices=['alicia', 'bessica'],
                       help='Robot model')
    parser.add_argument('--arm', type=str, default='left',
                       choices=['left', 'right'],
                       help='Arm to use (for bessica)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples for workspace computation')
    parser.add_argument('--visualize', action='store_true',
                       help='Show 3D visualization (requires matplotlib)')
    parser.add_argument('--dexterous', action='store_true',
                       help='Compute dexterous workspace')
    parser.add_argument('--singularity', action='store_true',
                       help='Find singularity-free regions')
    args = parser.parse_args()
    
    # Load robot model
    print("="*70)
    print("WORKSPACE ANALYSIS DEMO")
    print("="*70)
    print(f"\nLoading robot model...")
    
    if args.robot == 'alicia':
        urdf_path = os.path.join(Path(__file__).parent.parent,
                                 '../robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf')
        dof = 6
    else:  # bessica
        urdf_path = os.path.join(Path(__file__).parent.parent, '../robocore/assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf')
        dof = 7
    
    model = RobotModel(str(urdf_path))
    print(f"✓ Loaded {args.robot} ({dof}-DOF)")
    
    # Create analyzer
    analyzer = WorkspaceAnalyzer(model)
    print(f"✓ Workspace analyzer initialized")
    
    # Demo 1: Reachable workspace
    demo_reachable_workspace(analyzer, num_samples=args.samples, visualize=args.visualize)
    
    # Demo 2: Workspace density
    demo_workspace_density(analyzer)
    
    # Demo 3: Statistics
    demo_workspace_statistics(analyzer)
    
    # Optional: Dexterous workspace
    if args.dexterous:
        demo_dexterous_workspace(analyzer, num_samples=args.samples // 2)
    
    # Optional: Singularity-free workspace
    if args.singularity:
        demo_singularity_free_workspace(analyzer, num_samples=args.samples // 2)
    
    print("\n" + "="*70)
    print("✓ Workspace Analysis Demo Complete!")
    print("="*70)
    
    # Tips
    print("\nTips:")
    print("  • Increase --samples for more accurate results")
    print("  • Use --visualize to see 3D workspace plot")
    print("  • Use --dexterous to analyze dexterous workspace")
    print("  • Use --singularity to find safe regions")


if __name__ == '__main__':
    main()
