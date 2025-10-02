#!/usr/bin/env python3
"""
MuJoCo Trajectory Visualization Demo
=====================================

Demonstrates workspace-constrained trajectory planning with MuJoCo visualization.

Workflow:
1. Analyze workspace
2. Plan trajectory withi    # Load robot model
    print(f"\nLoading robot model...")
    if args.robot == 'alicia':
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Alicia-D_v5_4' / 'alicia_duo_with_gripper.urdf'
    else:
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Bessica-D_v1_0' / 'BessicaDCodver.urdf'
    
    model = RobotModel(str(urdf_path))
    dof = model.dof()
    print(f"✓ Loaded {args.robot} ({dof}-DOF)")
    
    # Find MJCF fileonstraints
3. Visualize in MuJoCo simulator

Usage:
    python demo_mujoco_trajectory.py --robot bessica --arm left
    python demo_mujoco_trajectory.py --robot bessica --arm left --samples 5000 --waypoints 4
    python demo_mujoco_trajectory.py --robot bessica --arm left --export video.mp4
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.planning.trajectory import multi_waypoint_trajectory

try:
    from robocore.bridge.sim.mujoco.trajectory_visualizer import TrajectoryVisualizer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available. Install with: pip install mujoco")


def find_mjcf_file(robot_name: str, arm: str = 'left') -> Path:
    """Find MuJoCo MJCF file for robot."""
    assets_dir = Path(__file__).parent.parent / 'robocore' / 'assets'
    
    # Search for MJCF files
    mjcf_dirs = [
        assets_dir / 'robot' / 'mjcf',
        assets_dir / 'world' / 'mjcf',
    ]
    
    for mjcf_dir in mjcf_dirs:
        if not mjcf_dir.exists():
            continue
        
        # Look for robot-specific MJCF
        patterns = [
            f"{robot_name}*.xml",
            f"{robot_name.upper()}*.xml",
            f"{robot_name.lower()}*.xml",
        ]
        
        for pattern in patterns:
            mjcf_files = list(mjcf_dir.glob(pattern))
            if mjcf_files:
                return mjcf_files[0]
    
    return None


def analyze_workspace_quick(model, num_samples=3000):
    """Quick workspace analysis for trajectory planning."""
    print("\n" + "="*70)
    print("Step 1: Workspace Analysis")
    print("="*70)
    
    analyzer = WorkspaceAnalyzer(model, backend='numpy')
    
    # Compute reachable workspace
    print(f"\nAnalyzing reachable workspace ({num_samples} samples)...")
    reachable_points = analyzer.compute_reachable_workspace(
        num_samples=num_samples,
        method='monte_carlo',
        seed=42
    )
    
    bounds = analyzer.get_workspace_bounds(reachable_points)
    print(f"✓ Bounds: X[{bounds['x'][0]:.2f}, {bounds['x'][1]:.2f}] "
          f"Y[{bounds['y'][0]:.2f}, {bounds['y'][1]:.2f}] "
          f"Z[{bounds['z'][0]:.2f}, {bounds['z'][1]:.2f}]")
    
    # Find safe regions
    print(f"\nFinding safe regions (singularity-free)...")
    safe_points = analyzer.find_singularity_free_regions(
        num_samples=num_samples // 2,
        manipulability_threshold=0.01
    )
    
    if len(safe_points) > 0:
        safe_bounds = analyzer.get_workspace_bounds(safe_points)
        print(f"✓ Safe points: {len(safe_points)} ({len(safe_points)/(num_samples//2):.1%})")
    else:
        print("[WARNING] No safe regions found, using reachable workspace")
        safe_bounds = bounds
    
    return {
        'analyzer': analyzer,
        'reachable_points': reachable_points,
        'bounds': bounds,
        'safe_points': safe_points,
        'safe_bounds': safe_bounds
    }


def plan_trajectory_simple(model, workspace_data, num_waypoints=4):
    """Plan simple trajectory in workspace."""
    print("\n" + "="*70)
    print("Step 2: Trajectory Planning")
    print("="*70)
    
    # Use safe bounds if available
    if len(workspace_data['safe_points']) > 0:
        bounds = workspace_data['safe_bounds']
        print("\nUsing safe workspace region")
    else:
        bounds = workspace_data['bounds']
        print("\nUsing reachable workspace region")
    
    # Start configuration
    q_start = np.zeros(model.dof())
    
    # Generate target waypoints in workspace
    waypoints_cartesian = []
    waypoints_joint = [q_start]
    
    # Get start position
    T_start = forward_kinematics(model, q_start, backend='numpy', return_end=True)
    p_start = T_start[:3, 3]
    waypoints_cartesian.append(p_start)
    
    print(f"\nGenerating {num_waypoints} waypoints...")
    print(f"Waypoint 0 (start): [{p_start[0]:.3f}, {p_start[1]:.3f}, {p_start[2]:.3f}]")
    
    # Generate waypoints
    q_current = q_start.copy()
    success_count = 0
    
    for i in range(1, num_waypoints):
        # Random target in workspace with some margin
        margin = 0.05
        target = np.array([
            np.random.uniform(bounds['x'][0] + margin, bounds['x'][1] - margin),
            np.random.uniform(bounds['y'][0] + margin, bounds['y'][1] - margin),
            np.random.uniform(bounds['z'][0] + margin, bounds['z'][1] - margin)
        ])
        
        waypoints_cartesian.append(target)
        
        # Solve IK
        T_current = forward_kinematics(model, q_current, backend='numpy', return_end=True)
        T_target = T_current.copy()
        T_target[:3, 3] = target
        
        result = inverse_kinematics(
            model, T_target, q_current,
            backend='numpy', method='dls',
            max_iters=200, pos_tol=1e-4, ori_tol=1e-4,
            multi_start=3,  # Multiple attempts
            multi_noise=0.3
        )
        
        if result['success']:
            waypoints_joint.append(result['q'])
            q_current = result['q']
            success_count += 1
            print(f"Waypoint {i}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] ✓")
        else:
            # Use best solution even if not converged
            waypoints_joint.append(result['q'])
            q_current = result['q']
            print(f"Waypoint {i}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] ~")
    
    print(f"\nIK success rate: {success_count}/{num_waypoints-1} = {success_count/(num_waypoints-1):.1%}")
    
    # Generate smooth trajectory
    print(f"\nGenerating smooth trajectory...")
    waypoints_joint = np.array(waypoints_joint)
    
    t, q, qd, qdd = multi_waypoint_trajectory(
        waypoints_joint,
        durations=2.0,  # 2 seconds per segment
        num_points_per_segment=100,
        method='quintic'
    )
    
    print(f"✓ Trajectory generated")
    print(f"  Total duration: {t[-1]:.2f}s")
    print(f"  Total points: {len(t)}")
    
    return {
        'time': t,
        'q': q,
        'qd': qd,
        'qdd': qdd,
        'waypoints': waypoints_joint
    }


def main():
    parser = argparse.ArgumentParser(
        description='MuJoCo Trajectory Visualization Demo'
    )
    parser.add_argument('--robot', type=str, default='alicia',
                       choices=['alicia', 'bessica'],
                       help='Robot model')
    parser.add_argument('--arm', type=str, default='left',
                       choices=['left', 'right'],
                       help='Arm selection (for bessica)')
    parser.add_argument('--samples', type=int, default=3000,
                       help='Workspace sampling count')
    parser.add_argument('--waypoints', type=int, default=4,
                       help='Number of waypoints')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed (1.0 = realtime)')
    parser.add_argument('--no-loop', action='store_true',
                       help='Disable loop playback')
    parser.add_argument('--export', type=str,
                       help='Export video to file (e.g., trajectory.mp4)')
    parser.add_argument('--mjcf', type=str,
                       help='MuJoCo MJCF file path')
    
    args = parser.parse_args()
    
    if not MUJOCO_AVAILABLE:
        print("\n[ERROR] MuJoCo not installed")
        print("Install: pip install mujoco")
        return 1
    
    print("="*70)
    print("MuJoCo Trajectory Visualization Demo")
    print("="*70)
    
    # Load robot model
    print(f"\nLoading robot model...")
    if args.robot == 'alicia':
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Alicia-D_v5_4' / 'alicia_duo_with_gripper.urdf'
        dof = 6
    else:
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Bessica-D_v1_0' / 'BessicaDCodver.urdf'
        dof = 7
    
    model = RobotModel(str(urdf_path))
    print(f"✓ Loaded {args.robot} ({dof}-DOF)")
    
    # Find MJCF file
    if args.mjcf:
        mjcf_path = Path(args.mjcf)
    else:
        print(f"\nSearching for MuJoCo MJCF file...")
        mjcf_path = find_mjcf_file(args.robot, args.arm)
    
    if mjcf_path is None or not mjcf_path.exists():
        print(f"\n[WARNING] MJCF file not found")
        print(f"\nTo create MuJoCo MJCF file:")
        print(f"  1. Manually convert URDF to MJCF format")
        print(f"  2. Or use --mjcf to specify MJCF file path")
        print(f"\nMJCF example location: robocore/assets/robot/mjcf/")
        print(f"\nFallback: Attempting to load URDF directly (may have compatibility issues)")
        
        # Try to use URDF directly
        mjcf_path = urdf_path
    
    print(f"✓ Using model file: {mjcf_path.name}")
    
    # Step 1: Analyze workspace
    workspace_data = analyze_workspace_quick(model, num_samples=args.samples)
    
    # Step 2: Plan trajectory
    trajectory = plan_trajectory_simple(model, workspace_data, num_waypoints=args.waypoints)
    
    # Step 3: Visualize in MuJoCo
    print("\n" + "="*70)
    print("Step 3: MuJoCo Visualization")
    print("="*70)
    
    try:
        # Create visualizer
        print(f"\nInitializing MuJoCo visualizer...")
        viz = TrajectoryVisualizer(mjcf_path=str(mjcf_path))
        
        # Check DOF mismatch (dual-arm vs single-arm, or gripper)
        if viz.model.nq != trajectory['q'].shape[1]:
            print(f"\n[WARNING] DOF mismatch: model={viz.model.nq}, trajectory={trajectory['q'].shape[1]}")
            
            # Case 1: Gripper DOF (trajectory DOF + 1 or + 2)
            if viz.model.nq == trajectory['q'].shape[1] + 1:
                print(f"Model has 1 extra DOF (gripper), extending trajectory...")
                q_full = np.zeros((len(trajectory['q']), viz.model.nq))
                q_full[:, :-1] = trajectory['q']
                trajectory['q'] = q_full
                
                if trajectory['qd'] is not None:
                    qd_full = np.zeros((len(trajectory['qd']), viz.model.nq))
                    qd_full[:, :-1] = trajectory['qd']
                    trajectory['qd'] = qd_full
                
                if trajectory['qdd'] is not None:
                    qdd_full = np.zeros((len(trajectory['qdd']), viz.model.nq))
                    qdd_full[:, :-1] = trajectory['qdd']
                    trajectory['qdd'] = qdd_full
                
                print(f"✓ Trajectory extended to {trajectory['q'].shape[1]} DOF")
            
            # Case 2: Dual gripper (trajectory DOF + 2)
            elif viz.model.nq == trajectory['q'].shape[1] + 2:
                print(f"Model has 2 extra DOF (dual gripper), extending trajectory...")
                q_full = np.zeros((len(trajectory['q']), viz.model.nq))
                q_full[:, :-2] = trajectory['q']
                trajectory['q'] = q_full
                
                if trajectory['qd'] is not None:
                    qd_full = np.zeros((len(trajectory['qd']), viz.model.nq))
                    qd_full[:, :-2] = trajectory['qd']
                    trajectory['qd'] = qd_full
                
                if trajectory['qdd'] is not None:
                    qdd_full = np.zeros((len(trajectory['qdd']), viz.model.nq))
                    qdd_full[:, :-2] = trajectory['qdd']
                    trajectory['qdd'] = qdd_full
                
                print(f"✓ Trajectory extended to {trajectory['q'].shape[1]} DOF")
            
            # Case 3: Dual-arm robot (trajectory is single arm)
            elif viz.model.nq == 14 and trajectory['q'].shape[1] == 7:
                print(f"Detected dual-arm robot, extending trajectory to dual-arm config...")
                # Extend trajectory to dual-arm (keep other arm at zero)
                q_full = np.zeros((len(trajectory['q']), 14))
                
                if args.arm == 'left':
                    # Left arm: joints 0-6
                    q_full[:, :7] = trajectory['q']
                else:
                    # Right arm: joints 7-13
                    q_full[:, 7:14] = trajectory['q']
                
                trajectory['q'] = q_full
                
                # Extend velocities and accelerations if present
                if trajectory['qd'] is not None:
                    qd_full = np.zeros((len(trajectory['qd']), 14))
                    if args.arm == 'left':
                        qd_full[:, :7] = trajectory['qd']
                    else:
                        qd_full[:, 7:14] = trajectory['qd']
                    trajectory['qd'] = qd_full
                
                if trajectory['qdd'] is not None:
                    qdd_full = np.zeros((len(trajectory['qdd']), 14))
                    if args.arm == 'left':
                        qdd_full[:, :7] = trajectory['qdd']
                    else:
                        qdd_full[:, 7:14] = trajectory['qdd']
                    trajectory['qdd'] = qdd_full
                
                print(f"✓ Trajectory extended to {trajectory['q'].shape[1]} DOF")
            
            elif viz.model.nq == 12 and trajectory['q'].shape[1] == 6:
                print(f"Detected dual-arm robot, extending trajectory to dual-arm config...")
                # Alicia dual-arm: 12 DOF (6+6)
                q_full = np.zeros((len(trajectory['q']), 12))
                
                if args.arm == 'left':
                    # Left arm: joints 0-5
                    q_full[:, :6] = trajectory['q']
                else:
                    # Right arm: joints 6-11
                    q_full[:, 6:12] = trajectory['q']
                
                trajectory['q'] = q_full
                
                # Extend velocities and accelerations if present
                if trajectory['qd'] is not None:
                    qd_full = np.zeros((len(trajectory['qd']), 12))
                    if args.arm == 'left':
                        qd_full[:, :6] = trajectory['qd']
                    else:
                        qd_full[:, 6:12] = trajectory['qd']
                    trajectory['qd'] = qd_full
                
                if trajectory['qdd'] is not None:
                    qdd_full = np.zeros((len(trajectory['qdd']), 12))
                    if args.arm == 'left':
                        qdd_full[:, :6] = trajectory['qdd']
                    else:
                        qdd_full[:, 6:12] = trajectory['qdd']
                    trajectory['qdd'] = qdd_full
                
                print(f"✓ Trajectory extended to {trajectory['q'].shape[1]} DOF")
        
        # Load trajectory
        viz.load_trajectory(
            q_trajectory=trajectory['q'],
            time=trajectory['time'],
            qd_trajectory=trajectory['qd'],
            qdd_trajectory=trajectory['qdd']
        )
        
        # Configure playback
        viz.set_speed(args.speed)
        viz.loop = not args.no_loop
        
        # Export video if requested
        if args.export:
            print(f"\nExporting video to: {args.export}")
            viz.export_video(args.export, fps=30, width=1280, height=720)
            print(f"✓ Video export complete")
            return 0
        
        # Start playback
        viz.play()
        
        # Open viewer
        print(f"\nLaunching MuJoCo viewer...")
        viz.visualize(title=f"{args.robot.capitalize()} Trajectory Visualization")
        
    except Exception as e:
        print(f"\n[ERROR] MuJoCo visualization failed: {e}")
        print(f"\nPossible causes:")
        print(f"  1. MJCF file format incorrect")
        print(f"  2. URDF to MuJoCo conversion issues")
        print(f"  3. Missing required mesh files")
        print(f"\nSuggestions:")
        print(f"  1. Check if MJCF file exists")
        print(f"  2. Ensure all mesh file paths are correct")
        print(f"  3. Manually convert URDF using MuJoCo tools")
        return 1
    
    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
