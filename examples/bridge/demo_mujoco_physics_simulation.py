#!/usr/bin/env python3
"""MuJoCo Physics Simulation Demo

Demonstrates real physics simulation for trajectory execution with different
control modes and trajectory smoothness.

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

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robocore.modeling.robot_model import RobotModel
from robocore.planning.trajectory import (
    quintic_polynomial_trajectory,
    linear_joint_trajectory,
    multi_waypoint_trajectory
)

try:
    from robocore.bridge.sim.mujoco.physics_simulator import PhysicsSimulator
    from robocore.bridge.sim.mujoco.trajectory_executor import TrajectoryExecutor
    from robocore.bridge.sim.mujoco.trajectory_evaluator import TrajectoryEvaluator
    from robocore.bridge.sim.mujoco.comparison_analyzer import ComparisonAnalyzer
    from robocore.bridge.sim.mujoco.trajectory_visualizer import TrajectoryVisualizer
    MUJOCO_AVAILABLE = True
except ImportError as e:
    MUJOCO_AVAILABLE = False
    print(f"⚠️  MuJoCo modules not available: {e}")
    print("Install with: pip install mujoco")


def find_mjcf_file(robot_name: str) -> Path:
    """Find MuJoCo MJCF file for robot."""
    # Try common locations
    possible_paths = [
        Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot_descriptions' / 'mjcf',
        Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'mjcf',
    ]
    
    for base_path in possible_paths:
        if not base_path.exists():
            continue
        
        patterns = [
            f"{robot_name}*.xml",
            f"{robot_name.upper()}*.xml",
            f"{robot_name.lower()}*.xml",
        ]
        
        for pattern in patterns:
            mjcf_files = list(base_path.glob(pattern))
            if mjcf_files:
                return mjcf_files[0]
    
    return None


def generate_trajectories(q_start: np.ndarray, q_end: np.ndarray, duration: float = 2.0):
    """Generate smooth and non-smooth trajectories.
    
    Parameters
    ----------
    q_start : np.ndarray
        Start joint configuration
    q_end : np.ndarray
        End joint configuration
    duration : float
        Trajectory duration
    
    Returns
    -------
    smooth_traj : dict
        Smooth trajectory (quintic polynomial)
    non_smooth_traj : dict
        Non-smooth trajectory (step function with sudden jumps)
    """
    # Smooth trajectory (quintic polynomial)
    t_smooth, q_smooth, qd_smooth, qdd_smooth = quintic_polynomial_trajectory(
        q_start, q_end, duration, num_points=200
    )
    
    # Extremely non-smooth trajectory: minimal waypoints with extreme overshoots/undershoots
    # Use only 3 segments to create maximum velocity discontinuities
    num_points = 200
    t_non_smooth = np.linspace(0, duration, num_points)
    
    # Create extreme waypoints: start -> huge overshoot -> undershoot -> end
    # This creates very high velocities and sudden direction changes
    waypoints = [
        (0.0, q_start),
        (duration * 0.25, q_start + (q_end - q_start) * 2.0),  # 200% overshoot!
        (duration * 0.6, q_start + (q_end - q_start) * 0.1),   # 90% undershoot!
        (duration, q_end)
    ]
    
    q_non_smooth = np.zeros((len(t_non_smooth), len(q_start)))
    qd_non_smooth = np.zeros((len(t_non_smooth), len(q_start)))
    qdd_non_smooth = np.zeros((len(t_non_smooth), len(q_start)))
    
    for i, t in enumerate(t_non_smooth):
        # Find which segment
        if t <= waypoints[0][0]:
            q_non_smooth[i] = waypoints[0][1]
            qd_non_smooth[i] = np.zeros_like(q_start)
        elif t >= waypoints[-1][0]:
            q_non_smooth[i] = waypoints[-1][1]
            qd_non_smooth[i] = np.zeros_like(q_start)
        else:
            # Find segment
            for j in range(len(waypoints) - 1):
                t_start, q_start_seg = waypoints[j]
                t_end, q_end_seg = waypoints[j + 1]
                
                if t_start <= t < t_end:
                    seg_duration = t_end - t_start
                    if seg_duration > 1e-6:
                        alpha = (t - t_start) / seg_duration
                        # Linear interpolation
                        q_non_smooth[i] = (1 - alpha) * q_start_seg + alpha * q_end_seg
                        # Very high constant velocity (creates huge jumps at boundaries)
                        qd_non_smooth[i] = (q_end_seg - q_start_seg) / seg_duration
                    else:
                        q_non_smooth[i] = q_start_seg
                        qd_non_smooth[i] = np.zeros_like(q_start)
                    break
        
        # Zero acceleration (infinite jerk at waypoints)
        qdd_non_smooth[i] = np.zeros_like(q_start)
    
    # Add sudden position jumps at multiple points for maximum non-smoothness
    jump_times = [duration * 0.15, duration * 0.45, duration * 0.75]
    for jump_time in jump_times:
        jump_idx = np.argmin(np.abs(t_non_smooth - jump_time))
        if 0 < jump_idx < len(q_non_smooth):
            # Add sudden position jump (20% of total displacement)
            jump_magnitude = 0.25 * (q_end - q_start) * np.random.choice([-1, 1], size=len(q_start))
            q_non_smooth[jump_idx:] += jump_magnitude
            # Create very high velocity spike at jump point
            dt = t_non_smooth[jump_idx] - t_non_smooth[jump_idx - 1]
            if dt > 0:
                qd_non_smooth[jump_idx] = jump_magnitude / dt
    
    return {
        'smooth': {
            'time': t_smooth,
            'q': q_smooth,
            'qd': qd_smooth,
            'qdd': qdd_smooth
        },
        'non_smooth': {
            'time': t_non_smooth,
            'q': q_non_smooth,
            'qd': qd_non_smooth,
            'qdd': qdd_non_smooth
        }
    }


def run_comparison(
    mjcf_path: str,
    control_modes: list = None,
    servo_limits: dict = None,
    plot: bool = True,
    visualize: bool = False
):
    """Run trajectory execution comparison.
    
    Parameters
    ----------
    mjcf_path : str
        Path to MuJoCo MJCF file
    control_modes : list, optional
        List of control modes to test
    servo_limits : dict, optional
        Servo limits: {'max_velocity': value, 'max_acceleration': value}
    plot : bool
        Whether to plot results
    """
    if control_modes is None:
        control_modes = ['position', 'velocity', 'torque']
    
    print("=" * 70)
    print("MuJoCo Physics Simulation - Trajectory Quality Evaluation")
    print("=" * 70)
    print()
    
    # Initialize simulator
    print("Initializing physics simulator...")
    simulator = PhysicsSimulator(mjcf_path=mjcf_path, timestep=0.002)
    print(f"✓ Model loaded: {simulator.nq} DOF, {simulator.nu} actuators")
    print()
    
    # Generate test trajectories
    print("Generating test trajectories...")
    n_joints = simulator.nu if simulator.nu > 0 else simulator.nq
    q_start = np.zeros(n_joints)
    # Generate end configuration (extend if needed for gripper joints)
    q_end_base = np.array([0.5, -0.3, 0.4, -0.2, 0.3, -0.1])
    if n_joints > len(q_end_base):
        # Pad with zeros for gripper joints
        q_end = np.concatenate([q_end_base, np.zeros(n_joints - len(q_end_base))])
    else:
        q_end = q_end_base[:n_joints]
    
    trajectories = generate_trajectories(q_start, q_end, duration=2.0)
    print("✓ Generated smooth (quintic) and non-smooth (linear) trajectories")
    print()
    
    # Initialize analyzer
    analyzer = ComparisonAnalyzer(dt=0.002)
    
    # Test each control mode
    print("Executing trajectories...")
    print("-" * 70)
    
    for control_mode in control_modes:
        print(f"\nControl Mode: {control_mode}")
        simulator.set_control_mode(control_mode)
        
        # Test smooth trajectory
        executor = TrajectoryExecutor(
            simulator,
            max_velocity=servo_limits.get('max_velocity') if servo_limits else None,
            max_acceleration=servo_limits.get('max_acceleration') if servo_limits else None
        )
        
        executor.load_trajectory(
            trajectories['smooth']['q'],
            trajectories['smooth']['time'],
            trajectories['smooth']['qd'],
            trajectories['smooth']['qdd']
        )
        
        exec_data_smooth = executor.execute()
        analyzer.add_comparison(
            f"{control_mode}_smooth",
            exec_data_smooth,
            metadata={'control_mode': control_mode, 'trajectory_type': 'smooth'}
        )
        print(f"  ✓ Smooth trajectory executed")
        
        # Test non-smooth trajectory
        executor.load_trajectory(
            trajectories['non_smooth']['q'],
            trajectories['non_smooth']['time'],
            trajectories['non_smooth']['qd'],
            trajectories['non_smooth']['qdd']
        )
        
        exec_data_non_smooth = executor.execute()
        analyzer.add_comparison(
            f"{control_mode}_non_smooth",
            exec_data_non_smooth,
            metadata={'control_mode': control_mode, 'trajectory_type': 'non_smooth'}
        )
        print(f"  ✓ Non-smooth trajectory executed")
    
    print()
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print()
    
    # Generate comparison report
    report = analyzer.generate_comparison_report()
    print(report)
    
    # Individual evaluation reports
    print("\n" + "=" * 70)
    print("Detailed Evaluation Reports")
    print("=" * 70)
    print()
    
    evaluator = TrajectoryEvaluator(dt=0.002)
    for comp in analyzer.comparisons:
        print(f"\n{comp['name']}:")
        print("-" * 70)
        metrics = comp['metrics']
        print(f"  Position RMS Error: {metrics['tracking_error']['position']['rms']:.6f}")
        print(f"  Position Max Error: {metrics['tracking_error']['position']['max']:.6f}")
        print(f"  Velocity RMS Error: {metrics['tracking_error']['velocity']['rms']:.6f}")
        
        # Handle jerk (may be array or scalar)
        jerk_max = metrics['smoothness']['jerk']['actual']['max']
        if isinstance(jerk_max, np.ndarray):
            jerk_max = np.max(jerk_max) if jerk_max.size > 0 else 0.0
        print(f"  Max Jerk: {jerk_max:.6f}")
        
        # Handle torque (may be array or scalar)
        torque_max = metrics['torque']['max']
        if isinstance(torque_max, np.ndarray):
            torque_max = np.max(torque_max) if torque_max.size > 0 else 0.0
        print(f"  Max Torque: {torque_max:.6f}")
        
        if 'vibration' in metrics:
            print(f"  Has Vibration: {metrics['vibration']['has_vibration']}")
    
    # Save reports
    output_dir = Path("physics_simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write(analyzer.generate_comparison_report())
    
    print(f"\n✓ Reports saved to {output_dir}/")
    
    # MuJoCo viewer visualization
    if visualize:
        print("\n" + "=" * 70)
        print("MuJoCo Viewer Visualization")
        print("=" * 70)
        print("\nSelect trajectory to visualize:")
        for i, comp in enumerate(analyzer.comparisons):
            print(f"  {i+1}. {comp['name']}")
        print("  0. Skip visualization")
        
        try:
            import sys
            if sys.stdin.isatty():
                choice = input("\nEnter choice (default: 1): ").strip()
                if not choice:
                    choice = "1"
                choice_idx = int(choice) - 1
            else:
                # Non-interactive mode, skip visualization
                print("Non-interactive mode, skipping visualization.")
                choice_idx = -1
            
            if 0 <= choice_idx < len(analyzer.comparisons):
                selected_comp = analyzer.comparisons[choice_idx]
                print(f"\nVisualizing: {selected_comp['name']}")
                
                # Create visualizer
                viz = TrajectoryVisualizer(mjcf_path=mjcf_path)
                viz.load_trajectory(
                    q_trajectory=selected_comp['execution_data']['q_desired'],
                    time=selected_comp['execution_data']['time'],
                    qd_trajectory=selected_comp['execution_data']['qd_desired'],
                    qdd_trajectory=selected_comp['execution_data']['qdd_desired']
                )
                viz.set_speed(1.0)
                viz.loop = True
                viz.play()
                
                print("\nOpening MuJoCo viewer...")
                print("Controls: Space=Play/Pause, R=Reset, []=Speed, ESC=Close")
                viz.visualize(title=f"Trajectory: {selected_comp['name']}")
            else:
                print("Skipping visualization.")
        except (ValueError, KeyboardInterrupt):
            print("\nSkipping visualization.")
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Position error comparison
            ax = axes[0, 0]
            for comp in analyzer.comparisons:
                time = comp['execution_data']['time']
                pos_error = comp['execution_data']['q'] - comp['execution_data']['q_desired']
                pos_error_norm = np.linalg.norm(pos_error, axis=1)
                ax.plot(time, pos_error_norm, label=comp['name'])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position Error (rad)')
            ax.set_title('Position Tracking Error')
            ax.legend()
            ax.grid(True)
            
            # Velocity error comparison
            ax = axes[0, 1]
            for comp in analyzer.comparisons:
                time = comp['execution_data']['time']
                vel_error = comp['execution_data']['qd'] - comp['execution_data']['qd_desired']
                vel_error_norm = np.linalg.norm(vel_error, axis=1)
                ax.plot(time, vel_error_norm, label=comp['name'])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity Error (rad/s)')
            ax.set_title('Velocity Tracking Error')
            ax.legend()
            ax.grid(True)
            
            # Torque comparison
            ax = axes[1, 0]
            for comp in analyzer.comparisons:
                time = comp['execution_data']['time']
                tau = comp['execution_data']['tau']
                tau_norm = np.linalg.norm(tau, axis=1)
                ax.plot(time, tau_norm, label=comp['name'])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Torque (Nm)')
            ax.set_title('Torque Requirements')
            ax.legend()
            ax.grid(True)
            
            # Summary bar chart
            ax = axes[1, 1]
            names = [comp['name'] for comp in analyzer.comparisons]
            rms_errors = [comp['metrics']['tracking_error']['position']['rms'] 
                         for comp in analyzer.comparisons]
            ax.bar(names, rms_errors)
            ax.set_ylabel('Position RMS Error (rad)')
            ax.set_title('Tracking Error Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / "comparison_plots.png", dpi=150)
            print(f"✓ Plots saved to {output_dir}/comparison_plots.png")
            plt.show()  # Display plots
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MuJoCo Physics Simulation Demo'
    )
    parser.add_argument('--robot', type=str, default='alicia',
                       choices=['alicia', 'bessica'],
                       help='Robot model')
    parser.add_argument('--mjcf', type=str,
                       help='MuJoCo MJCF file path')
    parser.add_argument('--control-modes', nargs='+',
                       choices=['position', 'velocity', 'torque'],
                       default=['position', 'velocity', 'torque'],
                       help='Control modes to test')
    parser.add_argument('--max-velocity', type=float,
                       help='Maximum velocity limit (rad/s) for servo control')
    parser.add_argument('--max-acceleration', type=float,
                       help='Maximum acceleration limit (rad/s²) for servo control')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate and display plots (default: True)')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                       help='Disable plot generation')
    parser.add_argument('--visualize', action='store_false',
                       help='Show MuJoCo viewer for trajectory visualization')
    
    args = parser.parse_args()
    
    # Find MJCF file
    if args.mjcf:
        mjcf_path = Path(args.mjcf)
    else:
        import synriard
        if args.robot == "alicia":
            mjcf_path = synriard.get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="mjcf")
        else:
            mjcf_path = synriard.get_model_path("Bessica_D", version="v1_0", variant="covered", model_format="mjcf")
        
    # Servo limits
    servo_limits = {}
    if args.max_velocity:
        servo_limits['max_velocity'] = args.max_velocity
    if args.max_acceleration:
        servo_limits['max_acceleration'] = args.max_acceleration
    
    # Run comparison
    run_comparison(
        str(mjcf_path),
        control_modes=args.control_modes,
        servo_limits=servo_limits if servo_limits else None,
        plot=args.plot,
        visualize=args.visualize
    )

