# MuJoCo Trajectory Visualization Guide

## Overview

RoboCore integrates with MuJoCo physics simulator to provide real-time 3D visualization of robot trajectories. This feature allows you to:

- ✅ Visualize planned trajectories in realistic 3D environment
- ✅ Interactive camera controls (zoom, rotate, pan)
- ✅ Playback controls (play/pause, speed, step)
- ✅ Export trajectories to video files
- ✅ Real-time trajectory tracking

## Installation

### Install MuJoCo

```bash
pip install mujoco
```

### Optional: Install video export dependencies

```bash
pip install imageio imageio-ffmpeg
```

## Quick Start

### Basic Visualization

```bash
# Visualize Alicia robot trajectory
python examples/demo_mujoco_trajectory.py --robot alicia --samples 3000 --waypoints 4

# Visualize Bessica robot trajectory
python examples/demo_mujoco_trajectory.py --robot bessica --arm left --samples 3000 --waypoints 4
```

### With Custom Settings

```bash
# Slower playback speed
python examples/demo_mujoco_trajectory.py --robot alicia --speed 0.5

# More waypoints for complex trajectory
python examples/demo_mujoco_trajectory.py --robot alicia --waypoints 6 --samples 5000

# Disable loop
python examples/demo_mujoco_trajectory.py --robot alicia --no-loop
```

## Controls

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Space** | Play / Pause |
| **R** | Reset to start |
| **[** | Decrease playback speed |
| **]** | Increase playback speed |
| **,** | Step backward (one frame) |
| **.** | Step forward (one frame) |
| **L** | Toggle loop mode |
| **H** | Show help |
| **ESC** | Exit viewer |

### Mouse Controls

| Action | Control |
|--------|---------|
| **Rotate** | Left mouse button + drag |
| **Zoom** | Right mouse button + drag |
| **Pan** | Middle mouse button + drag |
| **Zoom** | Mouse scroll wheel |

## Command Line Options

```bash
python examples/demo_mujoco_trajectory.py [OPTIONS]
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | `bessica` | Robot model (alicia/bessica) |
| `--arm` | str | `left` | Arm selection for dual-arm robots |
| `--samples` | int | `3000` | Workspace sampling points |
| `--waypoints` | int | `4` | Number of trajectory waypoints |
| `--speed` | float | `1.0` | Playback speed (1.0 = real-time) |
| `--no-loop` | flag | - | Disable trajectory looping |
| `--export` | str | - | Export to video file (e.g., trajectory.mp4) |
| `--mjcf` | str | - | Custom MuJoCo MJCF file path |

## Usage Examples

### 1. Basic Trajectory Visualization

```python
from robocore.bridge.sim.mujoco.trajectory_visualizer import TrajectoryVisualizer
import numpy as np

# Load MuJoCo model
viz = TrajectoryVisualizer(mjcf_path='robot.xml')

# Load trajectory data
trajectory = {
    'q': q_trajectory,      # (N, nq) joint positions
    'time': time_array,     # (N,) time stamps
    'qd': qd_trajectory,    # (N, nq) velocities (optional)
    'qdd': qdd_trajectory   # (N, nq) accelerations (optional)
}

viz.load_trajectory(
    q_trajectory=trajectory['q'],
    time=trajectory['time'],
    qd_trajectory=trajectory['qd']
)

# Configure playback
viz.set_speed(1.0)
viz.loop = True

# Start visualization
viz.play()
viz.visualize()
```

### 2. Export to Video

```bash
# Export trajectory to MP4 video
python examples/demo_mujoco_trajectory.py --robot alicia --export output.mp4 --samples 5000
```

Or programmatically:

```python
# Export with custom settings
viz.export_video(
    output_path='trajectory.mp4',
    fps=30,
    width=1920,
    height=1080,
    camera_id=0  # Use specific camera
)
```

### 3. Custom Trajectory Visualization

```python
from robocore.modeling.robot_model import RobotModel
from robocore.planning.trajectory import multi_waypoint_trajectory
from robocore.bridge.sim.mujoco.trajectory_visualizer import visualize_trajectory

# Plan trajectory
model = RobotModel('robot.urdf')
waypoints = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.5, -0.3, 0.0, 0.5, 0.0],
    [0.5, 1.0, 0.2, 0.3, 0.1, 0.5]
])

t, q, qd, qdd = multi_waypoint_trajectory(
    waypoints,
    durations=2.0,
    num_points_per_segment=100
)

# Visualize
trajectory_data = {'q': q, 'time': t, 'qd': qd, 'qdd': qdd}
visualize_trajectory(
    trajectory_data,
    mjcf_path='robot.xml',
    playback_speed=1.0,
    loop=True
)
```

### 4. Workspace-Constrained Trajectory with MuJoCo

```python
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics

# Step 1: Analyze workspace
analyzer = WorkspaceAnalyzer(model)
reachable_points = analyzer.compute_reachable_workspace(num_samples=5000)
bounds = analyzer.get_workspace_bounds(reachable_points)

# Step 2: Generate targets in workspace
targets = []
for i in range(5):
    target = np.array([
        np.random.uniform(bounds['x'][0], bounds['x'][1]),
        np.random.uniform(bounds['y'][0], bounds['y'][1]),
        np.random.uniform(bounds['z'][0], bounds['z'][1])
    ])
    targets.append(target)

# Step 3: Solve IK for each target
waypoints = [q_start]
q_current = q_start.copy()

for target in targets:
    T_target = np.eye(4)
    T_target[:3, 3] = target
    
    result = inverse_kinematics(model, T_target, q_current, method='dls')
    if result['success']:
        waypoints.append(result['q'])
        q_current = result['q']

# Step 4: Generate smooth trajectory
t, q, qd, qdd = multi_waypoint_trajectory(np.array(waypoints), durations=1.0)

# Step 5: Visualize in MuJoCo
viz = TrajectoryVisualizer(mjcf_path='robot.xml')
viz.load_trajectory(q, time=t, qd_trajectory=qd)
viz.play()
viz.visualize()
```

## Platform-Specific Notes

### macOS

On macOS, MuJoCo's passive viewer requires `mjpython`. The visualizer automatically falls back to a simplified mode with:

- ✅ Full mouse controls (rotate, zoom, pan)
- ✅ Keyboard controls (play/pause, speed, etc.)
- ✅ Real-time trajectory playback
- ✅ Status overlay

To use the interactive viewer on macOS:

```bash
mjpython examples/demo_mujoco_trajectory.py --robot alicia
```

### Linux / Windows

The full passive viewer works out of the box:

```bash
python examples/demo_mujoco_trajectory.py --robot alicia
```

## Troubleshooting

### Issue 1: DOF Mismatch

**Problem**: `Trajectory DOF (X) != model DOF (Y)`

**Solution**: The visualizer automatically handles common cases:
- Gripper DOF (+1 or +2 joints)
- Dual-arm configurations

If you still encounter issues, manually extend the trajectory:

```python
# Example: Extend 6-DOF trajectory to 8-DOF (6 arm + 2 gripper)
q_full = np.zeros((len(q_trajectory), 8))
q_full[:, :6] = q_trajectory
trajectory['q'] = q_full
```

### Issue 2: MJCF File Not Found

**Problem**: Cannot find MuJoCo MJCF file

**Solution**:
1. Specify MJCF file explicitly:
   ```bash
   python examples/demo_mujoco_trajectory.py --mjcf path/to/robot.xml
   ```

2. Or let it use URDF directly (may have warnings but usually works):
   ```bash
   python examples/demo_mujoco_trajectory.py --robot alicia
   ```

### Issue 3: Missing Mesh Files

**Problem**: MuJoCo cannot load mesh files

**Solution**: Ensure mesh files are in the correct location relative to URDF/MJCF:
```
robocore/assets/robot/
├── urdf/
│   └── Robot_Name/
│       ├── robot.urdf
│       └── meshes/
├── meshes/
│   └── Robot_Name/
└── mjcf/
    └── Robot_Name/
        └── robot.xml
```

### Issue 4: Slow Performance

**Problem**: Low FPS during playback

**Solutions**:
- Reduce trajectory points: Use fewer `num_points_per_segment`
- Simplify visualization: Disable shadows, reflections in MJCF
- Lower screen resolution
- Use faster playback speed for preview

### Issue 5: Video Export Fails

**Problem**: Cannot export video

**Solution**: Install video codec dependencies:
```bash
pip install imageio imageio-ffmpeg
```

## Advanced Features

### Custom Camera Setup

```python
viz = TrajectoryVisualizer(mjcf_path='robot.xml')

# Access camera directly (in simplified mode)
# Camera settings are in _visualize_simple() method
# Default:
#   distance: 2.5
#   azimuth: 90
#   elevation: -15
#   lookat: [0, 0, 0.8]
```

### Multiple Trajectories

```python
# Visualize sequence of trajectories
trajectories = [traj1, traj2, traj3]

for i, traj in enumerate(trajectories):
    viz.load_trajectory(traj['q'], traj['time'])
    viz.play()
    print(f"Playing trajectory {i+1}/{len(trajectories)}")
    viz.visualize()  # Blocks until closed
```

### Real-Time Trajectory Generation

```python
import time

# Generate trajectory on-the-fly
for i in range(10):
    # Plan new trajectory
    waypoints = generate_random_waypoints(model, workspace_bounds)
    t, q, qd, qdd = multi_waypoint_trajectory(waypoints, durations=2.0)
    
    # Load and play
    viz.load_trajectory(q, time=t)
    viz.play()
    
    # Non-blocking visualization (requires threading)
    # viz.visualize() in separate thread
    
    time.sleep(5)  # Wait before next trajectory
```

## Performance Tips

1. **Optimize Sampling**: Use fewer workspace samples for faster planning
   ```bash
   --samples 1000  # Fast, lower accuracy
   --samples 5000  # Balanced
   --samples 10000 # Slow, high accuracy
   ```

2. **Trajectory Resolution**: Adjust points per segment
   ```python
   multi_waypoint_trajectory(waypoints, num_points_per_segment=50)  # Fast
   multi_waypoint_trajectory(waypoints, num_points_per_segment=100) # Smooth
   ```

3. **Playback Speed**: Use faster speed for quick preview
   ```bash
   --speed 2.0  # 2x speed
   --speed 5.0  # 5x speed
   ```

4. **Loop Mode**: Disable for single playthrough
   ```bash
   --no-loop
   ```

## Integration with Other Modules

### With Workspace Analysis

```bash
# The demo automatically integrates workspace analysis
python examples/demo_mujoco_trajectory.py --robot alicia --samples 5000
```

This performs:
1. Workspace analysis (reachable + safe regions)
2. Trajectory planning within safe workspace
3. MuJoCo visualization

### With Configuration System

```python
from robocore.configs.config_manager import ConfigManager

# Load configuration
config = ConfigManager('bessica_config.yaml')

# Use config for trajectory planning
viz = TrajectoryVisualizer(mjcf_path=config.simulation.mjcf_path)
# ... plan trajectory with config parameters
viz.load_trajectory(q, time=t)
viz.visualize()
```

## Related Documentation

- [Trajectory Planning Guide](TRAJECTORY_PLANNING_GUIDE.md)
- [Workspace Analysis Guide](WORKSPACE_ANALYSIS_GUIDE.md)
- [Workspace-Constrained Planning](WORKSPACE_TRAJECTORY_GUIDE.md)
- [MuJoCo Official Documentation](https://mujoco.readthedocs.io/)

## Examples Summary

| Example | Description | Command |
|---------|-------------|---------|
| **Basic** | Simple trajectory visualization | `python examples/demo_mujoco_trajectory.py` |
| **Custom Robot** | Specify robot model | `--robot alicia` |
| **Slow Motion** | Reduced playback speed | `--speed 0.5` |
| **High Quality** | More workspace samples | `--samples 10000` |
| **Complex Path** | More waypoints | `--waypoints 6` |
| **Video Export** | Save to file | `--export output.mp4` |
| **Custom MJCF** | Use specific model file | `--mjcf path/to/robot.xml` |

## Conclusion

MuJoCo visualization provides powerful tools for:
- Validating trajectory planning algorithms
- Debugging robot motion
- Creating presentation videos
- Interactive exploration of workspace

The integration with workspace analysis and trajectory planning makes it a complete solution for robot motion planning and visualization.
