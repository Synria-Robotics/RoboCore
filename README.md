# RoboCore

A high-performance robotics kinematics and planning library with GPU acceleration support.

## Features

- ✅ **## Documentation

- 📘 [Trajectory Planning Guide](docs/TRAJECTORY_PLANNING_GUIDE.md)
- 🌐 [Workspace Analysis Guide](docs/WORKSPACE_ANALYSIS_GUIDE.md)
- 🎯 [Workspace-Constrained Planning Guide](docs/WORKSPACE_TRAJECTORY_GUIDE.md)
- 🎬 [MuJoCo Visualization Guide](docs/MUJOCO_VISUALIZATION_GUIDE.md)
- 🚀 [GPU Usage Guide](docs/GPU_USAGE_GUIDE.md)
- ⚙️ [Configuration Guide](docs/CONFIGURATION_GUIDE.md)
- 🔧 [Batch Parallel Technical Details](docs/BATCH_PARALLEL_GPU.md) Kinematics (FK)**: NumPy and PyTorch backends with batch parallel support
- ✅ **Inverse Kinematics (IK)**: Damped Least Squares, Jacobian-based solvers  
- ✅ **Jacobian Computation**: Geometric and analytic Jacobians
- ✅ **Singularity Analysis**: Condition number, manipulability, singular value analysis
- ✅ **Workspace Analysis**: Reachable/dexterous workspace, volume estimation, density analysis
- ✅ **Workspace-Constrained Planning**: Trajectory planning within workspace bounds and safe regions
- ✅ **MuJoCo Integration**: Real-time 3D visualization with interactive controls and video export
- ✅ **GPU Acceleration**: True batch parallel operations on CUDA and Apple Silicon (MPS)
- ✅ **Trajectory Planning**: Joint/Cartesian space trajectories with velocity profiles
- ✅ **Configuration Management**: OmegaConf-based parameter system
- ✅ **Quaternion Support**: Standard xyzw order with proper handling

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics

# Load robot model
model = RobotModel('robot.urdf')

# Forward kinematics
q = np.array([0.0, 0.5, -0.3, 0.0, 0.5, 0.0])
T = forward_kinematics(model, q, backend='numpy', return_end=True)

# Inverse kinematics
result = inverse_kinematics(model, T, q_init=np.zeros(6), method='dls')
print(f"IK Solution: {result['q']}")
```

### Trajectory Planning

```python
from robocore.planning.trajectory import quintic_polynomial_trajectory

# Generate smooth trajectory
q_start = np.zeros(6)
q_end = np.array([1.0, 0.5, -0.3, 0.0, 0.5, 0.0])

t, q, qd, qdd = quintic_polynomial_trajectory(
    q_start, q_end, duration=2.0, num_points=100
)
```

### Workspace Analysis

```python
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer

# Create analyzer
analyzer = WorkspaceAnalyzer(model)

# Compute reachable workspace
points = analyzer.compute_reachable_workspace(num_samples=10000)

# Get workspace bounds and volume
bounds = analyzer.get_workspace_bounds(points)
volume = analyzer.estimate_workspace_volume(points)

print(f"Workspace volume: {volume:.4f} m³")
```

### Workspace-Constrained Trajectory Planning

```python
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
from robocore.planning.trajectory import multi_waypoint_trajectory

# Analyze workspace
analyzer = WorkspaceAnalyzer(model)
reachable_points = analyzer.compute_reachable_workspace(num_samples=5000)
bounds = analyzer.get_workspace_bounds(reachable_points)

# Find safe regions (avoid singularities)
safe_points = analyzer.find_singularity_free_regions(
    num_samples=5000,
    manipulability_threshold=0.01
)

# Generate targets within safe workspace
targets = []
safe_bounds = analyzer.get_workspace_bounds(safe_points)
for i in range(3):
    target = np.array([
        np.random.uniform(safe_bounds['x'][0], safe_bounds['x'][1]),
        np.random.uniform(safe_bounds['y'][0], safe_bounds['y'][1]),
        np.random.uniform(safe_bounds['z'][0], safe_bounds['z'][1])
    ])
    targets.append(target)

# Verify reachability
for i, target in enumerate(targets):
    reachable = analyzer.check_point_in_workspace(target, reachable_points)
    print(f"Target {i}: {'✓ Reachable' if reachable else '✗ Unreachable'}")
```

### GPU Acceleration

```python
from robocore.kinematics.fk_utils.batch_fk_torch import batch_forward_kinematics_torch
import torch

# Generate batch of configurations
q_batch = torch.randn(1000, 6, device='cuda:0')

# Batch FK on GPU (10-50x faster than NumPy)
T_batch = batch_forward_kinematics_torch(model, q_batch, end_link='tool0')
```

## Documentation

- 📘 [Trajectory Planning Guide](docs/TRAJECTORY_PLANNING_GUIDE.md)
- � [Workspace Analysis Guide](docs/WORKSPACE_ANALYSIS_GUIDE.md)
- �🚀 [GPU Usage Guide](docs/GPU_USAGE_GUIDE.md)
- ⚙️ [Configuration Guide](docs/CONFIGURATION_GUIDE.md)
- 🔧 [Batch Parallel Technical Details](docs/BATCH_PARALLEL_GPU.md)

## Examples

```bash
# Basic FK/IK/Jacobian demo
python examples/demo_fk_ik_jacobian.py --robot bessica --arm left

# Trajectory planning demo
python examples/demo_trajectory.py --robot bessica --arm left --plot

# Workspace analysis demo
python examples/demo_workspace.py --robot bessica --arm left --samples 10000 --visualize

# Workspace-constrained trajectory planning
python examples/demo_workspace_trajectory.py --robot bessica --arm left --visualize

# GPU benchmark (requires CUDA or MPS)
python examples/benchmark_parallel_fk_ik.py --device cuda:0 --batch-sizes 100 500 1000
```

## Supported Robots

- **Alicia-D v5.4**: 6-DOF manipulator
- **Bessica-D v1.0**: 7-DOF dual-arm robot

## Performance

| Operation | NumPy (CPU) | PyTorch (CPU) | PyTorch (CUDA) | Speedup |
|-----------|-------------|---------------|----------------|---------|
| FK (batch=1000) | 1.00x | 0.95x | **45x** | 45x |
| IK (batch=500) | 1.00x | 1.05x | **28x** | 28x |
| Jacobian (batch=1000) | 1.00x | 0.98x | **52x** | 52x |

*Tested on NVIDIA RTX 3090*

## License

MIT License

## Citation

If you use RoboCore in your research, please cite:

```bibtex
@software{robocore2025,
  title = {RoboCore: High-Performance Robotics Kinematics and Planning},
  author = {RoboCore Team},
  year = {2025},
  url = {https://github.com/yourusername/RoboCore}
}
```

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
