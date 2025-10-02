# Workspace Analysis Guide

## Overview

RoboCore's workspace analysis module provides comprehensive tools for understanding and analyzing robot workspace characteristics:

- **Reachable Workspace**: All positions the end-effector can reach
- **Dexterous Workspace**: Positions reachable with multiple orientations
- **Workspace Volume**: Quantitative workspace size estimation
- **Singularity-Free Regions**: Safe operating zones
- **Workspace Density**: Distribution analysis
- **Visualization**: 3D workspace plotting

---

## Quick Start

```python
from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer

# Load robot
model = RobotModel('robot.urdf')

# Create analyzer
analyzer = WorkspaceAnalyzer(model)

# Compute reachable workspace
points = analyzer.compute_reachable_workspace(num_samples=10000)

# Get workspace bounds
bounds = analyzer.get_workspace_bounds(points)
print(f"Workspace: X={bounds['x']}, Y={bounds['y']}, Z={bounds['z']}")

# Estimate volume
volume = analyzer.estimate_workspace_volume(points)
print(f"Volume: {volume:.4f} m³")
```

---

## Reachable Workspace

The **reachable workspace** is the set of all points in 3D space that the end-effector can reach with at least one orientation.

### Computation Methods

#### 1. Monte Carlo Sampling (Default)

Random uniform sampling in joint space:

```python
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='monte_carlo',
    seed=42  # For reproducibility
)
```

**Pros**: Fast, simple, unbiased
**Cons**: May miss small regions
**Best for**: General workspace analysis

---

#### 2. Grid Sampling

Uniform grid in joint space:

```python
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='grid'
)
```

**Pros**: Systematic coverage, repeatable
**Cons**: Slower, curse of dimensionality
**Best for**: Detailed analysis, low DOF robots

---

#### 3. Sobol Sequence (Quasi-Random)

Low-discrepancy sequence for better coverage:

```python
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='sobol'
)
```

**Pros**: Better coverage than random, fewer samples needed
**Cons**: Requires scipy.stats.qmc
**Best for**: High-quality analysis with fewer samples

---

### Custom Joint Limits

Specify custom joint limits:

```python
import numpy as np

# Define limits: [min, max] for each joint
q_limits = np.array([
    [-np.pi, np.pi],      # Joint 0
    [-np.pi/2, np.pi/2],  # Joint 1
    [-np.pi, np.pi],      # Joint 2
    # ... more joints
])

points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    q_limits=q_limits
)
```

---

### Parallel Processing

For large datasets, use parallel processing:

```python
points = analyzer.compute_reachable_workspace(
    num_samples=50000,
    use_parallel=True,
    num_workers=8  # Number of CPU cores
)
```

**Note**: Parallel processing is beneficial for `num_samples > 10000`

---

## Workspace Bounds

Get workspace bounding box:

```python
bounds = analyzer.get_workspace_bounds(points)

print(f"X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
print(f"Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
print(f"Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")

# Calculate ranges
x_range = bounds['x'][1] - bounds['x'][0]
y_range = bounds['y'][1] - bounds['y'][0]
z_range = bounds['z'][1] - bounds['z'][0]

print(f"Workspace dimensions: {x_range:.3f} × {y_range:.3f} × {z_range:.3f} m")
```

---

## Workspace Volume

Estimate workspace volume using different methods:

### 1. Convex Hull (Upper Bound)

```python
volume = analyzer.estimate_workspace_volume(
    points,
    method='convex_hull'
)
```

- **Pros**: Fast, deterministic
- **Cons**: Overestimates (includes holes)
- **Result**: Upper bound on true volume

### 2. Voxel-Based (More Accurate)

```python
volume = analyzer.estimate_workspace_volume(
    points,
    method='voxel'
)
```

- **Pros**: More accurate, handles complex shapes
- **Cons**: Depends on grid resolution
- **Result**: Better estimate of actual volume

### Comparison

```python
vol_convex = analyzer.estimate_workspace_volume(points, method='convex_hull')
vol_voxel = analyzer.estimate_workspace_volume(points, method='voxel')

ratio = vol_voxel / vol_convex
print(f"Voxel/Convex ratio: {ratio:.2%}")
# Low ratio (< 50%) indicates complex, non-convex workspace
```

---

## Reachability Testing

Check if specific points are reachable:

```python
target = np.array([0.5, 0.2, 0.3])

reachable = analyzer.check_point_in_workspace(
    target,
    points,
    tolerance=0.05  # 5cm tolerance
)

if reachable:
    print("✓ Target is reachable")
else:
    print("✗ Target is NOT reachable")
```

**Use case**: Validate task positions before planning

---

## Dexterous Workspace

The **dexterous workspace** contains points reachable with multiple different orientations.

```python
dex_points = analyzer.compute_dexterous_workspace(
    num_samples=20000,
    num_orientations=8,  # Minimum orientations required
    tolerance=0.01       # Position grouping tolerance
)

print(f"Dexterous points: {len(dex_points)}")

# Compare with reachable workspace
reachable_points = analyzer._workspace_cache['reachable_points']
ratio = len(dex_points) / len(reachable_points)
print(f"Dexterous/Reachable: {ratio:.2%}")
```

**Interpretation**:
- High ratio (> 50%): Good manipulability throughout workspace
- Low ratio (< 20%): Limited dexterity, many positions have few orientations
- Zero points: Increase `num_samples` or lower `num_orientations`

**Applications**:
- Assembly tasks requiring specific orientations
- Object manipulation from multiple angles
- Workspace optimization

---

## Workspace Density

Analyze point distribution across workspace:

```python
density, (x, y, z) = analyzer.compute_workspace_density(
    points,
    grid_resolution=20
)

# Find densest region
max_idx = np.unravel_index(np.argmax(density), density.shape)
densest_pos = (x[max_idx[0]], y[max_idx[1]], z[max_idx[2]])

print(f"Densest region: {densest_pos}")
print(f"Density: {density[max_idx]} points/voxel")
```

**Applications**:
- Identify optimal operating regions
- Find workspace "sweet spots"
- Workspace-aware task planning

---

## Singularity-Free Workspace

Find regions without kinematic singularities:

```python
safe_points = analyzer.find_singularity_free_regions(
    num_samples=10000,
    manipulability_threshold=0.01  # Minimum manipulability
)

print(f"Safe points: {len(safe_points)}")
print(f"Safety ratio: {len(safe_points)/10000:.2%}")
```

**Manipulability Threshold**:
- Higher threshold → safer, but smaller workspace
- Lower threshold → larger workspace, but less safe
- Typical range: 0.001 - 0.1

**Use case**: 
- Define safe operating zones
- Avoid singular configurations
- High-precision tasks

---

## Workspace Visualization

### Basic Visualization

```python
analyzer.visualize_workspace(
    points,
    show_bounds=True,
    alpha=0.3
)
```

### Advanced Visualization

```python
# Visualize with density heatmap
analyzer.visualize_workspace(
    points,
    show_bounds=True,
    show_density=True,  # Show high-density regions
    alpha=0.2,
    figsize=(12, 10)
)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# Reachable workspace
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
           c='b', marker='.', alpha=0.1, s=1)
ax1.set_title('Reachable Workspace')

# Dexterous workspace
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(dex_points[:, 0], dex_points[:, 1], dex_points[:, 2],
           c='r', marker='.', alpha=0.3, s=2)
ax2.set_title('Dexterous Workspace')

plt.show()
```

---

## Workspace Comparison

Compare workspaces of two robots:

```python
from robocore.analysis.workspace_analyzer import analyze_workspace_comparison

model_6dof = RobotModel('robot_6dof.urdf')
model_7dof = RobotModel('robot_7dof.urdf')

comparison = analyze_workspace_comparison(
    model_6dof,
    model_7dof,
    num_samples=10000,
    names=("6-DOF Robot", "7-DOF Robot")
)

print(f"Volumes: {comparison['volumes']}")
print(f"Volume ratio: {comparison['volume_ratio']:.2f}")
```

**Applications**:
- Robot selection for tasks
- Design validation
- Performance benchmarking

---

## Performance Considerations

### Sample Size vs. Accuracy

| Samples | Accuracy | Time | Recommended For |
|---------|----------|------|-----------------|
| 1,000 | Low | < 1s | Quick preview |
| 5,000 | Medium | ~3s | Interactive analysis |
| 10,000 | Good | ~10s | Standard analysis |
| 50,000 | High | ~60s | Publication quality |
| 100,000+ | Very High | Minutes | Detailed research |

### GPU Acceleration

For large-scale analysis, use batch FK on GPU:

```python
import torch
from robocore.kinematics.fk_utils.batch_fk_torch import batch_forward_kinematics_torch

# Generate many samples
num_samples = 100000
q_samples = np.random.uniform(-np.pi, np.pi, (num_samples, dof))

# Batch FK on GPU (much faster!)
q_torch = torch.tensor(q_samples, dtype=torch.float32, device='cuda:0')
T_batch = batch_forward_kinematics_torch(model, q_torch, end_link='tool0')

# Extract positions
points_gpu = T_batch[:, :3, 3].cpu().numpy()
```

**Speedup**: 10-50x faster for large batches

---

## Example Applications

### 1. Task Feasibility Check

```python
# Check if task positions are reachable
task_positions = np.array([
    [0.5, 0.2, 0.3],
    [0.6, 0.1, 0.4],
    [0.4, 0.3, 0.5]
])

points = analyzer.compute_reachable_workspace(num_samples=10000)

for i, pos in enumerate(task_positions):
    reachable = analyzer.check_point_in_workspace(pos, points, tolerance=0.05)
    print(f"Position {i+1}: {'✓ OK' if reachable else '✗ FAIL'}")
```

### 2. Workspace-Aware Robot Placement

```python
# Find optimal robot base position
candidate_positions = [...]  # List of potential base positions

best_position = None
best_volume = 0

for base_pos in candidate_positions:
    # Update robot base position
    # ... (modify URDF or transform)
    
    # Compute workspace
    points = analyzer.compute_reachable_workspace(num_samples=5000)
    volume = analyzer.estimate_workspace_volume(points)
    
    if volume > best_volume:
        best_volume = volume
        best_position = base_pos

print(f"Optimal base position: {best_position}")
```

### 3. Safety Zone Definition

```python
# Define safe operating zone (no singularities)
safe_points = analyzer.find_singularity_free_regions(
    num_samples=20000,
    manipulability_threshold=0.05
)

# Get safe bounds
safe_bounds = analyzer.get_workspace_bounds(safe_points)

# Use for trajectory planning constraints
print("Safe workspace limits:")
print(f"  X: {safe_bounds['x']}")
print(f"  Y: {safe_bounds['y']}")
print(f"  Z: {safe_bounds['z']}")
```

---

## Demo Script

Run the comprehensive workspace demo:

```bash
# Basic demo (5000 samples)
python examples/demo_workspace.py --robot bessica --arm left --samples 5000

# High-resolution analysis
python examples/demo_workspace.py --robot bessica --arm left --samples 20000

# With 3D visualization
python examples/demo_workspace.py --robot bessica --arm left --samples 10000 --visualize

# Include dexterous workspace
python examples/demo_workspace.py --robot bessica --arm left --dexterous

# Include singularity analysis
python examples/demo_workspace.py --robot bessica --arm left --singularity

# Full analysis
python examples/demo_workspace.py --robot bessica --arm left --samples 20000 --dexterous --singularity --visualize
```

---

## API Reference

### WorkspaceAnalyzer

| Method | Description | Returns |
|--------|-------------|---------|
| `compute_reachable_workspace()` | Compute reachable points | `np.ndarray (N, 3)` |
| `compute_dexterous_workspace()` | Compute dexterous points | `np.ndarray (M, 3)` |
| `get_workspace_bounds()` | Get bounding box | `dict` |
| `estimate_workspace_volume()` | Estimate volume | `float` |
| `check_point_in_workspace()` | Check reachability | `bool` |
| `compute_workspace_density()` | Compute density grid | `tuple` |
| `find_singularity_free_regions()` | Find safe zones | `np.ndarray` |
| `visualize_workspace()` | 3D visualization | None |

### Standalone Functions

| Function | Description |
|----------|-------------|
| `analyze_workspace_comparison()` | Compare two robot workspaces |

---

## Best Practices

### 1. Choose Appropriate Sample Size

- Start small (1000-5000) for quick iteration
- Use 10000+ for analysis and reporting
- Use 50000+ for research/publication

### 2. Select Sampling Method

- **Monte Carlo**: General purpose, fast
- **Grid**: Systematic, repeatable
- **Sobol**: Best coverage-to-sample ratio

### 3. Validate Results

```python
# Check sampling quality
points = analyzer.compute_reachable_workspace(num_samples=10000)

# Should have high percentage of unique points
unique_ratio = len(np.unique(points, axis=0)) / len(points)
print(f"Unique points: {unique_ratio:.2%}")
# Target: > 95%
```

### 4. Cache Results

```python
# Compute once, use many times
points = analyzer.compute_reachable_workspace(num_samples=20000)

# All subsequent analyses use cached data
bounds = analyzer.get_workspace_bounds()  # Uses cached points
volume = analyzer.estimate_workspace_volume()  # Uses cached points
density, _ = analyzer.compute_workspace_density()  # Uses cached points
```

### 5. Consider Joint Limits

Always specify realistic joint limits:

```python
# Example: Collaborative robot with limited range
q_limits = np.array([
    [-np.pi, np.pi],       # Shoulder pan
    [-np.pi/2, np.pi/2],   # Shoulder lift (limited)
    [-np.pi, np.pi],       # Elbow
    # ...
])

points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    q_limits=q_limits
)
```

---

## References

1. **Robot Manipulators: Mathematics, Programming, and Control**
   - Richard Paul (MIT Press, 1981)
   - Chapter on workspace analysis

2. **Modern Robotics**
   - Lynch & Park (Cambridge, 2017)
   - Workspace and manipulability theory

3. **Workspace Analysis of Robots**
   - Kumar & Waldron (IEEE Transactions on Robotics, 1981)
   - Classic reference on workspace computation

---

## Next Steps

After workspace analysis, explore:

1. **Trajectory Planning**: Plan paths within workspace
2. **Collision Detection**: Check for workspace obstacles
3. **Optimal Placement**: Find best robot base position
4. **Task Planning**: Workspace-aware task sequencing
5. **Multi-Robot Coordination**: Shared workspace analysis

See `robocore/planning/` and `robocore/analysis/` for more tools.
