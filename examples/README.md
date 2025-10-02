# RoboCore Examples

This directory contains demonstration scripts showcasing RoboCore functionality.

## Quick Start

### Basic Usage
```bash
python examples/demo_basic.py
```
Shows the simplest FK and IK usage with the unified API.

## Examples Overview

### 1. `demo_basic.py` - Getting Started
- **Purpose**: Introduction to forward and inverse kinematics
- **Features**:
  - Load robot model from URDF
  - Compute forward kinematics
  - Solve inverse kinematics
  - Verify FK/IK closure
- **Usage**: `python examples/demo_basic.py`

### 2. `demo_ik.py` - Comprehensive IK Comparison
- **Purpose**: Compare IK methods and backends (NumPy vs PyTorch)
- **Features**:
  - Multiple IK methods: pinv, dls, transpose
  - Backend comparison: NumPy vs PyTorch
  - Multi-start support
  - Statistical analysis (success rate, convergence, timing)
- **Usage**:
  ```bash
  python examples/demo_ik.py --samples 10 --methods pinv dls
  python examples/demo_ik.py --backends torch --torch-device cpu
  ```

### 3. `demo_jacobian.py` - Jacobian Validation
- **Purpose**: Verify and compare Jacobian implementations
- **Features**:
  - Compare analytic vs numeric Jacobian
  - PyTorch: analytic vs numeric vs autograd
  - Accuracy metrics across random configurations
  - Performance benchmarking
- **Usage**:
  ```bash
  python examples/demo_jacobian.py --backend numpy
  python examples/demo_jacobian.py --backend torch --device cpu
  ```

### 4. `demo_dh.py` - Denavit-Hartenberg Parameters
- **Purpose**: DH parameter extraction and validation
- **Modes**:
  - `extract`: Extract standard and modified DH parameters
  - `compare`: Detailed comparison of URDF vs DH FK
  - `quality`: Statistical evaluation of DH approximation
- **Usage**:
  ```bash
  python examples/demo_dh.py --mode extract
  python examples/demo_dh.py --mode quality --samples 100
  ```

### 5. `demo_singularity.py` - Singularity Analysis
- **Purpose**: Analyze manipulability and singular configurations
- **Features**:
  - Manipulability index computation
  - Condition number analysis
  - Workspace sampling statistics
  - Singular configuration detection
- **Usage**: `python examples/demo_singularity.py`

### 6. `demo_closure.py` - FK/IK Closure Verification
- **Purpose**: Validate IK solutions through forward kinematics
- **Features**:
  - Random configuration sampling
  - FK → IK → FK roundtrip verification
  - Multiple IK method comparison
  - Convergence analysis
- **Usage**: `python examples/demo_closure.py`

### 7. `demo_trajectory.py` - Trajectory Planning
- **Purpose**: Generate smooth trajectories in joint and Cartesian space
- **Features**:
  - Joint space trajectories (quintic polynomial)
  - Multi-waypoint trajectory planning
  - Velocity profile generation (trapezoidal, S-curve)
  - Cartesian space trajectories (linear, circular)
  - Trajectory visualization
- **Usage**:
  ```bash
  python examples/demo_trajectory.py --robot bessica --arm left
  python examples/demo_trajectory.py --robot bessica --arm left --plot
  ```

### 8. `demo_workspace.py` - Workspace Analysis
- **Purpose**: Analyze robot reachable and dexterous workspace
- **Features**:
  - Reachable workspace computation (Monte Carlo, Grid, Sobol sampling)
  - Dexterous workspace analysis
  - Volume estimation (convex hull, voxel-based)
  - Density distribution analysis
  - Singularity-free region identification
  - 3D visualization
- **Usage**:
  ```bash
  python examples/demo_workspace.py --robot bessica --arm left --samples 5000
  python examples/demo_workspace.py --robot bessica --arm left --visualize --singularity
  ```

### 9. `demo_workspace_trajectory.py` - Workspace-Constrained Planning
- **Purpose**: Plan trajectories within workspace constraints
- **Features**:
  - Workspace analysis + trajectory planning integration
  - Target selection in reachable/safe regions
  - Singularity avoidance
  - Optimal workspace region identification
  - Trajectory validation against workspace bounds
  - 3D visualization of workspace + trajectory
- **Usage**:
  ```bash
  python examples/demo_workspace_trajectory.py --robot bessica --arm left
  python examples/demo_workspace_trajectory.py --robot bessica --arm left --visualize --waypoints 5
  python examples/demo_workspace_trajectory.py --robot bessica --arm left --target best --no-safe
  ```

### 10. `benchmark_performance.py` - Performance Metrics
- **Purpose**: Measure computational performance
- **Features**:
  - FK timing comparison
  - Jacobian computation benchmarks
  - IK solver performance
  - NumPy vs PyTorch comparison
- **Usage**:
  ```bash
  python examples/benchmark_performance.py
  python examples/benchmark_performance.py --torch-device cpu
  ```

## Common Arguments

Most examples support these common arguments:

- `--samples N`: Number of test samples (default varies by example)
- `--seed S`: Random seed for reproducibility
- `--backends [numpy|torch]`: Which backend(s) to test
- `--torch-device DEVICE`: PyTorch device (cpu, cuda, mps)
- `--torch-dtype DTYPE`: PyTorch dtype (float32, float64)

## Requirements

- **Minimum**: Python 3.8+, NumPy
- **Full features**: PyTorch (optional, for torch backend demos)

## Example Workflow

1. **Start with basics**:
   ```bash
   python examples/demo_basic.py
   ```

2. **Explore IK methods**:
   ```bash
   python examples/demo_ik.py --samples 20 --methods pinv dls transpose
   ```

3. **Validate Jacobians**:
   ```bash
   python examples/demo_jacobian.py --backend numpy --samples 50
   ```

4. **Check performance**:
   ```bash
   python examples/benchmark_performance.py
   ```

## Notes

- All examples use the Alicia-D robot model included in `robocore/assets/`
- Most demos output formatted tables and statistics
- Examples are self-contained and can be run independently
- For custom robots, modify the URDF path in each script

## Troubleshooting

**Import errors**: Ensure RoboCore is installed or run from repo root:
```bash
pip install -e .
# or
PYTHONPATH=/path/to/RoboCore python examples/demo_basic.py
```

**PyTorch not available**: Torch-related demos will skip gracefully if PyTorch is not installed.

**URDF not found**: Check that `robocore/assets/robot/urdf/` directory exists and contains robot models.
