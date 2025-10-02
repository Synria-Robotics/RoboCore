# OmegaConf Configuration Examples

This directory contains example YAML configuration files for RoboCore.

## Quick Start

```bash
# Use default configuration (Alicia-D robot, NumPy backend, CPU)
python examples/demo_with_config.py

# Load GPU-optimized configuration
python examples/demo_with_config.py --config robocore/configs/gpu_config.yaml

# Load Bessica robot configuration
python examples/demo_with_config.py --config robocore/configs/bessica_config.yaml

# Override specific parameters
python examples/demo_with_config.py \
    --config robocore/configs/default.yaml \
    robot.end_link=Link7 \
    kinematics.ik.solver.max_iterations=200
```

## Configuration Files

### `default.yaml`
Default configuration for Alicia-D robot with NumPy backend.

**Use cases:**
- Development and testing
- CPU-only systems
- Single-sample computations

**Key settings:**
- Robot: Alicia-D v5.4 (6-DOF)
- Backend: NumPy
- Device: CPU

### `gpu_config.yaml`
GPU-optimized configuration for batch processing.

**Use cases:**
- High-performance batch FK/IK/Jacobian
- GPU workstations (CUDA)
- Large-scale simulations

**Key settings:**
- Backend: PyTorch
- Device: CUDA (cuda:0)
- Batch size: 1000
- All kinematics use torch backend

### `bessica_config.yaml`
Configuration for Bessica-D dual-arm robot.

**Use cases:**
- Dual-arm manipulation
- 7-DOF arm kinematics
- Bessica robot projects

**Key settings:**
- Robot: Bessica-D v1.0 (7-DOF dual-arm)
- End link: `left_arm_gripper_left_finger`
- Backend: NumPy (for compatibility)

## Creating Custom Configurations

### Method 1: Copy and Edit YAML

```bash
# Copy a template
cp robocore/configs/default.yaml my_config.yaml

# Edit with your settings
nano my_config.yaml

# Use it
python examples/demo_with_config.py --config my_config.yaml
```

### Method 2: Python API

```python
from robocore.configs import ConfigManager

# Load base config
config = ConfigManager('robocore/configs/default.yaml')

# Update values
config.update({
    'robot.end_link': 'my_end_effector',
    'kinematics.ik.solver.max_iterations': 200,
    'compute.device': 'cuda:1'
})

# Save to file
config.save('my_custom_config.yaml')
```

### Method 3: From Scratch

Create a new YAML file:

```yaml
# my_robot_config.yaml
robot:
  urdf_path: path/to/my_robot.urdf
  end_link: my_end_effector
  base_link: null

kinematics:
  fk_backend: torch
  jacobian_backend: torch
  jacobian_method: analytic
  
  ik:
    method: dls
    backend: torch
    solver:
      max_iterations: 150
      position_tolerance: 1.0e-5
      orientation_tolerance: 1.0e-5
      damping: 0.02

  compute:
    backend: torch
    device: cuda:1
    dtype: float32
    batch_size: 500
    use_jit: false

verbose: true
log_level: INFO
seed: 42
```

## Configuration Schema

See full schema in `robocore/configs/schemas.py`:

- `RobotConfig`: URDF path, end link, base link
- `SolverConfig`: IK solver parameters (max iterations, tolerances, damping)
- `IKConfig`: IK method, backend, solver config
- `ComputeConfig`: Backend, device, dtype, batch size, JIT
- `KinematicsConfig`: FK/IK/Jacobian settings
- `RoboCoreConfig`: Top-level configuration

## Parameters Reference

### Robot Settings

```yaml
robot:
  urdf_path: "path/to/robot.urdf"    # Required
  end_link: "tool0"                  # End-effector link name
  base_link: null                    # Optional base link (default: URDF root)
```

### IK Solver Settings

```yaml
kinematics:
  ik:
    method: "dls"                    # dls | jacobian_transpose | pseudoinverse
    backend: "numpy"                 # numpy | torch
    solver:
      max_iterations: 100            # Max IK iterations
      position_tolerance: 1.0e-4     # Position error threshold (m)
      orientation_tolerance: 1.0e-4  # Orientation error threshold (rad)
      damping: 0.01                  # DLS damping factor
      step_size: 1.0e-6              # Numerical Jacobian step size
```

### Compute Settings

```yaml
compute:
  backend: "numpy"                   # numpy | torch
  device: "cpu"                      # cpu | cuda:0 | cuda:1 | mps
  dtype: "float32"                   # float32 | float64
  batch_size: 100                    # Batch size for parallel operations
  use_jit: false                     # Enable PyTorch JIT compilation
```

### Jacobian Settings

```yaml
kinematics:
  jacobian_method: "analytic"        # analytic | numeric | autograd
  jacobian_backend: "numpy"          # numpy | torch
```

## Backend and Device Selection

### NumPy Backend (CPU)
- **Pros**: Fast for single samples, no GPU required, widely compatible
- **Cons**: No batch parallelism, slower for large batches
- **Use when**: Single-sample computation, CPU-only systems

```yaml
kinematics:
  fk_backend: numpy
  ik.backend: numpy
  jacobian_backend: numpy
compute:
  backend: numpy
  device: cpu
```

### PyTorch Backend (CPU)
- **Pros**: Batch parallelism, unified API
- **Cons**: Slower than NumPy for single samples
- **Use when**: Batch processing on CPU

```yaml
kinematics:
  fk_backend: torch
  ik.backend: torch
  jacobian_backend: torch
compute:
  backend: torch
  device: cpu
  batch_size: 100
```

### PyTorch Backend (GPU)
- **Pros**: 10-50x faster for batches >500, massive parallelism
- **Cons**: Requires CUDA/MPS, overhead for small batches
- **Use when**: Large batch processing, GPU available

```yaml
kinematics:
  fk_backend: torch
  ik.backend: torch
  jacobian_backend: torch
compute:
  backend: torch
  device: cuda:0  # or cuda:1, cuda:2, mps
  batch_size: 1000
  dtype: float32
```

## Tips

1. **Start with `default.yaml`**: Copy and modify for your needs
2. **Use GPU config for batches**: 10-50x speedup for batch_size >500
3. **Adjust tolerances**: Tighter tolerances = higher accuracy but slower
4. **Set random seed**: For reproducible results
5. **Version control configs**: Track configuration changes in git
6. **Environment-specific configs**: Create `dev.yaml`, `prod.yaml`, etc.

## Troubleshooting

### Config file not found
```bash
# Use absolute path or path relative to project root
python demo_with_config.py --config robocore/configs/default.yaml
```

### CUDA device not available
```yaml
# Check device index matches your system
compute:
  device: cuda:0  # Change to cuda:1, cuda:2, etc.
```

Use `nvidia-smi` to list GPUs:
```bash
nvidia-smi --list-gpus
```

### Type validation errors
```yaml
# Ensure correct types
max_iterations: 100    # int, not "100"
damping: 0.01          # float, not 0.01e0
verbose: true          # bool, not "true"
```

## Documentation

For full documentation, see:
- `/docs/CONFIGURATION_GUIDE.md` - Complete configuration guide
- `robocore/configs/schemas.py` - Configuration schemas
- `robocore/configs/config_manager.py` - ConfigManager API

## Examples

```bash
# Example 1: Default configuration
python examples/demo_with_config.py

# Example 2: GPU configuration with random joints
python examples/demo_with_config.py \
    --config robocore/configs/gpu_config.yaml \
    --random --seed 42

# Example 3: Bessica robot with custom joints
python examples/demo_with_config.py \
    --config robocore/configs/bessica_config.yaml \
    --joints 0.5 -0.3 1.2 0.8 -0.5 1.5 0.0

# Example 4: Override specific parameters
python examples/demo_with_config.py \
    --config robocore/configs/default.yaml \
    robot.end_link=Link7 \
    kinematics.ik.solver.max_iterations=200 \
    compute.device=cpu

# Example 5: Save modified configuration
python examples/demo_with_config.py \
    --config robocore/configs/default.yaml \
    --save-config my_config.yaml
```

---

**For more examples and API reference, see the main [Configuration Guide](/docs/CONFIGURATION_GUIDE.md).**
