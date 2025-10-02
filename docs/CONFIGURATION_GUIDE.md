# RoboCore Configuration Guide

## Overview

RoboCore uses **OmegaConf** for clean, type-safe configuration management. All complex parameters are managed through YAML configuration files or Python dataclass objects, eliminating the need to pass dozens of individual parameters to functions.

### Why Configuration Management?

**Before (Complex Parameter Passing):**
```python
result = inverse_kinematics(
    robot_model,
    target_pose,
    q_init,
    backend='numpy',
    method='dls',
    max_iters=100,
    pos_tol=1e-4,
    ori_tol=1e-4,
    damping=0.01,
    device='cpu',
    verbose=True
)
```

**After (Configuration-Based):**
```python
config = load_config('my_config.yaml')
result = inverse_kinematics(robot_model, target_pose, q_init, config=config)
```

### Performance Impact

⚡ **Zero performance overhead**: Configuration is loaded once at startup, then passed as a lightweight object. No impact on computation speed.

---

## Installation

```bash
pip install omegaconf
```

Or install RoboCore with config support:
```bash
pip install robocore[config]
```

---

## Quick Start

### 1. Using Default Configuration

```python
from robocore.configs import get_default_config

# Get default configuration
config = get_default_config()

# Use with robot model
from robocore.modeling.robot_model import RobotModel
robot = RobotModel(config.cfg.robot.urdf_path, end_link=config.cfg.robot.end_link)
```

### 2. Loading from YAML File

```python
from robocore.configs import ConfigManager

# Load from file
config = ConfigManager('robocore/configs/default.yaml')

# Access values
print(config.cfg.robot.urdf_path)
print(config.cfg.kinematics.ik.solver.max_iterations)
```

### 3. Updating Configuration

```python
# Update specific values
config.update({
    'robot.end_link': 'tool0',
    'kinematics.ik.solver.max_iterations': 200
})

# Or use dot notation
from omegaconf import OmegaConf
OmegaConf.update(config.cfg, 'robot.end_link', 'tool0')

# Save to file
config.save('my_custom_config.yaml')
```

---

## Configuration Structure

RoboCore configuration is hierarchical with the following main sections:

### Top-Level Structure

```yaml
robot:           # Robot model settings
kinematics:      # Kinematics computation settings
verbose:         # Logging verbosity
log_level:       # Logging level
seed:            # Random seed
```

### Robot Configuration

```yaml
robot:
  urdf_path: robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf
  end_link: tool0
  base_link: null  # Optional, defaults to URDF root
```

### Kinematics Configuration

```yaml
kinematics:
  fk_backend: numpy          # numpy or torch
  jacobian_backend: numpy    # numpy or torch
  jacobian_method: analytic  # analytic, numeric, autograd
  
  ik:
    method: dls              # dls, jacobian_transpose, pseudoinverse
    backend: numpy           # numpy or torch
    solver:
      max_iterations: 100
      position_tolerance: 1.0e-4
      orientation_tolerance: 1.0e-4
      damping: 0.01
      step_size: 1.0
```

### Compute Configuration

```yaml
compute:
  backend: numpy           # numpy or torch
  device: cpu              # cpu, cuda:0, cuda:1, mps
  dtype: float32           # float32, float64
  batch_size: 100
  use_jit: false           # Use TorchScript JIT compilation
```

---

## Pre-Configured Templates

RoboCore includes several pre-configured YAML files:

### 1. Default Configuration (`default.yaml`)
- Alicia-D robot (6-DOF)
- NumPy backend (CPU)
- Standard tolerance settings
- **Use case**: Development, testing, CPU-only systems

```bash
python demo_with_config.py --config robocore/configs/default.yaml
```

### 2. GPU Configuration (`gpu_config.yaml`)
- PyTorch backend
- CUDA GPU acceleration (cuda:0)
- Larger batch sizes (1000)
- **Use case**: High-performance batch processing, GPU workstations

```bash
python demo_with_config.py --config robocore/configs/gpu_config.yaml
```

### 3. Bessica Configuration (`bessica_config.yaml`)
- Bessica-D robot (7-DOF dual-arm)
- end_link: `left_arm_gripper_left_finger`
- **Use case**: Dual-arm manipulation tasks

```bash
python demo_with_config.py --config robocore/configs/bessica_config.yaml
```

---

## Using Configuration with Demo Scripts

### demo_with_config.py (Recommended)

Clean, configuration-focused demo:

```bash
# Use default config
python examples/demo_with_config.py

# Load from file
python examples/demo_with_config.py --config robocore/configs/gpu_config.yaml

# Override specific values
python examples/demo_with_config.py \
    --config robocore/configs/default.yaml \
    robot.end_link=Link7 \
    kinematics.ik.solver.max_iterations=200

# Print current configuration
python examples/demo_with_config.py --print-config

# Save configuration
python examples/demo_with_config.py \
    --config robocore/configs/default.yaml \
    --save-config my_config.yaml
```

### demo_fk_ik_jacobian.py (Legacy + Config Support)

Original demo with backward compatibility:

```bash
# Traditional usage (still works)
python examples/demo_fk_ik_jacobian.py --random --seed 42

# New config-based usage
python examples/demo_fk_ik_jacobian.py --config robocore/configs/default.yaml --random

# Config + CLI overrides (CLI takes precedence)
python examples/demo_fk_ik_jacobian.py \
    --config robocore/configs/default.yaml \
    --override robot.end_link=tool0 \
    --ik-iters 200
```

---

## Creating Custom Configurations

### Method 1: YAML File

Create `my_robot_config.yaml`:

```yaml
robot:
  urdf_path: path/to/my_robot.urdf
  end_link: my_end_effector
  base_link: my_base_link

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
  device: cuda:0
  dtype: float32
  batch_size: 500
  use_jit: true

verbose: true
log_level: INFO
seed: 12345
```

Load it:
```python
config = ConfigManager('my_robot_config.yaml')
```

### Method 2: Python Dictionary

```python
from robocore.configs import ConfigManager

config_dict = {
    'robot': {
        'urdf_path': 'path/to/robot.urdf',
        'end_link': 'tool0'
    },
    'kinematics': {
        'ik': {
            'solver': {
                'max_iterations': 200
            }
        }
    }
}

config = ConfigManager(config_dict)
```

### Method 3: Programmatic Dataclass

```python
from robocore.configs.schemas import RoboCoreConfig, RobotConfig, IKConfig, SolverConfig
from robocore.configs import ConfigManager

robot_cfg = RobotConfig(
    urdf_path='path/to/robot.urdf',
    end_link='tool0',
    base_link=None
)

solver_cfg = SolverConfig(
    max_iterations=200,
    position_tolerance=1e-5,
    orientation_tolerance=1e-5
)

ik_cfg = IKConfig(
    method='dls',
    backend='torch',
    solver=solver_cfg
)

# ... build full RoboCoreConfig

config = ConfigManager(robocore_config)
```

---

## Configuration Schemas

All configurations are validated using dataclasses with type hints:

### RobotConfig
```python
@dataclass
class RobotConfig:
    urdf_path: str                    # Path to URDF file
    end_link: str                     # End-effector link name
    base_link: Optional[str] = None   # Base link (default: URDF root)
```

### SolverConfig
```python
@dataclass
class SolverConfig:
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-4
    damping: float = 0.01
    step_size: float = 1.0
```

### IKConfig
```python
@dataclass
class IKConfig:
    method: Literal["dls", "jacobian_transpose", "pseudoinverse"] = "dls"
    backend: Literal["numpy", "torch"] = "numpy"
    solver: SolverConfig = field(default_factory=SolverConfig)
```

### ComputeConfig
```python
@dataclass
class ComputeConfig:
    backend: Literal["numpy", "torch"] = "numpy"
    device: str = "cpu"  # cpu, cuda:0, cuda:1, mps
    dtype: str = "float32"
    batch_size: int = 100
    use_jit: bool = False
```

Full schema in `robocore/configs/schemas.py`.

---

## Advanced Usage

### Merging Configurations

```python
# Load base config
base_config = ConfigManager('robocore/configs/default.yaml')

# Load overrides
override_dict = {'kinematics': {'ik': {'solver': {'max_iterations': 200}}}}
base_config.update(override_dict)
```

### Environment-Specific Configs

```python
import os

# Choose config based on environment
if os.environ.get('USE_GPU') == '1':
    config = ConfigManager('robocore/configs/gpu_config.yaml')
else:
    config = ConfigManager('robocore/configs/default.yaml')
```

### Config Validation

```python
config = ConfigManager('my_config.yaml')

# Check if URDF exists
urdf_path = Path(config.cfg.robot.urdf_path)
if not urdf_path.exists():
    raise FileNotFoundError(f"URDF not found: {urdf_path}")

# Check device availability
device = config.cfg.compute.device
if device.startswith('cuda'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available but requested: {device}")
```

---

## Best Practices

### 1. Use Configuration Files for Projects
Store project-specific settings in YAML files:
```
my_project/
├── configs/
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
└── main.py
```

### 2. Environment Variables for Secrets
Don't store paths or credentials directly:
```yaml
robot:
  urdf_path: ${oc.env:ROBOT_URDF_PATH}
```

### 3. Validate Early
Check configuration immediately after loading:
```python
config = ConfigManager('config.yaml')
assert Path(config.cfg.robot.urdf_path).exists()
assert config.cfg.kinematics.ik.solver.max_iterations > 0
```

### 4. Version Control Configs
- ✅ Commit: Template configs (`default.yaml`, `gpu_config.yaml`)
- ❌ Don't commit: Machine-specific configs with absolute paths
- Use `.gitignore`: `configs/local_*.yaml`

### 5. Document Custom Configs
Add comments to YAML files:
```yaml
kinematics:
  ik:
    solver:
      # Increased iterations for high-precision tasks
      max_iterations: 200
      # Tighter tolerances for assembly operations
      position_tolerance: 1.0e-5
```

---

## Migration from Legacy Code

### Before (Legacy)
```python
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics

robot = RobotModel(
    'robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf',
    end_link='tool0'
)

result = inverse_kinematics(
    robot,
    target_pose,
    q_init,
    backend='numpy',
    method='dls',
    max_iters=100,
    pos_tol=1e-4,
    ori_tol=1e-4
)
```

### After (Config-Based)
```python
from robocore.configs import ConfigManager
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics

# Load configuration
config = ConfigManager('robocore/configs/default.yaml')

# Create robot from config
robot = RobotModel(
    config.cfg.robot.urdf_path,
    end_link=config.cfg.robot.end_link
)

# IK with config (parameters extracted internally)
result = inverse_kinematics(
    robot,
    target_pose,
    q_init,
    config=config  # Much cleaner!
)
```

**Note**: Legacy parameter passing still works for backward compatibility. Config parameter is optional.

---

## Troubleshooting

### Issue: `ImportError: No module named 'omegaconf'`
**Solution**: Install OmegaConf
```bash
pip install omegaconf
```

### Issue: Config file not found
**Solution**: Use absolute paths or paths relative to project root
```python
from pathlib import Path
config_path = Path(__file__).parent / 'configs' / 'default.yaml'
config = ConfigManager(config_path)
```

### Issue: Type validation errors
**Solution**: Check YAML syntax and data types
```yaml
# ❌ Wrong (string instead of number)
max_iterations: "100"

# ✅ Correct
max_iterations: 100
```

### Issue: Config changes not reflected
**Solution**: Reload config or restart Python
```python
# Reload from file
config = ConfigManager('config.yaml')

# Or update in-place
config.update({'robot.end_link': 'new_link'})
```

---

## API Reference

### ConfigManager

```python
class ConfigManager:
    def __init__(self, config: Union[str, Path, Dict, DictConfig, None] = None)
    def save(self, path: Union[str, Path]) -> None
    def update(self, updates: Dict) -> None
    def get(self, key: str, default=None) -> Any
    def to_dict() -> Dict
    def to_yaml() -> str
```

### Helper Functions

```python
def load_config(path: Union[str, Path]) -> ConfigManager
def get_default_config() -> ConfigManager
```

---

## Examples

See:
- `examples/demo_with_config.py` - Clean config-based demo
- `examples/demo_fk_ik_jacobian.py` - Legacy demo with config support
- `robocore/configs/*.yaml` - Pre-configured templates

---

## Summary

✅ **Benefits of Configuration Management:**
- Clean, readable code
- Type-safe with dataclass schemas
- Easy to version control
- No performance overhead
- Supports multiple environments (dev, prod, GPU, CPU)
- Backward compatible with legacy code

🚀 **Get Started:**
```bash
python examples/demo_with_config.py
```

For more examples, see the `examples/` directory.
