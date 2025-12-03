![](./imgs/logo.jpeg)

# RoboCore: Unified High-Throughput Robotics Library

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Developed by [Synria Robotics Co., Ltd.](https://synriarobotics.ai)** 🤖

---

## 🔥 Features & Roadmap

| Module | Features | Status |
|--------|----------|--------|
| **Kinematics** | Forward Kinematics (NumPy/PyTorch Batch) | ✅ |
| | Inverse Kinematics (NumPy/PyTorch Batch) | ✅ |
| | Jacobian (NumPy/PyTorch Batch) | ✅ |
| | Bimanual FK (indep/relative/mirror) | ✅ |
| | Bimanual IK (indep/relative/mirror) | ✅ |
| | Bimanual Jacobian (indep/relative) | ✅ |
| **Modeling** | URDF parsing | ✅ |
| | MJCF parsing | ✅ |
| | Robot model abstraction | ✅ |
| | Bimanual robot model | ✅ |
| **Transform** | SE(3) operations | ✅ |
| | SO(3) operations | ✅ |
| | Rotation conversions | ✅ |
| | Quaternion operations | ✅ |
| **Analysis** | Workspace analysis | 🟡 |
| | Singularity analysis | 🟡 |
| **Planning** | Joint space trajectory | 🟡 |
| | Cartesian space trajectory | 🟡 |
| | Velocity profiles | 🟡 |
| **WDF** | SDF (Signed Distance Field) | ✅ |
| | RDF (Relative Distance Field) | ✅ |
| | Visualization | ✅ |
| **Config** | YAML configuration | ✅ |
| | Config schemas | ✅ |
| **Bridge** | MuJoCo simulation bridge | ✅ |
| | Real robot bridge (partial) | 🟡 |
| **Dynamics** | Inverse dynamics | ⚪ |
| | Forward dynamics | ⚪ |
| | Mass matrix computation | ⚪ |
| **Control** | PID controller | ⚪ |
| | MPC controller | ⚪ |
| | Impedance controller | ⚪ |
| **Collision** | Mesh-based collision detection | ⚪ |
| | Distance computation | ⚪ |
| **Path Planning** | RRT/RRT* algorithms | ⚪ |
| | PRM algorithms | ⚪ |
| | Optimization-based planning | ⚪ |

### Supported Robot Formats
- ✅ **URDF** (Unified Robot Description Format)
- ✅ **MJCF** (MuJoCo XML) - *Subset implementation for serial chains*

### Backend Support
- ✅ **NumPy** - CPU-optimized, 50-100x faster than pure Python
- ✅ **PyTorch** - GPU acceleration for batch operations

---

## 🚀 Performance Benchmarks

**Test Platform**: Intel i7-10700K, NVIDIA RTX 3080, 6-DOF Manipulator

### Single Configuration

| Operation | Pure Python | NumPy | Speedup |
|-----------|-------------|-------|---------|
| Forward Kinematics | 2.5 ms | **0.05 ms** | **50x** |
| Inverse Kinematics | 450 ms | **5.6 ms** | **80x** |
| Jacobian (Analytic) | 3.2 ms | **0.03 ms** | **107x** |
| Jacobian (Numeric) | 18 ms | **0.35 ms** | **51x** |

### Batch Processing (1000 configs)

| Operation | NumPy (CPU) | PyTorch (GPU) | Speedup |
|-----------|-------------|---------------|---------|
| Forward Kinematics | 45 ms | **3.2 ms** | **14x** |
| Jacobian (Analytic) | 28 ms | **2.1 ms** | **13x** |

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Synria-Robotics/RoboCore.git
cd RoboCore

# Install (development mode)
pip install -e .
pip install -r requirements.txt
```

---

## 🎯 Quick Start

### Basic Example

```python
from robocore.modeling import RobotModel

# Load robot (auto-detects URDF/MJCF)
robot = RobotModel("path/to/robot.urdf")

# Display model info
robot.summary(show_chain=True)
robot.print_tree()

# Forward Kinematics
q = [0.0] * robot.num_dof
pose = robot.fk(q, return_end=True)

# Inverse Kinematics
result = robot.ik(pose, q_initial=q, method='pinv')
print(f"Solution: {result['q']}, Success: {result['success']}")

# Jacobian
J = robot.jacobian(q, method='analytic')  # Shape: (6, dof)
```

### Batch Processing (GPU)

```python
import torch
import robocore as rc

# Generate random configurations
q_batch = robot.random_q_batch(batch_size=1000)

# Set global backend for GPU
rc.set_backend('torch', device='cuda')

# Batch FK on GPU
poses = robot.fk(
    torch.tensor(q_batch), 
    device='cuda',
    return_end=True
)
```

---

## 📚 Examples

```bash
# Robot model loading and validation
python examples/modeling/demo_robot_model.py --validate --show-tree

# Forward/Inverse kinematics
python examples/kinematics/demo_fk.py
python examples/kinematics/demo_ik.py

# Jacobian computation
python examples/kinematics/demo_jacobian.py

# Workspace analysis
python examples/analysis/demo_workspace.py --samples 10000

# Performance benchmark
python examples/kinematics/benchmark.py
```


## 🏗️ Project Structure

```
RoboCore/
├── robocore/
│   ├── modeling/          # Robot model abstraction & parsers
│   ├── kinematics/        # FK/IK/Jacobian solvers
│   ├── transform/         # SE(3)/SO(3) operations
│   ├── planning/          # Motion planning (WIP)
│   ├── analysis/          # Workspace/singularity analysis
│   ├── configs/           # Configuration management
│   └── utils/             # Backend abstraction, utilities
├── examples/              # Demo scripts
├── test/                  # Unit & integration tests
└── docs/                  # Documentation
```

---

## 📄 License

**GPL-3.0 License**  
Copyright © 2025 **Synria Robotics Co., Ltd.**

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

See the [LICENSE](LICENSE) file for the full license text.

---

## 📧 Contact

- **Website**: [synriarobotics.ai](https://synriarobotics.ai)
- **Email**: support@synriarobotics.ai

---

## 📖 Citation

```bibtex
@software{robocore2025,
  title = {RoboCore: High-Performance Robotics Kinematics Library},
  author = {Synria Robotics Team},
  year = {2025},
  publisher = {Synria Robotics Co., Ltd.},
  url = {https://github.com/Synria-Robotics/RoboCore},
  version = {1.0.0}
}
```