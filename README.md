![](./imgs/logo.jpeg)

# RoboCore: Unified High-Throughput Robotics Library

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Developed by [Synria Robotics Co., Ltd.](https://synriarobotics.ai)** 🤖

---

## ✨ Features

| Module | Functionality | Status |
|--------|---------------|--------|
| **Modeling** | URDF/MJCF parsing, Robot model abstraction | ✅ Stable |
| **Forward Kinematics** | NumPy/PyTorch backends, Batch processing | ✅ Stable |
| **Inverse Kinematics** | DLS/Pinv/Transpose methods, Multi-start | ✅ Stable |
| **Jacobian** | Analytic/Numeric/Autograd methods | ✅ Stable |
| **Transform** | SE(3)/SO(3) operations, Conversions | ✅ Stable |
| **Analysis** | Workspace/Singularity analysis | ✅ Beta |
| **Planning** | Trajectory generation | 🚧 Alpha |
| **Visualization** | Kinematic tree display | ✅ Stable |
| **Configuration** | YAML-based config management | ✅ Stable |

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

# Optional: Install with PyTorch for GPU support
pip install torch torchvision
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
q = [0.0] * robot.num_dof()
pose = robot.fk(q, backend='numpy', return_end=True)

# Inverse Kinematics
result = robot.ik(pose, q_initial=q, method='pinv')
print(f"Solution: {result['q']}, Success: {result['success']}")

# Jacobian
J = robot.jacobian(q, method='analytic')  # Shape: (6, dof)
```

### Batch Processing (GPU)

```python
import torch

# Generate random configurations
q_batch = robot.random_q_batch(batch_size=1000)

# Batch FK on GPU
poses = robot.fk(
    torch.tensor(q_batch), 
    backend='torch', 
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

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Specific test suites
pytest test/unit/          # Unit tests
pytest test/integration/   # Integration tests

# With coverage report
pytest --cov=robocore --cov-report=html
```

---

## 📋 TODO & Roadmap

### High Priority 🔴
- [ ] **Collision Detection** - Mesh-based collision checking
- [ ] **Path Planning** - RRT/RRT*/PRM algorithms
- [ ] **Dynamics** - Inverse/Forward dynamics computation
- [ ] **Control** - PID, MPC, impedance controllers

### Medium Priority 🟡
- [ ] **Optimization** - Further SIMD/vectorization improvements
- [ ] **Mobile Manipulators** - Support for mobile bases
- [ ] **Multi-Arm Systems** - Coordinated multi-robot control

### Low Priority 🟢
- [ ] **Visualization** - 3D interactive visualization (PyBullet/MuJoCo)
- [ ] **Documentation** - API docs, tutorials, best practices


---

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