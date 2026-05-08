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
| | Multi-chain support | ✅ |
| | Bimanual robot model | ✅ |
| | Workspace analysis | ✅ |
| **Transform** | SE(3) operations | ✅ |
| | SO(3) operations | ✅ |
| | Rotation conversions | ✅ |
| | Quaternion operations | ✅ |
| **Planning** | Joint space (polynomial/spline/multi-segment) | ✅ |
| | Cartesian space (linear/circular/spline) | ✅ |
| | Orientation planning (SLERP) | ✅ |
| | Velocity profiles (trapezoidal/S-curve) | ✅ |
| **Control** | Joint position controller (PD/PID) | ✅ |
| | Joint velocity controller | ✅ |
| | Joint trajectory tracking controller | ✅ |
| | Cartesian position controller | ✅ |
| | Cartesian velocity controller | ✅ |
| | Cartesian trajectory tracking controller | ✅ |
| | Computed torque controller | ⚪ |
| | Impedance controller | ⚪ |
| | MPC controller | ⚪ |
| **Analysis** | Workspace analysis | ✅ |
| | Singularity analysis | 🟡 |
| **WDF** | SDF (Signed Distance Field) | ✅ |
| | RDF (Relative Distance Field) | ✅ |
| | Visualization | ✅ |
| **Config** | YAML configuration | ✅ |
| | Config schemas | ✅ |
| **Bridge** | MuJoCo simulation bridge | ✅ |
| | Physics simulation & evaluation | ✅ |
| | Real robot bridge (partial) | 🟡 |
| **Dynamics** | Inverse dynamics | ⚪ |
| | Forward dynamics | ⚪ |
| | Mass matrix computation | ⚪ |
| | Coriolis & gravity computation | ⚪ |
| **Collision** | Mesh-based collision detection | ⚪ |
| | Distance computation | ⚪ |
| **Path Planning** | RRT/RRT* algorithms | ⚪ |
| | PRM algorithms | ⚪ |
| | Optimization-based planning | ⚪ |

### Supported Robot Formats
- ✅ **URDF** - Unified Robot Description Format
- ✅ **MJCF** - MuJoCo XML

### Backend Support
- ✅ **NumPy** - CPU vectorized reference implementation; see **Performance benchmarks** below
- ✅ **PyTorch** - Same API with optional **CUDA** for batched FK / Jacobian / IK (`--device cuda` in the benchmark scripts)
- ✅ **C++ / Eigen** - Same FK, analytic Jacobian, and **DLS IK** API via pybind11 extensions (`cpp` / `IKSolverCpp`); see [Installation](#installation)

---

## 🚀 Performance Benchmarks

**Representative run** (Apple Silicon laptop, **CPU** PyTorch for RoboCore and PK; C++ extensions built).

**vs C++** columns: latency ratio `t_backend / t_RoboCore C++` for the same batch shape (**1×** = same speed as C++; **>1×** = that many times **slower** than C++).

### Forward kinematics (end-effector 4×4)

| Backend | batch=1 (ms/call) | batch=100 (ms/call) | vs C++ b=1 | vs C++ b=100 |
|---------|-------------------|---------------------|------------|--------------|
| pytorch_kinematics | 0.209 | 0.331 | ~209× | ~39× |
| pinocchio | 0.002 | 0.198 | ~2.0× | ~23× |
| RoboCore NumPy | 0.134 | 0.353 | ~134× | ~42× |
| RoboCore PyTorch (CPU) | 0.561 | 0.652 | ~561× | ~77× |
| **RoboCore C++ / Eigen** | **0.0010** | **0.0085** | **1×** | **1×** |


### Analytic Jacobian (6 × n)

| Backend | batch=1 (ms/call) | batch=100 (ms/call) | vs C++ b=1 | vs C++ b=100 |
|---------|-------------------|---------------------|------------|--------------|
| pytorch_kinematics | 0.652 | 0.900 | ~540× | ~43× |
| pinocchio | 0.0029 | 0.307 | ~2.4× | ~15× |
| RoboCore NumPy | 0.116 | 11.68 | ~97× | ~560× |
| RoboCore PyTorch (CPU) | 0.608 | 1.50 | ~510× | ~72× |
| **RoboCore C++ / Eigen** | **0.0012** | **0.0209** | **1×** | **1×** |


### Inverse kinematics (single chain)

| Backend | batch=1 (ms/call) | batch=100 (ms/call) | vs C++ b=1 | vs C++ b=100 |
|---------|-------------------|---------------------|------------|--------------|
| pytorch_kinematics (PseudoInverseIK) | 195 | 259 | ~24k× | ~340× |
| pinocchio (CLIK, Python) | 0.68 | 410 | ~85× | ~530× |
| RoboCore NumPy (DLS) | 2.195 | 62.03 | ~270× | ~81× |
| RoboCore PyTorch (CPU, DLS) | 3.49 | 57.17 | ~440× | ~74× |
| **RoboCore C++ / Eigen** (DLS) | **0.008** | **0.77** | **1×** | **1×** |


---

## 📦 Installation

RoboCore is packaged as a **light core** with optional extras.

- Core install: NumPy-based kinematics, transforms, planning, analysis, configs, URDF parsing, and native `cpp` backend support.
- `torch` extra: PyTorch-based solvers and GPU execution.
- `mujoco` extra: MJCF parsing and MuJoCo runtime support.
- `sim` extra: MuJoCo simulation bridge support, including `mujoco-py` compatibility.
- `descriptions` extra: Synria and Open robot-description packages (`synriard`, `openrd`) for examples and tests.

FK, Jacobian, and IK can run on **NumPy**, **PyTorch**, or **`cpp`** backends; only the **`cpp`** path uses native extensions, built with **pybind11** + **Eigen3** (headers only) + a **C++17** compiler.

- **pybind11** — installed automatically for the build via `pyproject.toml` (`[build-system] requires`) when you run `pip install -e .` (PEP 517 isolated environment).
- **Eigen** — *not* a pip package; install on the system (see below) or set **`EIGEN3_INCLUDE_DIR`** to the directory that contains the `Eigen/` folder (so `Eigen/Dense` exists).
- If Eigen or pybind11 is missing when the `.cpp` sources are present, the build **fails** with an error (extensions are required).

### C++ compiler

| Platform | How to install |
|----------|----------------|
| **macOS** | `xcode-select --install` (Command Line Tools, includes `clang++`) or full Xcode from the App Store. |
| **Debian / Ubuntu** | `sudo apt update && sudo apt install build-essential` |
| **Fedora** | `sudo dnf install gcc-c++ make` |
| **Windows** | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with “Desktop development with C++”, or use **WSL2** and follow the Linux steps above. |

### Eigen3

| Platform | Command / notes |
|----------|-----------------|
| **macOS (Homebrew)** | `brew install eigen` — headers are under `/opt/homebrew/include/eigen3` (Apple Silicon) or `/usr/local/include/eigen3` (Intel); `setup.py` checks these paths. |
| **Debian / Ubuntu** | `sudo apt install libeigen3-dev` — typically `/usr/include/eigen3`. |
| **Fedora** | `sudo dnf install eigen3-devel` |
| **Windows** | e.g. `vcpkg install eigen3`, or download Eigen and set `EIGEN3_INCLUDE_DIR` to the folder that **directly contains** the `Eigen` directory. |

Custom location:

```bash
export EIGEN3_INCLUDE_DIR="/path/to/include/eigen3"   # must contain Eigen/Dense
```

### Clone and install

```bash
# Clone repository
git clone https://github.com/Synria-Robotics/RoboCore.git
cd RoboCore

conda create -n synria python=3.10 -y
conda activate synria

# Install core package in development mode
pip install -e .

# Optional extras
pip install -e ".[torch]"
pip install -e ".[mujoco]"
pip install -e ".[sim]"
pip install -e ".[descriptions]"

# Everything
pip install -e ".[all]"
```

### PyPI installs

```bash
# Core package
pip install synria-robocore

# Typical feature sets
pip install "synria-robocore[torch]"
pip install "synria-robocore[mujoco]"
pip install "synria-robocore[sim]"
pip install "synria-robocore[descriptions]"

# Full environment
pip install "synria-robocore[all]"
```

`synriard` and `openrd` are only needed for bundled examples/tests that load published robot-description assets. They are not required for the RoboCore core library itself.

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

**MIT License**  
Copyright © 2025 **Synria Robotics Co., Ltd.**

RoboCore is released under the MIT License.

See [LICENSE](LICENSE) for the full text. Files under
`robocore/modeling/parser/mjcf_parser` include preserved upstream Apache-2.0
copyright and license notices where applicable.

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