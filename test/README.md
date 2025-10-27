# RoboCore Tests

This directory contains automated tests for the RoboCore library.

## Structure

- `unit/` - Unit tests for individual components (FK, IK, Jacobian)
- `integration/` - Integration tests for end-to-end workflows
- `benchmarks/` - Performance benchmarks (in examples/)

## Running Tests

### Run all tests
```bash
pytest test/
```

### Run specific test categories
```bash
# Unit tests only
pytest test/unit/

# Integration tests only
pytest test/integration/

# Specific test file
pytest test/unit/test_jacobian.py

# Specific test class/method
pytest test/unit/test_jacobian.py::TestJacobianConsistency::test_numpy_vs_torch_single
```

### Run with verbose output
```bash
pytest test/ -v
```

### Run with coverage
```bash
pytest test/ --cov=robocore --cov-report=html
```

## Test Categories

### Unit Tests (`test/unit/`)

**`test_fk.py`** - Forward Kinematics
- Backend consistency (NumPy vs PyTorch)
- Batch mode correctness
- Edge cases (zero configuration)

**`test_jacobian.py`** - Jacobian Computation
- Backend consistency
- Batch vs single mode
- Numeric vs analytic methods

**`test_ik.py`** - Inverse Kinematics
- IK method comparison (DLS, pseudoinverse, transpose)
- FK-IK-FK closure property validation
- Convergence tracking
- Joint limit enforcement
- Edge cases (infeasible poses, zero configuration)

**`test_transform.py`** - SO(3)/SE(3) Transforms
- Rotation matrix conversions (quaternion, axis-angle, Euler)
- SE(3) homogeneous transform operations
- Batch mode consistency
- Backend parity (NumPy vs PyTorch)

**`test_robot_model.py`** - Robot Model
- URDF/MJCF parsing
- DOF computation
- Joint chain construction
- Model property validation

**`test_backend.py`** - Backend Management
- Backend switching (NumPy ↔ PyTorch)
- Singleton pattern enforcement
- Context isolation
- Array type creation consistency

**`test_config.py`** - Configuration System
- YAML loading and parsing
- Schema validation
- Nested key access
- Error handling (invalid paths, malformed YAML)

### Integration Tests (`test/integration/`)

**`test_ik_accuracy.py`** - IK End-to-End Accuracy
- Success rate validation (NumPy vs PyTorch batch)
- Multi-sample statistical tests
- Ensures batch IK achieves ≥70% success rate
- Backend parity within 5% success rate difference

**`test_workspace.py`** - Workspace Analysis
- Reachability computation
- Workspace volume estimation
- Point reachability queries
- Boundary detection
- Cross-section analysis

**`test_singularity.py`** - Singularity Analysis
- Manipulability measure computation
- Jacobian condition number
- Singular configuration detection
- Singularity type classification
- Null space analysis

## Continuous Integration

These tests are designed to be run in CI pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest test/ --cov=robocore
```

## Requirements

- pytest >= 7.0
- pytest-cov (optional, for coverage)
- All RoboCore dependencies (PyTorch, NumPy, etc.)

## Adding New Tests

1. Choose appropriate directory (`unit/` vs `integration/`)
2. Create `test_*.py` file
3. Use pytest fixtures for setup (see existing tests)
4. Follow naming convention: `test_<feature>.py`, `class Test<Feature>`, `def test_<case>`
5. Add assertions with clear failure messages

## Known Issues

- Tests require URDF file at `robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf`
- GPU tests require CUDA-capable device (automatically skip if unavailable)
