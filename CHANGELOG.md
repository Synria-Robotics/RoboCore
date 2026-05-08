# Changelog

All notable changes to RoboCore will be documented in this file.

## [2.5.0] - 2026-05-08

### 🚀 New Features
- **C++ kinematics backend**: pybind11 + Eigen3 accelerated FK, IK, and Jacobian solvers (`_fk_chain_core`, `_ik_chain_core`, `_multichain_ik_core`, `_jacobian_chain_core`)
- **Multi-chain IK**: unified solver for bimanual and humanoid configurations via `MultiChainSolver`
- **Dynamics module**: inverse dynamics (ID), forward dynamics (FD), mass matrix, gravity/NLE via Pinocchio integration
- **Motion planning**: joint-space and Cartesian-space planners with trapezoidal and S-curve velocity profiles, spline and polynomial trajectories
- **MuJoCo bridge**: physics simulation, trajectory execution/evaluation/visualization, interactive IK, comparison analyzer
- **Config system**: YAML-based robot configuration manager with schema validation
- **WDF module**: workspace/reachability/signed-distance field analysis with BBO-based planning
- **Analysis module**: workspace analyzer and singularity analyzer

### 📦 Packaging
- Migrated to light-core + optional extras model:
  - `pip install synria-robocore` — core only (numpy, scipy, pyyaml)
  - `pip install synria-robocore[torch]` — PyTorch backend
  - `pip install synria-robocore[mujoco]` — MuJoCo bridge
  - `pip install synria-robocore[sim]` — simulation utilities
  - `pip install synria-robocore[descriptions]` — robot URDF packages
  - `pip install synria-robocore[all]` — everything
- Lazy imports for optional dependencies (torch, mujoco) — no ImportError on core install
- PEP 639 compliant: `license = "MIT"` in `pyproject.toml`
- Native extension build via `setup.py` with pybind11 + Eigen3

### ⚖️ License
- **Migrated from GPL-3.0 to MIT License**
- All project-owned source files re-headered to MIT
- Apache-2.0 headers preserved in `robocore/modeling/parser/mjcf_parser/` (dm_control-derived)

### Changed
- `robocore/__init__.py`: lazy submodule loading via `__getattr__`, `__license__ = "MIT"`
- Removed direct URL dependencies for internal robot packages from core deps

## [1.0.0] - 2024-10-04

### 🎉 Major Release - Production Ready

### Fixed
- **[CRITICAL]** Batch IK accuracy issue: Fixed FK chain traversal to include fixed joints
  - Success rate improved from 41% to 78% (now matches NumPy baseline)
  - Jacobian matrix accuracy improved to perfect match (max_diff: 0.126 → 0.0)
  - File: `robocore/kinematics/ik_utils/ik_solver_torch.py`

- **Backend contamination**: NumPy solvers now properly isolate backend state
  - Files: `robocore/kinematics/fk_utils/fk_solver_numpy.py`, `robocore/kinematics/jacobian_utils/jacobian_solver_numpy.py`

- **Type annotations**: Fixed Tensor type imports for proper IDE support
  - File: `robocore/kinematics/ik_utils/ik_solver_torch.py`

### Changed
- Cleaned debug code from production solvers
  - Removed `debug` parameter from `_solve_batch()`
  - Removed debug print statements

### Added
- **Test infrastructure**: Comprehensive pytest-based test suite
  - Unit tests: FK, Jacobian validation
  - Integration tests: End-to-end IK accuracy
  - Configuration: `pytest.ini` with markers
  - Documentation: `test/README.md`

- **Documentation**:
  - `docs/CODE_QUALITY_REPORT.md`: Detailed bug analysis and fixes
  - `docs/VALIDATION_SUMMARY.md`: Final validation results
  - Updated `examples/README.md` with test references

### Removed
- Moved debug scripts from `examples/` to `test/integration/`
- Cleaned obsolete test files

### Performance
- **Batch IK** (64 samples, CUDA):
  - Speed: 7.7 ms/sample (2.1x faster than NumPy)
  - Success rate: 75.0% (vs NumPy 76.6%)
  
- **Batch IK** (128 samples, CUDA):
  - Speed: 3.8 ms/sample (3.9x faster than NumPy)
  - Success rate: 78.1% (vs NumPy 80.5%)

- **Batch FK** (64 samples, CUDA):
  - Speed: 0.042 ms/sample (5.5x faster than NumPy)

### Testing
```bash
$ pytest test/ -v
==================================================
5 passed, 3 skipped (non-critical), 0 failed
==================================================
```

## [0.9.0] - 2024-10-03

### Added
- Initial batch IK implementation (with accuracy issues, fixed in 1.0.0)
- Adaptive damping and plateau detection
- Multi-backend support (NumPy, PyTorch)

---

## Version Guidelines

Format: `[Major.Minor.Patch]`
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

Categories:
- `Added`: New features
- `Changed`: Changes to existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Security fixes
- `Performance`: Performance improvements
