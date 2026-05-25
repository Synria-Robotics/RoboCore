# Dynamics Examples

Demos and validation scripts for `robocore.dynamics`: inverse/forward dynamics, mass matrix, gravity, nonlinear effects, and optional Pinocchio comparison.

## Running

From the project root (or with `PYTHONPATH` including the repo):

```bash
# Activate your environment (e.g. conda activate synria)
conda activate synria

# Inverse dynamics: tau = RNEA(q, v, a)
python examples/dynamics/01a_demo_id.py

# With custom model and joint angles
python examples/dynamics/01a_demo_id.py --model-path /path/to/robot.urdf --joint-angles 0 0 0 0 0 0

# Forward dynamics: solve M(q) ddq = tau - nle(q, v)
python examples/dynamics/02a_demo_fd.py

# Mass matrix M(q) and M^{-1}(q)
python examples/dynamics/03a_demo_mass_matrix.py

# Gravity g(q), nonlinear effects nle(q,v), Coriolis C(q,v)
python examples/dynamics/04a_demo_gravity_nle.py

# Benchmark: ID, FD, mass_matrix, gravity, nle timing
python examples/dynamics/05_benchmark.py --runs 1000

# Self-consistency validation (no Pinocchio): tau = M*ddq + nle, ID–FD roundtrip
python examples/dynamics/06_validation.py --seed 42
```

**Recommended:** Run `06_validation.py` first to verify self-consistency (τ = M·ddq + nle and ID–FD roundtrip).

## Optional: Pinocchio comparison

**Unified chain (base_link, end_link):** RoboCore defines the chain by `base_link` and `end_link` (e.g. arm only → 6 DOF). Pinocchio loads the full URDF (e.g. 8 DOF with gripper). The `*_pk.py` scripts use Pinocchio’s `buildReducedModel` to lock joints outside the RC chain, so both sides use the **same segment and same DOF** for comparison.

If Pinocchio is installed (e.g. in the `synria` conda environment), the `*_pk.py` scripts compare RoboCore with Pinocchio on the same URDF and configuration:

```bash
python examples/dynamics/01c_demo_id_pk.py
python examples/dynamics/02c_demo_fd_pk.py
python examples/dynamics/03c_demo_mass_matrix_pk.py
```

Use `--no-pinocchio` to skip the comparison and only run RoboCore.

## Debugging

- **Validation first:** Run `06_validation.py` to ensure τ = M·ddq + nle and the ID–FD roundtrip hold. If these fail, fix the core implementation before comparing with Pinocchio.
- **RC vs Pin mismatch:** Check that the URDF is identical (same joint order, units, inertia). Compare q/v/tau and RC vs Pin outputs in the demo scripts; use `import pdb; pdb.set_trace()` or IDE breakpoints after `inverse_dynamics` / `rnea` to inspect intermediate quantities.
- **Joint mapping:** Chain DOF and order must match. The `*_pk.py` scripts map RoboCore chain joints to Pinocchio via joint names; if your model has different names or a different tree, the mapping may need adjustment.
- **Inertia:** Dynamics expect link inertials from the URDF. Links without `<inertial>` get zero mass/inertia; for full dynamics use a URDF with proper `<inertial>` tags.
