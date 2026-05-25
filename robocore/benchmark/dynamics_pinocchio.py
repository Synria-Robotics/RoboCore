"""Optional dynamics validation against Pinocchio."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robocore import dynamics


@dataclass
class DynamicsPinocchioTrial:
    trial: int
    inverse_dynamics_max_abs: float
    mass_matrix_max_abs: float
    gravity_max_abs: float
    nonlinear_effects_max_abs: float
    forward_dynamics_max_abs: float


def _pin_context(model: Any, urdf_path: str | Path):
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("Pinocchio is required for dynamics validation. Install with the benchmark extra.") from exc

    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()
    chain_indices = model._get_joint_indices(model.base_link, model.end_link)
    joint_names = [model.joint_list[idx].name for idx in chain_indices]
    q_indices: list[int] = []
    v_indices: list[int] = []
    for name in joint_names:
        if not pin_model.existJointName(name):
            raise RuntimeError(f"Pinocchio model is missing joint {name!r}")
        jid = pin_model.getJointId(name)
        q_indices.append(pin_model.idx_qs[jid])
        v_indices.append(pin_model.idx_vs[jid])
    return pin, pin_model, pin_data, q_indices, v_indices


def _pin_qva(pin, pin_model, q_indices: list[int], v_indices: list[int], q, v, a):
    q_full = pin.neutral(pin_model).copy()
    v_full = np.zeros(pin_model.nv, dtype=np.float64)
    a_full = np.zeros(pin_model.nv, dtype=np.float64)
    for i, qidx in enumerate(q_indices):
        q_full[qidx] = float(q[i])
    for i, vidx in enumerate(v_indices):
        v_full[vidx] = float(v[i])
        a_full[vidx] = float(a[i])
    return q_full, v_full, a_full


def validate_dynamics_against_pinocchio(
    rc_model: Any,
    urdf_path: str | Path,
    *,
    trials: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare RoboCore rigid-body dynamics APIs with Pinocchio on one model."""
    pin, pin_model, pin_data, q_indices, v_indices = _pin_context(rc_model, urdf_path)
    rng = np.random.default_rng(seed)
    n = rc_model.num_chain_dof
    rows: list[DynamicsPinocchioTrial] = []

    for trial in range(trials):
        q = rng.uniform(-0.4, 0.4, n)
        v = rng.uniform(-0.1, 0.1, n)
        a = rng.uniform(-0.1, 0.1, n)
        tau = rng.uniform(-0.5, 0.5, n)
        q_full, v_full, a_full = _pin_qva(pin, pin_model, q_indices, v_indices, q, v, a)

        rc_id = dynamics.inverse_dynamics(rc_model, q, v, a)
        pin_id = np.asarray(pin.rnea(pin_model, pin_data, q_full, v_full, a_full), dtype=np.float64)[v_indices]

        rc_M = dynamics.mass_matrix(rc_model, q)
        pin_M_full = np.asarray(pin.crba(pin_model, pin_data, q_full), dtype=np.float64)
        pin_M = pin_M_full[np.ix_(v_indices, v_indices)]
        pin_M = 0.5 * (pin_M + pin_M.T)

        rc_g = dynamics.gravity(rc_model, q)
        pin_g = np.asarray(pin.computeGeneralizedGravity(pin_model, pin_data, q_full), dtype=np.float64)[v_indices]

        rc_nle = dynamics.nonlinear_effects(rc_model, q, v)
        pin_nle = np.asarray(pin.nonLinearEffects(pin_model, pin_data, q_full, v_full), dtype=np.float64)[v_indices]

        rc_fd = dynamics.forward_dynamics(rc_model, q, v, tau)
        # RoboCore dynamics is built on the selected serial chain, so compare
        # against the same Pinocchio chain projection: ddq = M^-1(tau - nle).
        # Calling full-model ABA can include passive/off-chain joints from the URDF.
        pin_fd = np.linalg.solve(pin_M, tau - pin_nle)

        rows.append(DynamicsPinocchioTrial(
            trial=trial,
            inverse_dynamics_max_abs=float(np.max(np.abs(rc_id - pin_id))),
            mass_matrix_max_abs=float(np.max(np.abs(rc_M - pin_M))),
            gravity_max_abs=float(np.max(np.abs(rc_g - pin_g))),
            nonlinear_effects_max_abs=float(np.max(np.abs(rc_nle - pin_nle))),
            forward_dynamics_max_abs=float(np.max(np.abs(rc_fd - pin_fd))),
        ))

    summary = {
        "trials": trials,
        "max": {
            "inverse_dynamics": max(row.inverse_dynamics_max_abs for row in rows),
            "mass_matrix": max(row.mass_matrix_max_abs for row in rows),
            "gravity": max(row.gravity_max_abs for row in rows),
            "nonlinear_effects": max(row.nonlinear_effects_max_abs for row in rows),
            "forward_dynamics": max(row.forward_dynamics_max_abs for row in rows),
        },
        "rows": [asdict(row) for row in rows],
    }
    return summary
