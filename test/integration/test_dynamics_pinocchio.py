"""Optional Pinocchio dynamics regression tests."""

from __future__ import annotations

import pytest

from robocore.benchmark.dynamics_pinocchio import validate_dynamics_against_pinocchio
from robocore.modeling import RobotModel


pytest.importorskip("pinocchio")


def test_dynamics_matches_pinocchio_smoke(alicia_urdf_path):
    model = RobotModel(alicia_urdf_path, base_link="base_link", end_link="tool0")
    result = validate_dynamics_against_pinocchio(model, alicia_urdf_path, trials=2, seed=0)
    assert result["max"]["inverse_dynamics"] < 1e-6
    assert result["max"]["mass_matrix"] < 1e-6
    assert result["max"]["gravity"] < 1e-6
    assert result["max"]["nonlinear_effects"] < 1e-6
    assert result["max"]["forward_dynamics"] < 1e-6
