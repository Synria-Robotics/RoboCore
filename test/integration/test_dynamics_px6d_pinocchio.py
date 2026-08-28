"""PX6D follower dynamics validation against Pinocchio.

Run in the ``arx`` conda env::

    conda activate arx
    pip install -e . pybind11 pytest
    conda install -y --override-channels -c conda-forge pinocchio
    ./scripts/test_dynamics_px6d.sh -q

Notes:
- Do **not** ``pip install pinocchio`` (PyPI package is unrelated to robotics).
- If pytest crashes loading ROS ``launch_testing``, use
  ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` (see ``scripts/test_dynamics_px6d.sh``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robocore.benchmark.dynamics_pinocchio import validate_dynamics_against_pinocchio
from robocore.modeling import RobotModel


pytest.importorskip("pinocchio")

PX6D_URDF = Path(
    "/home/ubuntu/Alicia/Synria-Robot-Descriptions/synriard/urdf/Alicia_M_v1_2/"
    "Alicia_M_v1_2_follower_ARX_PX6D.urdf"
)


@pytest.fixture(scope="session")
def px6d_urdf_path() -> str:
    if not PX6D_URDF.is_file():
        pytest.skip(f"PX6D URDF not found: {PX6D_URDF}")
    return str(PX6D_URDF)


def test_px6d_dynamics_matches_pinocchio(px6d_urdf_path: str):
    model = RobotModel(px6d_urdf_path, base_link="base_link", end_link="tool0")
    result = validate_dynamics_against_pinocchio(model, px6d_urdf_path, trials=2, seed=1)
    assert result["max"]["inverse_dynamics"] < 1e-6
    assert result["max"]["mass_matrix"] < 1e-6
    assert result["max"]["gravity"] < 1e-6
    assert result["max"]["nonlinear_effects"] < 1e-6
    assert result["max"]["forward_dynamics"] < 1e-6
