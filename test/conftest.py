"""Shared pytest fixtures for robot description assets."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_backend_between_tests():
    """Keep global backend state from leaking across tests."""
    from robocore.utils.backend import get_backend, set_backend

    if get_backend() != "numpy":
        set_backend("numpy")
    yield
    if get_backend() != "numpy":
        set_backend("numpy")


def _description_or_skip(uri: str) -> str:
    try:
        from robocore.configs import resolve_description_path
        return resolve_description_path(uri)
    except Exception as exc:
        pytest.skip(f"Robot description unavailable for {uri}: {exc}")


@pytest.fixture(scope="session")
def alicia_urdf_path() -> str:
    return _description_or_skip("synriard://Alicia_D/v5_6/gripper_100mm/urdf")


@pytest.fixture(scope="session")
def bessica_urdf_path() -> str:
    return _description_or_skip("synriard://Bessica_D/v1_1/covered/urdf")


@pytest.fixture(scope="session")
def bessica_mjcf_path() -> str:
    return _description_or_skip("synriard://Bessica_D/v1_1/covered/mjcf")
