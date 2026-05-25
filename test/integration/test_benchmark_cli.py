"""Benchmark CLI smoke tests."""

from __future__ import annotations

import json

from robocore.benchmark.__main__ import main


def test_kinematics_benchmark_cli_writes_json_and_markdown(tmp_path, alicia_urdf_path):
    code = main([
        "kinematics",
        "--urdf", alicia_urdf_path,
        "--backends", "cpp,numpy",
        "--samples", "3",
        "--ik-samples", "3",
        "--repeats", "1",
        "--inner-single", "1",
        "--inner-batch", "1",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    payload = json.loads((tmp_path / "kinematics.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["samples"] == 3
    assert payload["metadata"]["ik_samples"] == 3
    assert payload["results"]
    assert (tmp_path / "kinematics.md").is_file()


def test_dynamics_benchmark_cli_writes_json_and_markdown(tmp_path, alicia_urdf_path):
    code = main([
        "dynamics",
        "--urdf", alicia_urdf_path,
        "--backends", "cpp,numpy",
        "--samples", "3",
        "--repeats", "1",
        "--inner-single", "1",
        "--inner-batch", "1",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    payload = json.loads((tmp_path / "dynamics.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["samples"] == 3
    assert payload["results"]
    assert (tmp_path / "dynamics.md").is_file()
