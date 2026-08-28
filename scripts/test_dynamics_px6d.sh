#!/usr/bin/env bash
# Run PX6D dynamics validation in the ``arx`` conda env.
#
# Prerequisites (once):
#   conda activate arx
#   pip install -e /home/ubuntu/Synria/RoboCore pybind11 pytest
#   conda install -y --override-channels -c conda-forge pinocchio
#
# Do NOT ``pip install pinocchio`` — PyPI pinocchio is unrelated to robotics.
#
# ROS Humble pytest plugins conflict with pytest 9; disable autoload below.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cd "$ROOT"
exec "$PY" -m pytest -p pytest test/integration/test_dynamics_px6d_pinocchio.py "$@"
