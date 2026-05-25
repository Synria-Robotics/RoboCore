"""Validate RoboCore dynamics against Pinocchio.

Install optional dependencies first:

    pip install -e ".[benchmark,descriptions]"
"""

from __future__ import annotations

import argparse
import json

from robocore.benchmark.dynamics_pinocchio import validate_dynamics_against_pinocchio
from robocore.configs import resolve_description_path
from robocore.modeling import RobotModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="tool0")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    urdf = args.urdf or resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")
    model = RobotModel(urdf, base_link=args.base_link, end_link=args.end_link)
    result = validate_dynamics_against_pinocchio(model, urdf, trials=args.trials, seed=args.seed)
    print(json.dumps(result["max"], indent=2))
    return 0 if all(value < args.atol for value in result["max"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
