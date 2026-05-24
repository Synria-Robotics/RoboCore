"""Singularity analysis demo.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from robocore.modeling.robot_model import RobotModel
from robocore.analysis.singularity_analyzer import SingularityAnalyzer
from robocore.configs import resolve_description_path
from robocore.utils.beauty_logger import beauty_print


def main():
    urdf = resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf")
    model = RobotModel(str(urdf), end_link="tool0")

    analyzer = SingularityAnalyzer(model)

    beauty_print("Singularity Analysis for " + model.name, type="module")

    # Test specific configurations
    configs = {
        "Zero pose": [0.0] * model.num_dof,
        "Extended": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
        "Folded": [0.0, 1.57, 0.0, 0.0, 0.0, 0.0],
    }

    for name, q in configs.items():
        beauty_print(f"\nConfiguration: {name}", type="module")
        beauty_print(f"q = {[round(x, 3) for x in q]}")
        try:
            result = analyzer.analyze_configuration(q)
            beauty_print(
                f"Manipulability: {result['manipulability']:.6f}  "
                f"Condition: {result['condition_number']:.2f}  "
                f"σ_min: {result['min_singular_value']:.6f}  "
                f"Singular: {result['is_singular']}",
                type="success" if not result["is_singular"] else "warning",
            )
            beauty_print(f"Singular values: {[f'{s:.4f}' for s in result['singular_values']]}")
        except Exception as e:
            beauty_print(f"Error: {e}", type="error")

    # Workspace sampling
    beauty_print("\nWorkspace Sampling Analysis", type="module")
    stats = analyzer.sample_workspace(n_samples=1000)
    if "error" not in stats:
        beauty_print(f"Samples: {stats['n_samples']}")
        beauty_print(f"Singular configs: {stats['singular_configs']} ({stats['singular_ratio']*100:.2f}%)")
        beauty_print(
            f"Manipulability: mean={stats['manipulability_mean']:.6f} "
            f"min={stats['manipulability_min']:.6f} max={stats['manipulability_max']:.6f}"
        )
        beauty_print(
            f"Condition number: mean={stats['condition_number_mean']:.2f} max={stats['condition_number_max']:.2f}"
        )
    else:
        beauty_print(stats["error"], type="error")


if __name__ == "__main__":
    main()
