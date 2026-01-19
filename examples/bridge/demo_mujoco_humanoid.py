#!/usr/bin/env python3
"""MuJoCo Humanoid Interactive Demo

Demonstrates interactive IK control for humanoid robot with draggable thumb and toe targets.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from robocore.bridge.sim.mujoco.interactive_humanoid import InteractiveHumanoidIK
from openrd import get_model_path


def main(args):
    # Model path (G1 is a humanoid robot)
    mjcf_path = get_model_path(
        args.model, 
        version=args.version, 
        variant=args.variant, 
        model_format="mjcf"
    )

    # End-effector links for thumbs and toes
    left_thumb_end = "left_hand_thumb_2_link"
    right_thumb_end = "right_hand_thumb_2_link"
    left_toe_end = "left_ankle_roll_link"
    right_toe_end = "right_ankle_roll_link"

    # Create interactive IK controller
    controller = InteractiveHumanoidIK(
        mjcf_path, 
        left_thumb_end, 
        right_thumb_end,
        left_toe_end,
        right_toe_end
    )

    # Run interactive visualization
    controller.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Humanoid IK Control")
    parser.add_argument('--model', type=str, default="unitree_g1", 
                        help='Model: unitree_g1 (humanoid robot)')
    parser.add_argument('--version', type=str, default=None,
                        help='Version (optional for most robots)')
    parser.add_argument('--variant', type=str, default="g1_body29_hand14_interactive",
                        help='Variant: g1_body29_hand14_interactive (with draggable targets)')
    args = parser.parse_args()
    main(args)
