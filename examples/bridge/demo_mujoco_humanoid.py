#!/usr/bin/env python3
"""MuJoCo Humanoid Interactive Demo

Demonstrates interactive IK control for humanoid robot with draggable thumb and toe targets.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from robocore.bridge.sim.mujoco.interactive_ik import InteractiveIK
from openrd import get_model_path


def main(args):
    # Model path (G1 is a humanoid robot)
    mjcf_path = get_model_path(
        args.model, 
        version=args.version, 
        variant=args.variant, 
        model_format="mjcf"
    )

    # Map target names to end-effector links
    # The new interface auto-discovers targets from MJCF based on naming convention:
    # - ik_target_left_thumb -> left_hand_thumb_2_link
    # - ik_target_right_thumb -> right_hand_thumb_2_link
    # - ik_target_left_toe -> left_ankle_roll_link
    # - ik_target_right_toe -> right_ankle_roll_link
    target_end_links = {
        "left_thumb": "left_hand_thumb_2_link",
        "right_thumb": "right_hand_thumb_2_link",
        "left_toe": "left_ankle_roll_link",
        "right_toe": "right_ankle_roll_link",
    }

    # Create interactive IK controller
    controller = InteractiveIK(mjcf_path, target_end_links=target_end_links)

    # Run interactive visualization
    controller.run(mode=args.mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Humanoid IK Control")
    parser.add_argument('--model', type=str, default="unitree_g1", 
                        help='Model: unitree_g1 (humanoid robot)')
    parser.add_argument('--version', type=str, default=None,
                        help='Version (optional for most robots)')
    parser.add_argument('--variant', type=str, default="g1_body29_hand14_interactive",
                        help='Variant: g1_body29_hand14_interactive (with draggable targets)')
    parser.add_argument('--mode', type=str, default='independent',
                        choices=['independent', 'relative', 'mirror'],
                        help='Mode: independent (independent mode), relative (relative mode), mirror (mirror mode)')
    args = parser.parse_args()
    main(args)
