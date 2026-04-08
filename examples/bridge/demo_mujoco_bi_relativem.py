#!/usr/bin/env python3
"""Demo 2: Cooperative Dual-Arm Control with Relative Constraint.

1. Drag red/blue spheres to set grasp configuration (relative pose)
2. Drag green sphere to move both arms cooperatively maintaining grasp

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from synriard import get_model_path

INITIAL_JOINT_STATE = {
    "joint1_l": 1.57,
    "joint2_l": 0.785,
    "joint3_l": -1.57,
    "joint4_l": 0,#-1.18,
    "joint5_l": 0.0,
    "joint6_l": 0.0,
    "joint7_l": 0.0,
    "joint8_l": 0.0,
    "joint1_r": -1.57,#1.21,
    "joint2_r": 0.785,
    "joint3_r": -1.57,
    "joint4_r": 0.0,
    "joint5_r": 0.0,
    "joint6_r": 0.0,
    "joint7_r": 0.0,
    "joint8_r": 0.0,
}


def main():
    """Demo 2: Cooperative dual-arm control with relative constraint."""
    
    # Model path (Bessica is a dual-arm robot)
    mjcf_path = get_model_path("Alicia_M", version="v1_1", variant="bimanual_interactive", model_format="mjcf")
    # mjcf_path = get_model_path("Bessica_D", version="v1_1", variant="covered_interactive", model_format="mjcf")
    
    
    # End-effector links
    # left_end = "left_arm_link7"
    # right_end = "right_arm_link7"

    left_end = "tool0_site_l"
    right_end = "tool0_site_r"
    # Create interactive IK controller
    controller = InteractiveDualArmIK(
        mjcf_path,
        left_end,
        right_end,
        initial_joint_state=INITIAL_JOINT_STATE,
    )
    
    # Run in cooperative mode
    controller.run(mode='relative')


if __name__ == '__main__':
    main()
