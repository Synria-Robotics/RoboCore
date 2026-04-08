#!/usr/bin/env python3
"""Demo 2: Cooperative Dual-Arm Control with Relative Constraint.

1. Drag red/blue spheres to set grasp configuration (relative pose)
2. Drag green sphere to move both arms cooperatively maintaining grasp

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from synriard import get_model_path


def main():
    """Demo 2: Cooperative dual-arm control with relative constraint."""
    
    # Model path (Bessica is a dual-arm robot)
    # mjcf_path = get_model_path("Alicia_D", version="v5_6", variant="bimanual_interactive", model_format="mjcf")
    mjcf_path = get_model_path("Bessica_D", version="v1_1", variant="skeleton_interactive", model_format="mjcf")
    # mjcf_path = get_model_path("Bessica_D", version="v1_1", variant="covered_interactive", model_format="mjcf")
    
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"

    # left_end = "tool0_site_l"
    # right_end = "tool0_site_r"
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in cooperative mode
    controller.run(mode='relative')


if __name__ == '__main__':
    main()
