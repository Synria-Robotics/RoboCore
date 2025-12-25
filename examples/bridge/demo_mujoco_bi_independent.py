#!/usr/bin/env python3
"""Demo 1: Independent Dual-Arm Control.

Drag red and blue spheres independently to control left and right arms.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from synriard import get_model_path


def main():
    """Demo 1: Independent dual-arm IK control."""
    
    # Model path (Bessica is a dual-arm robot)
    mjcf_path = get_model_path("Bessica_D", version="v1_1", variant="skeleton_interactive", model_format="mjcf")
    # mjcf_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in independent mode
    controller.run(mode='independent')


if __name__ == '__main__':
    main()
