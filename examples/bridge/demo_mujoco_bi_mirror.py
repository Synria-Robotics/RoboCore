#!/usr/bin/env python3
"""Demo 3: Mirror Symmetric Dual-Arm Control.

Drag red sphere (left arm) and right arm automatically mirrors the motion
across the robot's centerline.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from synriard import get_model_path


def main():
    """Demo 3: Mirror symmetric dual-arm control."""
    
    # Model path (Bessica is a dual-arm robot)
    mjcf_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in mirror mode
    controller.run(mode='mirror')


if __name__ == '__main__':
    main()
