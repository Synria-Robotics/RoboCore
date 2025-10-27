#!/usr/bin/env python3
"""Demo 3: Mirror Symmetric Dual-Arm Control.

Drag red sphere (left arm) and right arm automatically mirrors the motion
across the robot's centerline.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from robocore.utils.path import get_robocore_path


def main():
    """Demo 3: Mirror symmetric dual-arm control."""
    
    # Model paths
    mjcf_path = get_robocore_path("assets/robot_descriptions/mjcf/Bessica-D_v1_0/Bessica_D_Covered_Interactive")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in mirror mode
    controller.run(mode='mirror')


if __name__ == '__main__':
    main()
