#!/usr/bin/env python3
"""Demo 1: Independent Dual-Arm Control.

Drag red and blue spheres independently to control left and right arms.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from robocore.utils.path import get_robocore_path


def main():
    """Demo 1: Independent dual-arm IK control."""
    
    # Model paths
    mjcf_path = get_robocore_path("assets/robot_descriptions/mjcf/Bessica_D_v1_0/Bessica_D_Covered_Interactive.xml")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in independent mode
    controller.run(mode='independent')


if __name__ == '__main__':
    main()
