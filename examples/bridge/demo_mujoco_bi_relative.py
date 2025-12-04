#!/usr/bin/env python3
"""Demo 2: Cooperative Dual-Arm Control with Relative Constraint.

1. Drag red/blue spheres to set grasp configuration (relative pose)
2. Drag green sphere to move both arms cooperatively maintaining grasp

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from robotcore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from robotcore.utils.path import get_robocore_path


def main():
    """Demo 2: Cooperative dual-arm control with relative constraint."""
    
    # Model paths
    mjcf_path = get_robocore_path("assets/robot/mjcf/Bessica-D_v1_0/Bessica-D_Interactive.xml")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in cooperative mode
    controller.run(mode='relative')


if __name__ == '__main__':
    main()
