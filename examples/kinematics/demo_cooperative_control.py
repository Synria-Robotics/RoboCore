"""Bimanual cooperative control examples using Group + Task framework.

Demonstrates how single-arm API naturally extends to multi-link/multi-task control.
"""
import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.task import absolute_task, relative_task, centering_task


def demo_bimanual_cooperative_control():
    """Demonstrate bimanual cooperative control with Group + Task framework."""
    
    # Load dual-arm robot model (Bessica)
    urdf_path = Path(__file__).parent.parent.parent / "robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"
    robot = RobotModel(str(urdf_path))
    
    # Define groups (left_arm, right_arm)
    groups = robot.add_groups({
        'left_arm': 'left_arm_link7',
        'right_arm': 'right_arm_link7'
    })
    
    print("Available groups:", list(robot.groups().keys()))
    print("Left arm info:", robot.group('left_arm'))
    
    # Initial configurations
    q0_left = np.zeros(7)  # 7-DOF left arm
    q0_right = np.zeros(7)  # 7-DOF right arm
    
    # Example 1: Absolute-Absolute (two hands reach independent targets)
    print("\n=== Example 1: Absolute-Absolute ===")
    T_left_target = np.eye(4)
    T_left_target[0:3, 3] = [0.5, 0.3, 0.8]  # Position target
    
    T_right_target = np.eye(4)
    T_right_target[0:3, 3] = [0.5, -0.3, 0.8]  # Position target
    
    result1 = robot.ik_bimanual_absolute(
        T_left_target, T_right_target,
        q0_left, q0_right,
        weights={'left': 1.0, 'right': 1.0},
        max_iters=50, verbose=True
    )
    print(f"Success: {result1['success']}")
    print(f"Final residual: {result1['residual']:.6f}")
    
    # Example 2: Relative constraint (maintain relative pose for grasping)
    print("\n=== Example 2: Relative Constraint ===")
    T_rel_desired = np.eye(4)
    T_rel_desired[0:3, 3] = [0.0, 0.0, 0.1]  # 10cm apart in Z
    
    result2 = robot.ik_bimanual_relative(
        T_rel_desired,
        q0_left, q0_right,
        weight=2.0,
        max_iters=50, verbose=True
    )
    print(f"Success: {result2['success']}")
    print(f"Final residual: {result2['residual']:.6f}")
    
    # Example 3: Mixed (absolute + relative)
    print("\n=== Example 3: Mixed Absolute + Relative ===")
    T_left_target = np.eye(4)
    T_left_target[0:3, 3] = [0.6, 0.2, 0.7]
    
    T_right_target = np.eye(4)
    T_right_target[0:3, 3] = [0.6, -0.2, 0.7]
    
    T_rel_desired = np.eye(4)
    T_rel_desired[0:3, 3] = [0.0, 0.0, 0.15]  # 15cm apart
    
    result3 = robot.ik_bimanual_mixed(
        T_left_target, T_right_target, T_rel_desired,
        q0_left, q0_right,
        weights={'left': 1.0, 'right': 1.0, 'relative': 3.0},  # Prioritize relative
        max_iters=100, verbose=True
    )
    print(f"Success: {result3['success']}")
    print(f"Final residual: {result3['residual']:.6f}")
    
    # Example 4: Using unified task interface
    print("\n=== Example 4: Unified Task Interface ===")
    tasks = [
        absolute_task('left_arm', T_left_target, weight=1.0),
        absolute_task('right_arm', T_right_target, weight=1.0),
        relative_task('left_arm', 'right_arm', T_rel_desired, weight=2.0)
    ]
    
    result4 = robot.ik_tasks(
        tasks, 
        {'left_arm': q0_left, 'right_arm': q0_right},
        mode='weighted',
        max_iters=100, verbose=True
    )
    print(f"Success: {result4['success']}")
    print(f"Final residual: {result4['residual']:.6f}")


if __name__ == "__main__":
    demo_bimanual_cooperative_control()
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("The Group + Task framework is ready for future humanoid extension.")
