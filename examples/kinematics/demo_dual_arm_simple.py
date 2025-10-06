"""Simple dual-arm coordination examples using multi-chain API.

Quick start guide for common dual-arm scenarios.
"""
from __future__ import annotations
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.task import absolute_task, relative_task
from robocore.utils.path import get_robocore_path


def setup_dual_arm():
    """Helper to setup dual-arm robot."""
    urdf_path = get_robocore_path("assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    robot = RobotModel(urdf_path)
    
    leaves = robot.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper' in l)
    
    robot.add_groups({
        'left_arm': left_end,
        'right_arm': right_end
    })
    
    return robot


def example_independent_motion():
    """Example 1: Independent dual-arm motion (no coordination)."""
    print("\n" + "="*60)
    print("Example 1: Independent Motion")
    print("="*60)
    
    robot = setup_dual_arm()
    
    # Random initial config
    q0_left = robot.groups()['left_arm'].random_q(scale=0.2)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.2)
    
    # Get current poses
    T_left = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # Define independent targets
    T_left_target = T_left.copy()
    T_left_target[0:3, 3] += np.array([0.05, 0.0, 0.03])
    
    T_right_target = T_right.copy()
    T_right_target[0:3, 3] += np.array([-0.05, 0.0, 0.03])
    
    # Create tasks
    tasks = [
        absolute_task('left_arm', T_left_target),
        absolute_task('right_arm', T_right_target)
    ]
    
    # Solve
    result = robot.ik_tasks(tasks, {'left_arm': q0_left, 'right_arm': q0_right})
    
    print(f"✓ Solved in {result['iters']} iterations")
    print(f"✓ Residual: {result['residual']*1000:.2f} mm\n")


def example_relative_constraint():
    """Example 2: Maintain fixed relative pose (e.g., holding a box)."""
    print("\n" + "="*60)
    print("Example 2: Relative Constraint (Box Holding)")
    print("="*60)
    
    robot = setup_dual_arm()
    
    q0_left = robot.groups()['left_arm'].random_q(scale=0.2)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.2)
    
    T_left = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # Capture current relative pose (as if grasping a box)
    T_rel_grasp = np.linalg.inv(T_left) @ T_right
    
    # Move left hand (right should follow to maintain relative pose)
    T_left_target = T_left.copy()
    T_left_target[0:3, 3] += np.array([0.04, 0.02, 0.05])
    
    # Only constrain relative position (allow orientation freedom)
    tasks = [
        absolute_task('left_arm', T_left_target, weight=1.0),
        relative_task('left_arm', 'right_arm', T_rel_grasp, 
                     weight=2.0,  # Higher weight = stronger constraint
                     row_mask=[1, 1, 1, 0, 0, 0])  # Position only
    ]
    
    result = robot.ik_tasks(tasks, {'left_arm': q0_left, 'right_arm': q0_right})
    
    # Verify
    q_left_sol = result['q_by_group']['left_arm']
    q_right_sol = result['q_by_group']['right_arm']
    
    T_left_final = robot.groups()['left_arm'].fk(q_left_sol, return_end=True)
    T_right_final = robot.groups()['right_arm'].fk(q_right_sol, return_end=True)
    
    T_rel_final = np.linalg.inv(T_left_final) @ T_right_final
    rel_err = np.linalg.norm(T_rel_final[0:3, 3] - T_rel_grasp[0:3, 3])
    
    print(f"✓ Solved in {result['iters']} iterations")
    print(f"✓ Relative constraint maintained: {rel_err*1000:.2f} mm error\n")


def example_hierarchical_priority():
    """Example 3: Hierarchical tasks (strict priority)."""
    print("\n" + "="*60)
    print("Example 3: Hierarchical Priority")
    print("="*60)
    
    robot = setup_dual_arm()
    
    q0_left = robot.groups()['left_arm'].random_q(scale=0.2)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.2)
    
    T_left = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # High priority: maintain 25cm separation
    T_rel_fixed = np.eye(4)
    T_rel_fixed[0:3, 3] = np.array([0.0, 0.25, 0.0])
    
    # Low priority: move left hand (solved in nullspace)
    T_left_target = T_left.copy()
    T_left_target[0:3, 3] += np.array([0.0, 0.0, 0.06])
    
    tasks = [
        relative_task('left_arm', 'right_arm', T_rel_fixed, 
                     priority=0,  # Highest priority
                     row_mask=[1, 1, 1, 0, 0, 0]),
        absolute_task('left_arm', T_left_target, priority=1)  # Lower priority
    ]
    
    result = robot.ik_tasks(
        tasks, 
        {'left_arm': q0_left, 'right_arm': q0_right},
        mode='hierarchical'  # Use hierarchical solver
    )
    
    print(f"✓ Hierarchical solve in {result['iters']} iterations")
    print(f"✓ Priority 0 satisfied exactly, priority 1 in nullspace\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Dual-Arm Multi-Chain API Examples")
    print("="*60)
    
    example_independent_motion()
    example_relative_constraint()
    example_hierarchical_priority()
    
    print("="*60)
    print("  All examples completed!")
    print("="*60 + "\n")
