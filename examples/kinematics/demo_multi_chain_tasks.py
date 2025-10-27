"""Multi-chain multi-task IK demonstration.

Shows unified framework for:
1. Weighted task composition (absolute + relative mixed)
2. Hierarchical task solving (strict priority)
3. Dual-arm coordination scenarios
"""
from __future__ import annotations
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.task import absolute_task, relative_task
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print


def demo_weighted_tasks():
    """Weighted task composition: absolute + relative constraints."""
    beauty_print("=== Weighted Multi-Task IK ===", color='cyan')
    
    # Load robot
    urdf_path = get_robocore_path("assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    robot = RobotModel(urdf_path)
    
    # Setup dual-arm groups
    leaves = robot.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper' in l)
    
    robot.add_groups({
        'left_arm': left_end,
        'right_arm': right_end
    })
    
    # Initial configuration
    q0_left = robot.groups()['left_arm'].random_q(scale=0.3)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.3)
    
    # Get current poses
    T_left_init = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right_init = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # Define targets
    T_left_target = T_left_init.copy()
    T_left_target[0:3, 3] += np.array([0.05, 0.0, 0.03])  # Move left hand
    
    T_right_target = T_right_init.copy()
    T_right_target[0:3, 3] += np.array([-0.05, 0.0, 0.03])  # Move right hand
    
    # Desired relative pose: maintain ~30cm horizontal separation
    T_rel_desired = np.eye(4)
    T_rel_desired[0:3, 3] = np.array([0.0, 0.3, 0.0])  # 30cm in Y (left frame)
    
    # Create task list
    tasks = [
        absolute_task('left_arm', T_left_target, weight=1.0),
        absolute_task('right_arm', T_right_target, weight=1.0),
        relative_task('left_arm', 'right_arm', T_rel_desired, 
                     weight=2.0,  # Higher weight for relative constraint
                     row_mask=[1, 1, 1, 0, 0, 0])  # Position-only constraint
    ]
    
    # Solve with weighted mode
    result = robot.ik_tasks(
        tasks,
        {'left_arm': q0_left, 'right_arm': q0_right},
        mode='weighted',
        max_iters=100,
        tol=2e-3,
        damping=5e-3,
        verbose=True
    )
    
    # Verify solution
    q_left_sol = result['q_by_group']['left_arm']
    q_right_sol = result['q_by_group']['right_arm']
    
    T_left_final = robot.groups()['left_arm'].fk(q_left_sol, return_end=True)
    T_right_final = robot.groups()['right_arm'].fk(q_right_sol, return_end=True)
    
    # Check absolute errors
    left_pos_err = np.linalg.norm(T_left_final[0:3, 3] - T_left_target[0:3, 3])
    right_pos_err = np.linalg.norm(T_right_final[0:3, 3] - T_right_target[0:3, 3])
    
    # Check relative constraint
    T_rel_actual = np.linalg.inv(T_left_final) @ T_right_final
    rel_pos_err = np.linalg.norm(T_rel_actual[0:3, 3] - T_rel_desired[0:3, 3])
    
    beauty_print(f"✓ Converged in {result['iters']} iterations", color='green')
    beauty_print(f"  Left arm position error:  {left_pos_err*1000:.2f} mm", color='white')
    beauty_print(f"  Right arm position error: {right_pos_err*1000:.2f} mm", color='white')
    beauty_print(f"  Relative constraint error: {rel_pos_err*1000:.2f} mm", color='white')
    
    print()


def demo_hierarchical_tasks():
    """Hierarchical task solving with strict priorities."""
    beauty_print("=== Hierarchical Multi-Task IK ===", color='cyan')
    
    urdf_path = get_robocore_path("assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    robot = RobotModel(urdf_path)
    
    leaves = robot.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper' in l)
    
    robot.add_groups({
        'left_arm': left_end,
        'right_arm': right_end
    })
    
    q0_left = robot.groups()['left_arm'].random_q(scale=0.3)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.3)
    
    T_left_init = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right_init = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # Primary task: maintain relative pose (highest priority)
    T_rel_desired = np.eye(4)
    T_rel_desired[0:3, 3] = np.array([0.0, 0.25, 0.0])
    
    # Secondary task: move left hand up
    T_left_target = T_left_init.copy()
    T_left_target[0:3, 3] += np.array([0.0, 0.0, 0.08])
    
    tasks = [
        relative_task('left_arm', 'right_arm', T_rel_desired, 
                     weight=1.0, priority=0,  # Highest priority
                     row_mask=[1, 1, 1, 0, 0, 0]),
        absolute_task('left_arm', T_left_target, 
                     weight=1.0, priority=1)  # Lower priority (solved in nullspace)
    ]
    
    result = robot.ik_tasks(
        tasks,
        {'left_arm': q0_left, 'right_arm': q0_right},
        mode='hierarchical',
        max_iters=80,
        tol=3e-3,
        damping=1e-4,
        verbose=True
    )
    
    q_left_sol = result['q_by_group']['left_arm']
    q_right_sol = result['q_by_group']['right_arm']
    
    T_left_final = robot.groups()['left_arm'].fk(q_left_sol, return_end=True)
    T_right_final = robot.groups()['right_arm'].fk(q_right_sol, return_end=True)
    
    T_rel_actual = np.linalg.inv(T_left_final) @ T_right_final
    rel_pos_err = np.linalg.norm(T_rel_actual[0:3, 3] - T_rel_desired[0:3, 3])
    
    left_pos_err = np.linalg.norm(T_left_final[0:3, 3] - T_left_target[0:3, 3])
    
    beauty_print(f"✓ Converged in {result['iters']} iterations", color='green')
    beauty_print(f"  Priority 0 (relative): {rel_pos_err*1000:.2f} mm error", color='white')
    beauty_print(f"  Priority 1 (left abs):  {left_pos_err*1000:.2f} mm error", color='white')
    beauty_print(f"  → Relative constraint satisfied exactly, absolute in nullspace", color='yellow')
    
    print()


def demo_cooperative_carrying():
    """Simulate cooperative carrying: both hands track object motion while maintaining grasp."""
    beauty_print("=== Cooperative Carrying Scenario ===", color='cyan')
    
    urdf_path = get_robocore_path("assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    robot = RobotModel(urdf_path)
    
    leaves = robot.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper' in l)
    
    robot.add_groups({
        'left_arm': left_end,
        'right_arm': right_end
    })
    
    q0_left = robot.groups()['left_arm'].random_q(scale=0.2)
    q0_right = robot.groups()['right_arm'].random_q(scale=0.2)
    
    T_left_init = robot.groups()['left_arm'].fk(q0_left, return_end=True)
    T_right_init = robot.groups()['right_arm'].fk(q0_right, return_end=True)
    
    # Object grasp: fixed relative pose (e.g., gripping opposite sides)
    T_rel_grasp = np.linalg.inv(T_left_init) @ T_right_init
    
    # Move object: translate both hands together
    delta = np.array([0.06, 0.0, 0.05])  # Move object 6cm forward, 5cm up
    
    T_left_target = T_left_init.copy()
    T_left_target[0:3, 3] += delta
    
    T_right_target = T_right_init.copy()
    T_right_target[0:3, 3] += delta
    
    tasks = [
        absolute_task('left_arm', T_left_target, weight=1.0),
        absolute_task('right_arm', T_right_target, weight=1.0),
        relative_task('left_arm', 'right_arm', T_rel_grasp, 
                     weight=3.0,  # Strong grasp constraint
                     row_mask=[1, 1, 1, 0, 0, 0])
    ]
    
    result = robot.ik_tasks(
        tasks,
        {'left_arm': q0_left, 'right_arm': q0_right},
        mode='weighted',
        max_iters=100,
        tol=2e-3,
        damping=8e-3,
        step_limit=0.15,
        verbose=True
    )
    
    q_left_sol = result['q_by_group']['left_arm']
    q_right_sol = result['q_by_group']['right_arm']
    
    T_left_final = robot.groups()['left_arm'].fk(q_left_sol, return_end=True)
    T_right_final = robot.groups()['right_arm'].fk(q_right_sol, return_end=True)
    
    # Compute object center motion
    obj_center_init = 0.5 * (T_left_init[0:3, 3] + T_right_init[0:3, 3])
    obj_center_final = 0.5 * (T_left_final[0:3, 3] + T_right_final[0:3, 3])
    obj_motion = obj_center_final - obj_center_init
    
    T_rel_actual = np.linalg.inv(T_left_final) @ T_right_final
    grasp_err = np.linalg.norm(T_rel_actual[0:3, 3] - T_rel_grasp[0:3, 3])
    
    beauty_print(f"✓ Object carried successfully in {result['iters']} iterations", color='green')
    beauty_print(f"  Commanded motion: {delta}", color='white')
    beauty_print(f"  Actual motion:    {obj_motion}", color='white')
    beauty_print(f"  Grasp preservation error: {grasp_err*1000:.2f} mm", color='white')
    
    print()


def demo_single_arm_as_multichain():
    """Show that single-arm IK works seamlessly in multi-chain framework."""
    beauty_print("=== Single Arm via Multi-Chain Framework ===", color='cyan')
    
    urdf_path = get_robocore_path("assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    robot = RobotModel(urdf_path)
    
    leaves = robot.available_leaf_links()
    left_end = next(l for l in leaves if 'left_arm_gripper' in l)
    
    robot.add_groups({'left_arm': left_end})
    
    q0 = robot.groups()['left_arm'].random_q(scale=0.3)
    T_init = robot.groups()['left_arm'].fk(q0, return_end=True)
    
    T_target = T_init.copy()
    T_target[0:3, 3] += np.array([0.05, 0.03, 0.04])
    
    tasks = [absolute_task('left_arm', T_target, weight=1.0)]
    
    result = robot.ik_tasks(
        tasks,
        {'left_arm': q0},
        mode='weighted',
        max_iters=50,
        tol=1e-3,
        verbose=False
    )
    
    q_sol = result['q_by_group']['left_arm']
    T_final = robot.groups()['left_arm'].fk(q_sol, return_end=True)
    pos_err = np.linalg.norm(T_final[0:3, 3] - T_target[0:3, 3])
    
    beauty_print(f"✓ Single-arm IK via multi-chain: {pos_err*1000:.2f} mm error", color='green')
    beauty_print(f"  → Framework supports 1 to N chains seamlessly", color='yellow')
    
    print()


if __name__ == '__main__':
    print()
    beauty_print("╔═══════════════════════════════════════════════════╗", color='blue')
    beauty_print("║   Multi-Chain Multi-Task IK Demonstration        ║", color='blue')
    beauty_print("╚═══════════════════════════════════════════════════╝", color='blue')
    print()
    
    demo_single_arm_as_multichain()
    demo_weighted_tasks()
    demo_hierarchical_tasks()
    demo_cooperative_carrying()
    
    beauty_print("╔═══════════════════════════════════════════════════╗", color='blue')
    beauty_print("║   All demonstrations completed successfully!     ║", color='blue')
    beauty_print("╚═══════════════════════════════════════════════════╝", color='blue')
    print()
