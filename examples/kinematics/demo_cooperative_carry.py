"""Demo: Phase 2.1 Cooperative Carry - Weighted Task Composition.

Demonstrates:
1. Recording relative pose during initial grasp
2. Moving left arm to target while maintaining relative pose
3. Weighted task composition (relative + absolute)

Scenario: Robot grasps an object with both arms, then carries it to a new location
while maintaining the grasp configuration.

Copyright (c) 2025 Synria Robotics Co., Ltd.
License: GPL-3.0
"""

import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import Task, dual_ik_weighted, relative_pose_error
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import rpy_to_matrix
from robocore.utils.beauty_logger import beauty_print


def main():
    beauty_print("🤖 Phase 2.1 Demo: Cooperative Carry with Weighted Tasks", type="module")
    
    # ========== Step 1: Load dual-arm model ==========
    beauty_print("\n📦 Step 1: Loading Bessica dual-arm model", type="module")
    
    urdf_path = Path("robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    
    left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
    right = RobotModel(str(urdf_path), end_link='right_arm_gripper_right_finger')
    
    beauty_print(f"✓ Left arm: {left.num_dof()} DOF", type="success")
    beauty_print(f"✓ Right arm: {right.num_dof()} DOF", type="success")
    
    # ========== Step 2: Define initial grasp configuration ==========
    beauty_print("\n🔧 Step 2: Setting initial grasp configuration", type="module")
    
    # Arms in front, holding object (more conservative pose)
    q_grasp_left = np.array([0.0, 0.2, 0.0, -0.8, 0.0, 0.3, 0.0])
    q_grasp_right = np.array([0.0, -0.2, 0.0, -0.8, 0.0, -0.3, 0.0])
    
    T_left_grasp = left.fk(q_grasp_left, return_end=True)
    T_right_grasp = right.fk(q_grasp_right, return_end=True)
    
    if hasattr(T_left_grasp, 'detach'):
        T_left_grasp = T_left_grasp.detach().cpu().numpy()
        T_right_grasp = T_right_grasp.detach().cpu().numpy()
    
    # Record relative pose (this is what we want to maintain)
    T_rel_grasp = np.linalg.inv(T_left_grasp) @ T_right_grasp
    
    p_left = T_left_grasp[0:3, 3]
    p_right = T_right_grasp[0:3, 3]
    p_rel = T_rel_grasp[0:3, 3]
    
    beauty_print(f"Left gripper position: [{p_left[0]:.3f}, {p_left[1]:.3f}, {p_left[2]:.3f}] m", type="info")
    beauty_print(f"Right gripper position: [{p_right[0]:.3f}, {p_right[1]:.3f}, {p_right[2]:.3f}] m", type="info")
    beauty_print(f"Relative position (in left frame): [{p_rel[0]:.3f}, {p_rel[1]:.3f}, {p_rel[2]:.3f}] m", type="info")
    beauty_print(f"Grasp distance: {np.linalg.norm(p_left - p_right):.3f} m", type="success")
    
    # ========== Step 3: Define target destination ==========
    beauty_print("\n🎯 Step 3: Planning movement to target location", type="module")
    
    # Move left arm 10cm forward (X+) and 5cm up (Z+) - smaller movement
    T_left_target = T_left_grasp.copy()
    T_left_target[0, 3] += 0.10  # +10cm in X
    T_left_target[2, 3] += 0.05  # +5cm in Z
    
    p_target = T_left_target[0:3, 3]
    beauty_print(f"Target left position: [{p_target[0]:.3f}, {p_target[1]:.3f}, {p_target[2]:.3f}] m", type="info")
    beauty_print(f"Movement: Δx={0.10:.2f}m, Δz={0.05:.2f}m", type="info")
    
    # ========== Step 4: Solve weighted IK ==========
    beauty_print("\n🔄 Step 4: Solving weighted cooperative IK", type="module")
    
    # Define tasks
    tasks = [
        Task(
            type='relative',
            target=T_rel_grasp,
            weight=3.0,  # High weight: maintain grasp
            priority=0,
            row_mask=np.array([1, 1, 1, 0, 0, 0])  # Only position (allow orientation freedom)
        ),
        Task(
            type='absolute_left',
            target=T_left_target,
            weight=1.0,  # Lower weight: reach target
            priority=0
        ),
    ]
    
    beauty_print("📋 Tasks configured:", type="info")
    beauty_print("  • Relative position (weight=3.0): Maintain grasp distance (position only)", type="info")
    beauty_print("  • Left arm absolute (weight=1.0): Move to target (full 6D)", type="info")
    
    # Solve IK
    result = dual_ik_weighted(
        left, right,
        tasks=tasks,
        q0_left=q_grasp_left,
        q0_right=q_grasp_right,
        max_iters=200,
        tol=1e-2,  # 10mm total tolerance
        damping=5e-3,
        step_limit=0.15,
        verbose=True
    )
    
    # ========== Step 5: Validate solution ==========
    beauty_print("\n✅ Step 5: Validating solution", type="module")
    
    if result['success']:
        beauty_print(f"✓ IK converged in {result['iters']} iterations", type="success")
        beauty_print(f"  Final residual: {result['residual']*1000:.2f} mm", type="success")
    else:
        print(f"⚠️ IK failed to converge (residual: {result['residual']*1000:.2f} mm)")
        print("Note: This may indicate target is outside workspace or incompatible constraints")
        return
    
    # Get final poses
    q_left_final = np.array(result['q_left'])
    q_right_final = np.array(result['q_right'])
    
    T_left_final = left.fk(q_left_final, return_end=True)
    T_right_final = right.fk(q_right_final, return_end=True)
    
    if hasattr(T_left_final, 'detach'):
        T_left_final = T_left_final.detach().cpu().numpy()
        T_right_final = T_right_final.detach().cpu().numpy()
    
    # Check relative pose error
    e_rel = relative_pose_error(T_left_final, T_right_final, T_rel_grasp)
    e_rel_pos = np.linalg.norm(e_rel[:3])
    e_rel_ori = np.linalg.norm(e_rel[3:])
    
    beauty_print("\n🔍 Relative pose validation:", type="module")
    beauty_print(f"  Position error: {e_rel_pos*1000:.2f} mm", type="success" if e_rel_pos < 0.01 else "warning")
    beauty_print(f"  Orientation error: {e_rel_ori*1000:.2f} mrad", type="success" if e_rel_ori < 0.02 else "warning")
    
    # Check absolute pose error
    p_left_final = T_left_final[0:3, 3]
    e_abs_pos = np.linalg.norm(p_left_final - T_left_target[0:3, 3])
    
    beauty_print("\n🎯 Absolute position validation:", type="module")
    beauty_print(f"  Target reached: {e_abs_pos*1000:.2f} mm error", type="success" if e_abs_pos < 0.01 else "warning")
    beauty_print(f"  Final left position: [{p_left_final[0]:.3f}, {p_left_final[1]:.3f}, {p_left_final[2]:.3f}] m", type="info")
    
    # Check grasp distance preservation
    p_right_final = T_right_final[0:3, 3]
    dist_final = np.linalg.norm(p_left_final - p_right_final)
    dist_initial = np.linalg.norm(p_left - p_right)
    dist_change = abs(dist_final - dist_initial)
    
    beauty_print("\n📏 Grasp distance preservation:", type="module")
    beauty_print(f"  Initial distance: {dist_initial*1000:.1f} mm", type="info")
    beauty_print(f"  Final distance: {dist_final*1000:.1f} mm", type="info")
    beauty_print(f"  Change: {dist_change*1000:.2f} mm", type="success" if dist_change < 0.01 else "warning")
    
    # ========== Step 6: Display joint configurations ==========
    beauty_print("\n🦾 Step 6: Joint configurations", type="module")
    
    beauty_print("Initial joints (grasp):", type="info")
    beauty_print(f"  Left:  {np.array2string(q_grasp_left, precision=3, suppress_small=True)}", type="info")
    beauty_print(f"  Right: {np.array2string(q_grasp_right, precision=3, suppress_small=True)}", type="info")
    
    beauty_print("\nFinal joints (after carry):", type="info")
    beauty_print(f"  Left:  {np.array2string(q_left_final, precision=3, suppress_small=True)}", type="info")
    beauty_print(f"  Right: {np.array2string(q_right_final, precision=3, suppress_small=True)}", type="info")
    
    joint_change_L = np.linalg.norm(q_left_final - q_grasp_left)
    joint_change_R = np.linalg.norm(q_right_final - q_grasp_right)
    
    beauty_print(f"\nJoint space movement:", type="info")
    beauty_print(f"  Left arm: {joint_change_L:.3f} rad", type="info")
    beauty_print(f"  Right arm: {joint_change_R:.3f} rad", type="info")
    
    # ========== Summary ==========
    beauty_print("\n" + "="*70, type="module")
    beauty_print("📊 DEMO SUMMARY", type="module")
    beauty_print("="*70, type="module")
    
    if result['success'] and e_rel_pos < 0.01 and e_abs_pos < 0.01:
        beauty_print("✅ SUCCESS: Cooperative carry completed!", type="success")
        beauty_print(f"   • Moved object {0.10:.2f}m forward and {0.05:.2f}m up", type="success")
        beauty_print(f"   • Grasp maintained within {e_rel_pos*1000:.1f}mm precision", type="success")
        beauty_print(f"   • Target reached with {e_abs_pos*1000:.1f}mm accuracy", type="success")
    else:
        beauty_print("⚠️  PARTIAL SUCCESS: Some tolerances exceeded", type="warning")


if __name__ == "__main__":
    main()
