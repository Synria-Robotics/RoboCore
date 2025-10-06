"""Demo: Cooperative Carry with Hierarchical Task Priorities.

Demonstrates Phase 2.2 hierarchical IK solving the oscillation problem
from Phase 2.1's weighted approach.

Scenario:
1. Robot grasps object with both arms (relative pose constraint)
2. Move left arm to target destination
3. Hierarchical priority ensures grasp is maintained strictly

Copyright (c) 2025 Synria Robotics Co., Ltd.
License: GPL-3.0
"""

import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import Task, dual_ik_hierarchical, relative_pose_error


def main():
    print("="*70)
    print("  Phase 2.2: Hierarchical Cooperative Carry Demo")
    print("="*70)
    
    # Load models
    urdf_path = Path("robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    
    left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
    right = RobotModel(str(urdf_path), end_link='right_arm_gripper_right_finger')
    
    print(f"\n✓ Models loaded: {left.num_dof()} DOF each\n")
    
    # ========== Initial Grasp Configuration ==========
    print("Step 1: Establishing grasp configuration")
    print("-" * 70)
    
    q_grasp_left = np.array([0.0, 0.2, 0.0, -0.8, 0.0, 0.3, 0.0])
    q_grasp_right = np.array([0.0, -0.2, 0.0, -0.8, 0.0, -0.3, 0.0])
    
    T_left_grasp = left.fk(q_grasp_left, return_end=True)
    T_right_grasp = right.fk(q_grasp_right, return_end=True)
    
    if hasattr(T_left_grasp, 'detach'):
        T_left_grasp = T_left_grasp.detach().cpu().numpy()
        T_right_grasp = T_right_grasp.detach().cpu().numpy()
    
    # Record relative pose (PRIMARY constraint - must maintain)
    T_rel_grasp = np.linalg.inv(T_left_grasp) @ T_right_grasp
    
    p_left = T_left_grasp[0:3, 3]
    p_right = T_right_grasp[0:3, 3]
    p_rel = T_rel_grasp[0:3, 3]
    
    print(f"Left gripper:  [{p_left[0]:.3f}, {p_left[1]:.3f}, {p_left[2]:.3f}] m")
    print(f"Right gripper: [{p_right[0]:.3f}, {p_right[1]:.3f}, {p_right[2]:.3f}] m")
    print(f"Relative pos (left frame): [{p_rel[0]:.3f}, {p_rel[1]:.3f}, {p_rel[2]:.3f}] m")
    print(f"Grasp distance: {np.linalg.norm(p_left - p_right)*1000:.1f} mm\n")
    
    # ========== Define Movement Target ==========
    print("Step 2: Planning movement")
    print("-" * 70)
    
    # Move left arm 10cm forward and 5cm up
    T_left_target = T_left_grasp.copy()
    T_left_target[0, 3] += 0.10  # +10cm X
    T_left_target[2, 3] += 0.05  # +5cm Z
    
    p_target = T_left_target[0:3, 3]
    print(f"Target: [{p_target[0]:.3f}, {p_target[1]:.3f}, {p_target[2]:.3f}] m")
    print(f"Movement: Δx=+10cm, Δz=+5cm\n")
    
    # ========== Hierarchical Task Definition ==========
    print("Step 3: Configuring hierarchical tasks")
    print("-" * 70)
    
    # PRIMARY: Maintain grasp (relative position)
    primary_tasks = [
        Task(
            type='relative',
            target=T_rel_grasp,
            row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only (allow orientation freedom)
        ),
    ]
    
    # SECONDARY: Move left arm to target
    secondary_tasks = [
        Task(
            type='absolute_left',
            target=T_left_target,
            row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only
        ),
    ]
    
    print("Priority 0 (PRIMARY - strict):")
    print("  • Relative position: Maintain grasp distance")
    print("Priority 1 (SECONDARY - nullspace optimization):")
    print("  • Left arm position: Reach target\n")
    
    # ========== Solve Hierarchical IK ==========
    print("Step 4: Solving hierarchical IK")
    print("-" * 70)
    
    result = dual_ik_hierarchical(
        left, right,
        task_groups=[primary_tasks, secondary_tasks],
        q0_left=q_grasp_left,
        q0_right=q_grasp_right,
        max_iters=150,
        tol_primary=1e-3,     # 1mm for primary
        tol_secondary=5e-3,   # 5mm for secondary
        damping=1e-3,
        step_limit=0.10,
        verbose=True
    )
    
    print("\n" + "="*70)
    
    # ========== Validate Results ==========
    if result['success']:
        print(f"✓ SUCCESS: Primary task converged in {result['iters']} iterations")
        print(f"  Primary residual: {result['residuals'][0]*1000:.2f} mm")
        if len(result['residuals']) > 1:
            print(f"  Secondary residual: {result['residuals'][1]*1000:.2f} mm")
        
        # Get final poses
        q_left_final = np.array(result['q_left'])
        q_right_final = np.array(result['q_right'])
        
        T_left_final = left.fk(q_left_final, return_end=True)
        T_right_final = right.fk(q_right_final, return_end=True)
        
        if hasattr(T_left_final, 'detach'):
            T_left_final = T_left_final.detach().cpu().numpy()
            T_right_final = T_right_final.detach().cpu().numpy()
        
        # Validate primary constraint (relative pose)
        e_rel = relative_pose_error(T_left_final, T_right_final, T_rel_grasp)
        e_rel_pos = np.linalg.norm(e_rel[:3])
        
        print("\n" + "-"*70)
        print("PRIMARY CONSTRAINT VALIDATION (Grasp Maintenance)")
        print("-"*70)
        print(f"Relative position error: {e_rel_pos*1000:.2f} mm")
        
        # Check grasp distance preservation
        p_left_final = T_left_final[0:3, 3]
        p_right_final = T_right_final[0:3, 3]
        
        dist_initial = np.linalg.norm(p_left - p_right)
        dist_final = np.linalg.norm(p_left_final - p_right_final)
        dist_error = abs(dist_final - dist_initial)
        
        print(f"Grasp distance:")
        print(f"  Initial: {dist_initial*1000:.1f} mm")
        print(f"  Final:   {dist_final*1000:.1f} mm")
        print(f"  Change:  {dist_error*1000:.2f} mm")
        
        if e_rel_pos < 0.005:
            print("  ✓ PRIMARY SATISFIED (< 5mm)")
        else:
            print(f"  ⚠ PRIMARY DEGRADED ({e_rel_pos*1000:.1f}mm)")
        
        # Validate secondary task (left arm target)
        e_abs_pos = np.linalg.norm(p_left_final - T_left_target[0:3, 3])
        
        print("\n" + "-"*70)
        print("SECONDARY TASK VALIDATION (Target Reaching)")
        print("-"*70)
        print(f"Left arm position error: {e_abs_pos*1000:.2f} mm")
        print(f"Final position: [{p_left_final[0]:.3f}, {p_left_final[1]:.3f}, {p_left_final[2]:.3f}] m")
        
        if e_abs_pos < 0.015:
            print("  ✓ SECONDARY OPTIMIZED (< 15mm)")
        else:
            print(f"  ⚠ SECONDARY PARTIAL ({e_abs_pos*1000:.1f}mm)")
        
        # Joint configurations
        print("\n" + "-"*70)
        print("JOINT CONFIGURATIONS")
        print("-"*70)
        
        joint_change_L = np.linalg.norm(q_left_final - q_grasp_left)
        joint_change_R = np.linalg.norm(q_right_final - q_grasp_right)
        
        print(f"Left arm movement:  {joint_change_L:.3f} rad")
        print(f"Right arm movement: {joint_change_R:.3f} rad")
        
        # Overall assessment
        print("\n" + "="*70)
        print("OVERALL ASSESSMENT")
        print("="*70)
        
        if e_rel_pos < 0.005 and e_abs_pos < 0.020:
            print("✅ EXCELLENT: Both tasks satisfied!")
            print(f"   • Grasp maintained: {e_rel_pos*1000:.1f}mm error")
            print(f"   • Target reached: {e_abs_pos*1000:.1f}mm error")
            print(f"   • Moved object: {0.10:.2f}m forward, {0.05:.2f}m up")
        elif e_rel_pos < 0.005:
            print("✅ GOOD: Primary constraint strictly satisfied")
            print(f"   • Grasp: {e_rel_pos*1000:.1f}mm (excellent)")
            print(f"   • Target: {e_abs_pos*1000:.1f}mm (partial)")
        else:
            print("⚠️  PARTIAL: Check workspace limits")
        
    else:
        print(f"✗ FAILED: Primary task did not converge")
        print(f"  Primary residual: {result['residuals'][0]*1000:.1f} mm")
    
    print("="*70)


if __name__ == "__main__":
    main()
