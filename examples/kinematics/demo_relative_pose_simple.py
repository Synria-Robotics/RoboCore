"""Demo: Simple relative pose IK test.

Test the basic functionality of relative pose constraints without
complex conflicting tasks.

Copyright (c) 2025 Synria Robotics Co., Ltd.
License: GPL-3.0
"""

import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import Task, dual_ik_weighted, relative_pose_error
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import rpy_to_matrix


def main():
    print("="*70)
    print("  Phase 2.1: Simple Relative Pose Constraint Test")
    print("="*70)
    
    # Load models
    urdf_path = Path("robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")
    
    left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
    right = RobotModel(str(urdf_path), end_link='right_arm_gripper_right_finger')
    
    print(f"\n✓ Models loaded: {left.num_dof()} DOF each\n")
    
    # Simple initial configuration
    q_init_left = np.array([0.1, 0.2, 0.0, -0.6, 0.0, 0.2, 0.0])
    q_init_right = np.array([0.1, -0.2, 0.0, -0.6, 0.0, -0.2, 0.0])
    
    # Get initial relative pose
    T_L_init = left.fk(q_init_left, return_end=True)
    T_R_init = right.fk(q_init_right, return_end=True)
    
    if hasattr(T_L_init, 'detach'):
        T_L_init = T_L_init.detach().cpu().numpy()
        T_R_init = T_R_init.detach().cpu().numpy()
    
    T_rel_desired = np.linalg.inv(T_L_init) @ T_R_init
    p_rel = T_rel_desired[0:3, 3]
    
    print(f"Initial relative position: [{p_rel[0]:.3f}, {p_rel[1]:.3f}, {p_rel[2]:.3f}] m")
    print(f"Distance: {np.linalg.norm(p_rel):.3f} m\n")
    
    # Perturb the configuration
    q_perturbed_left = q_init_left + np.array([0.2, 0.1, -0.1, 0.15, -0.1, 0.1, 0.0])
    q_perturbed_right = q_init_right + np.array([0.2, -0.1, -0.1, 0.15, 0.1, -0.1, 0.0])
    
    # Check initial error
    T_L_pert = left.fk(q_perturbed_left, return_end=True)
    T_R_pert = right.fk(q_perturbed_right, return_end=True)
    
    if hasattr(T_L_pert, 'detach'):
        T_L_pert = T_L_pert.detach().cpu().numpy()
        T_R_pert = T_R_pert.detach().cpu().numpy()
    
    e_init = relative_pose_error(T_L_pert, T_R_pert, T_rel_desired)
    print(f"Initial error (perturbed config):")
    print(f"  Position: {np.linalg.norm(e_init[:3])*1000:.1f} mm")
    print(f"  Orientation: {np.linalg.norm(e_init[3:])*1000:.1f} mrad\n")
    
    # Solve IK with relative constraint only
    print("Solving IK (relative constraint only)...")
    
    tasks = [
        Task(
            type='relative',
            target=T_rel_desired,
            weight=1.0,
            row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only for simplicity
        ),
    ]
    
    result = dual_ik_weighted(
        left, right,
        tasks=tasks,
        q0_left=q_perturbed_left,
        q0_right=q_perturbed_right,
        max_iters=50,
        tol=5e-3,
        damping=1e-2,
        verbose=True
    )
    
    print("\n" + "="*70)
    if result['success']:
        print(f"✓ SUCCESS: Converged in {result['iters']} iterations")
        print(f"  Final residual: {result['residual']*1000:.2f} mm")
        
        # Validate
        q_L_final = np.array(result['q_left'])
        q_R_final = np.array(result['q_right'])
        
        T_L_final = left.fk(q_L_final, return_end=True)
        T_R_final = right.fk(q_R_final, return_end=True)
        
        if hasattr(T_L_final, 'detach'):
            T_L_final = T_L_final.detach().cpu().numpy()
            T_R_final = T_R_final.detach().cpu().numpy()
        
        e_final = relative_pose_error(T_L_final, T_R_final, T_rel_desired)
        
        print(f"\nFinal error:")
        print(f"  Position: {np.linalg.norm(e_final[:3])*1000:.2f} mm")
        print(f"  Orientation: {np.linalg.norm(e_final[3:])*1000:.2f} mrad")
        
        T_rel_final = np.linalg.inv(T_L_final) @ T_R_final
        p_rel_final = T_rel_final[0:3, 3]
        print(f"\nFinal relative position: [{p_rel_final[0]:.3f}, {p_rel_final[1]:.3f}, {p_rel_final[2]:.3f}] m")
        print(f"Distance: {np.linalg.norm(p_rel_final):.3f} m")
        
    else:
        print(f"✗ FAILED: Did not converge (residual: {result['residual']*1000:.1f} mm)")
    
    print("="*70)


if __name__ == "__main__":
    main()
