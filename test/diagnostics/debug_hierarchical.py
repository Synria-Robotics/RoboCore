"""Debug hierarchical IK convergence."""

import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import Task, dual_ik_hierarchical, relative_pose_error

urdf_path = Path("robocore/assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")

left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
right = RobotModel(str(urdf_path), end_link='right_arm_gripper_right_finger')

# Grasp config
q_grasp_left = np.array([0.0, 0.2, 0.0, -0.8, 0.0, 0.3, 0.0])
q_grasp_right = np.array([0.0, -0.2, 0.0, -0.8, 0.0, -0.3, 0.0])

T_L_grasp = left.fk(q_grasp_left, return_end=True).detach().cpu().numpy()
T_R_grasp = right.fk(q_grasp_right, return_end=True).detach().cpu().numpy()

T_rel_grasp = np.linalg.inv(T_L_grasp) @ T_R_grasp

# Check initial error
e_init = relative_pose_error(T_L_grasp, T_R_grasp, T_rel_grasp)
print(f"Initial relative pose error: {np.linalg.norm(e_init[:3])*1000:.6f} mm")
print(f"This should be ~0 since we're using the same config")

# Target
T_left_target = T_L_grasp.copy()
T_left_target[0, 3] += 0.10
T_left_target[2, 3] += 0.05

# Try starting from same config
print("\n--- Test 1: Starting from grasp config (should have error) ---")

primary_tasks = [
    Task(
        type='relative',
        target=T_rel_grasp,
        row_mask=np.array([1, 1, 1, 0, 0, 0])
    ),
]

secondary_tasks = [
    Task(
        type='absolute_left',
        target=T_left_target,
        row_mask=np.array([1, 1, 1, 0, 0, 0])
    ),
]

print(f"Left arm current: {T_L_grasp[0:3, 3]}")
print(f"Left arm target:  {T_left_target[0:3, 3]}")
print(f"Initial left arm error: {np.linalg.norm(T_L_grasp[0:3, 3] - T_left_target[0:3, 3])*1000:.1f} mm")

result = dual_ik_hierarchical(
    left, right,
    task_groups=[primary_tasks, secondary_tasks],
    q0_left=q_grasp_left,
    q0_right=q_grasp_right,
    max_iters=150,
    tol_primary=1e-3,
    damping=1e-3,
    verbose=True
)

print(f"\nResult: {result['success']}, iters={result['iters']}")
print(f"Residuals: {[f'{r*1000:.2f}mm' for r in result['residuals']]}")

q_L_final = np.array(result['q_left'])
q_R_final = np.array(result['q_right'])

T_L_final = left.fk(q_L_final, return_end=True).detach().cpu().numpy()
T_R_final = right.fk(q_R_final, return_end=True).detach().cpu().numpy()

e_rel_final = relative_pose_error(T_L_final, T_R_final, T_rel_grasp)
e_abs_final = T_L_final[0:3, 3] - T_left_target[0:3, 3]

print(f"\nFinal relative error: {np.linalg.norm(e_rel_final[:3])*1000:.2f} mm")
print(f"Final absolute error: {np.linalg.norm(e_abs_final)*1000:.2f} mm")
