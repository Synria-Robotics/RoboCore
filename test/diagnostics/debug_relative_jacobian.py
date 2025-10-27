"""Debug relative Jacobian computation."""

import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import relative_jacobian
from robocore.transform.conversions import matrix_to_axis_angle

# Load models
urdf_path = Path("robocore/assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf")

left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
right = RobotModel(str(urdf_path), end_link='right_arm_gripper_left_finger')

q_left = np.array([0.2, 0.3, -0.1, -0.8, 0.2, 0.3, 0.0])
q_right = np.array([0.2, -0.3, -0.1, -0.8, 0.2, -0.3, 0.0])

# Analytical
J_analytical = relative_jacobian(left, right, q_left, q_right)
print("Analytical Jacobian shape:", J_analytical.shape)
print("Analytical J[0, :]:", J_analytical[0, :])

# Numerical differentiation
eps = 1e-6

def rel_pose_vector(qL, qR):
    """Convert relative pose to 6D vector."""
    TL = left.fk(qL, return_end=True)
    TR = right.fk(qR, return_end=True)
    if hasattr(TL, 'detach'):
        TL = TL.detach().cpu().numpy()
        TR = TR.detach().cpu().numpy()
    
    T_rel = np.linalg.inv(TL) @ TR
    
    p = T_rel[0:3, 3]
    axis, angle = matrix_to_axis_angle(T_rel[0:3, 0:3])
    return np.concatenate([p, axis * angle])

q_combined = np.concatenate([q_left, q_right])
J_numerical = np.zeros((6, 14))

p0 = rel_pose_vector(q_left, q_right)
print("\nBase relative pose (6D):", p0)

for i in range(14):
    q_plus = q_combined.copy()
    q_plus[i] += eps
    
    qL_p = q_plus[:7]
    qR_p = q_plus[7:]
    
    p_plus = rel_pose_vector(qL_p, qR_p)
    J_numerical[:, i] = (p_plus - p0) / eps

print("\nNumerical J[0, :]:", J_numerical[0, :])

# Check left arm FK
TL = left.fk(q_left, return_end=True)
TR = right.fk(q_right, return_end=True)
if hasattr(TL, 'detach'):
    TL = TL.detach().cpu().numpy()
    TR = TR.detach().cpu().numpy()

print("\nLeft arm pose:")
print(TL)
print("\nRight arm pose:")
print(TR)

T_rel = np.linalg.inv(TL) @ TR
print("\nRelative pose:")
print(T_rel)

# Compare
print("\n\nDifference:")
print(J_analytical - J_numerical)
print("\nMax absolute difference:", np.max(np.abs(J_analytical - J_numerical)))
