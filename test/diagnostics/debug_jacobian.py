#!/usr/bin/env python3
"""快速诊断双臂雅克比问题"""

import numpy as np
from robocore.modeling.robot_model import BimanualRobotModel
from robocore.utils.path import get_robocore_path

# 加载机器人
mjcf_path = get_robocore_path("assets/robot/mjcf/Bessica-D_v1_0/Bessica-D_Interactive.xml")
robot = BimanualRobotModel(mjcf_path, "left_arm_link7", "right_arm_link7")

# 测试配置
q_left = np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.1, 0.0])
q_right = np.array([0.1, -0.2, 0.1, -0.3, 0.0, -0.1, 0.0])

print("=" * 80)
print("双臂雅克比快速诊断")
print("=" * 80)

# 1. 测试单臂雅克比
print("\n1. 单臂雅克比检查:")
J_left = robot.left_model.jacobian(q_left, backend='numpy')
J_right = robot.right_model.jacobian(q_right, backend='numpy')
print(f"   左臂雅克比形状: {J_left.shape}")
print(f"   右臂雅克比形状: {J_right.shape}")

# 2. 测试双臂雅克比（independent）
print("\n2. Block-diagonal 雅克比:")
J_bi = robot.jacobian(q_left, q_right, backend='numpy', mode='indep')
print(f"   形状: {J_bi.shape}")
print(f"   左上块 (0:6, 0:7):\n{J_bi[0:6, 0:7]}")
print(f"   右下块 (6:12, 7:14):\n{J_bi[6:12, 7:14]}")

# 3. 对比单臂雅克比
print("\n3. 验证块拼接:")
print(f"   左臂单独计算:\n{J_left}")
print(f"   从双臂提取:\n{J_bi[0:6, 0:7]}")
print(f"   差异: {np.max(np.abs(J_left - J_bi[0:6, 0:7])):.2e}")

print(f"\n   右臂单独计算:\n{J_right}")
print(f"   从双臂提取:\n{J_bi[6:12, 7:14]}")
print(f"   差异: {np.max(np.abs(J_right - J_bi[6:12, 7:14])):.2e}")

# 4. 数值微分验证（只验证右臂第一个关节）
print("\n4. 数值微分验证 (右臂关节0):")
epsilon = 1e-6
q_right_plus = q_right.copy()
q_right_plus[0] += epsilon

result_0 = robot.fk(q_left, q_right, backend='numpy', mode='indep')
result_plus = robot.fk(q_left, q_right_plus, backend='numpy', mode='indep')

T_right_0 = np.array(result_0['right'])
T_right_plus = np.array(result_plus['right'])

# 位置导数
dp_num = (T_right_plus[:3, 3] - T_right_0[:3, 3]) / epsilon
dp_ana = J_bi[6:9, 7]  # 右臂位置，关节0

print(f"   数值微分: {dp_num}")
print(f"   解析雅克比: {dp_ana}")
print(f"   差异: {np.linalg.norm(dp_num - dp_ana):.2e}")

# 角速度导数
R_0 = T_right_0[:3, :3]
R_plus = T_right_plus[:3, :3]
R_delta = R_plus @ R_0.T

trace = np.trace(R_delta)
angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

if angle < 1e-10:
    omega_num = np.zeros(3)
else:
    axis = np.array([
        R_delta[2, 1] - R_delta[1, 2],
        R_delta[0, 2] - R_delta[2, 0],
        R_delta[1, 0] - R_delta[0, 1]
    ]) / (2 * np.sin(angle))
    omega_num = axis * angle / epsilon

omega_ana = J_bi[9:12, 7]  # 右臂角速度，关节0

print(f"\n   角速度数值微分: {omega_num}")
print(f"   角速度解析雅克比: {omega_ana}")
print(f"   差异: {np.linalg.norm(omega_num - omega_ana):.2e}")

print("\n" + "=" * 80)
