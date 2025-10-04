#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 RoboCore 与 JS 软件的四元数输出

四元数有双重覆盖性质：q 和 -q 表示相同的旋转
"""

import numpy as np
from robocore.transform import quaternion_to_matrix


def quaternion_to_rotation_matrix(quat_xyzw):
    """Convert quaternion (xyzw) to rotation matrix using new transform API."""
    return quaternion_to_matrix(np.array(quat_xyzw))

def rotation_matrices_equal(R1, R2, tol=1e-4):
    """Check if two rotation matrices are equal within tolerance."""
    return np.allclose(R1, R2, atol=tol)

# JS 软件输出
js_position = np.array([0.25, -0.69, 0.84])
js_quat_xyzw = np.array([0.06, 0.06, -0.70, 0.70])

# RoboCore 输出 (left_arm_gripper_left_finger, q=[1.39, 0, 0, 0, 0, 0, 0])
robocore_position = np.array([0.24969, -0.68743, 0.83836])
robocore_quat_xyzw = np.array([-0.063835, -0.063833, 0.704220, -0.704219])

# 说明
print("测试配置:")
print("  机器人: Bessica-D v1.0")
print("  End Link: left_arm_gripper_left_finger")
print("  关节角度: q = [1.39, 0, 0, 0, 0, 0, 0] rad")
print("           = [79.64°, 0°, 0°, 0°, 0°, 0°, 0°]")
print()

print("="*60)
print("验证 RoboCore vs JS 软件输出")
print("="*60)

# 1. 检查位置
print("\n1. 位置对比:")
print(f"   JS:       [{js_position[0]:.2f}, {js_position[1]:.2f}, {js_position[2]:.2f}]")
print(f"   RoboCore: [{robocore_position[0]:.5f}, {robocore_position[1]:.5f}, {robocore_position[2]:.5f}]")
pos_diff = np.linalg.norm(js_position - robocore_position)
print(f"   差异: {pos_diff:.6f} m")
print(f"   ✓ 位置匹配！(差异 < 1mm)" if pos_diff < 0.001 else "   ✗ 位置不匹配")

# 2. 检查四元数 (原始)
print("\n2. 四元数对比 (xyzw 顺序):")
print(f"   JS:       [{js_quat_xyzw[0]:.2f}, {js_quat_xyzw[1]:.2f}, {js_quat_xyzw[2]:.2f}, {js_quat_xyzw[3]:.2f}]")
print(f"   RoboCore: [{robocore_quat_xyzw[0]:.6f}, {robocore_quat_xyzw[1]:.6f}, {robocore_quat_xyzw[2]:.6f}, {robocore_quat_xyzw[3]:.6f}]")

quat_diff = np.linalg.norm(js_quat_xyzw - robocore_quat_xyzw)
print(f"   四元数差异: {quat_diff:.6f}")

# 3. 检查是否是取反的四元数
print("\n3. 检查四元数双重覆盖 (q vs -q):")
robocore_quat_neg = -robocore_quat_xyzw
print(f"   -RoboCore: [{robocore_quat_neg[0]:.6f}, {robocore_quat_neg[1]:.6f}, {robocore_quat_neg[2]:.6f}, {robocore_quat_neg[3]:.6f}]")

quat_neg_diff = np.linalg.norm(js_quat_xyzw - robocore_quat_neg)
print(f"   与 -RoboCore 的差异: {quat_neg_diff:.6f}")
print(f"   ✓ 四元数匹配！(q = -q')" if quat_neg_diff < 0.01 else "   ✗ 四元数不匹配")

# 4. 转换为旋转矩阵对比
print("\n4. 旋转矩阵对比:")
R_js = quaternion_to_rotation_matrix(js_quat_xyzw)
R_robocore = quaternion_to_rotation_matrix(robocore_quat_xyzw)

print("   JS 旋转矩阵:")
for row in R_js:
    print(f"     [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

print("\n   RoboCore 旋转矩阵:")
for row in R_robocore:
    print(f"     [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

R_diff = np.linalg.norm(R_js - R_robocore, 'fro')
print(f"\n   旋转矩阵 Frobenius 范数差异: {R_diff:.6e}")
print(f"   ✓ 旋转矩阵完全匹配！" if R_diff < 1e-3 else "   ✗ 旋转矩阵不匹配")

# 5. 结论
print("\n" + "="*60)
print("结论:")
print("="*60)
if pos_diff < 0.001 and R_diff < 1e-3:
    print("✓✓✓ RoboCore 与 JS 软件输出完全一致！")
    print()
    print("说明:")
    print("  - 位置完全匹配")
    print("  - 旋转完全匹配")
    print("  - 四元数差异是由于双重覆盖性质：q 和 -q 表示相同旋转")
    print("  - 这是正常现象，不是错误！")
else:
    print("✗ 输出不匹配，需要进一步检查")
    
print("="*60)
