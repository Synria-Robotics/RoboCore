"""Multi-Chain FK Concept Demo

解释多链FK如何处理固定关节和可动关节。

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np

print("=" * 80)
print("Multi-Chain FK 概念解释")
print("=" * 80)
print()

print("问题：机器人有9个link，但只给了6个关节值，如何计算Link7和Link8的pose？")
print()

print("答案：")
print("=" * 80)
print()

print("1. 关节类型分类：")
print("   - 可动关节（actuated joints）：需要关节值")
print("     * Joint1-6: 6个旋转关节（revolute）")
print("     * left_finger, right_finger: 2个滑动关节（prismatic）")
print("     → 总共8个可动关节 = 8 DOF")
print()
print("   - 固定关节（fixed joints）：不需要关节值，变换是静态的")
print("     * 例如：Grasp_base可能通过fixed joint连接到Link6")
print()

print("2. 多链FK的工作原理：")
print("=" * 80)
print()

print("对于任意link的变换计算：")
print("   T_link = T_parent @ T_origin @ T_motion")
print()
print("   其中：")
print("   - T_parent: 父link的变换（递归计算或从缓存读取）")
print("   - T_origin: 关节的静态偏移（origin_xyz, origin_rpy）")
print("   - T_motion: 关节的运动变换")
print()

print("对于不同类型的关节：")
print()
print("   a) 旋转关节（revolute）:")
print("      T_motion = rotation_matrix(axis, q[joint_name])")
print()
print("   b) 滑动关节（prismatic）:")
print("      T_motion = translation_matrix(axis * q[joint_name])")
print()
print("   c) 固定关节（fixed）:")
print("      T_motion = Identity  （单位矩阵，无运动）")
print("      → T_link = T_parent @ T_origin")
print()

print("3. 实际例子：")
print("=" * 80)
print()

print("假设机器人结构：")
print("   base_link")
print("   └── Joint1 (revolute, q[0]=0.1)")
print("       └── Link1")
print("           └── Joint2 (revolute, q[1]=0.2)")
print("               └── Link2")
print("                   └── ... (Joint3-6)")
print("                       └── Link6")
print("                           ├── left_finger (prismatic, q[6]=0.01)")
print("                           │   └── Link7")
print("                           └── right_finger (prismatic, q[7]=-0.01)")
print("                               └── Link8")
print()

print("计算Link7的pose：")
print("   1. 获取父link（Link6）的变换: T_Link6")
print("   2. 应用left_finger关节的origin偏移: T_origin")
print("   3. 应用滑动运动: T_motion = translate(axis * q[6])")
print("   4. 最终: T_Link7 = T_Link6 @ T_origin @ T_motion")
print()

print("4. 多链FK的优势：")
print("=" * 80)
print("   - 一次性计算所有link的pose")
print("   - 变换复用：父link的变换可以被多个子link重用")
print("   - 支持树形结构（不仅仅是串行链）")
print("   - 对于WDF等应用很重要，需要知道所有link的位置")
print()

print("5. 为什么之前显示6个DOF？")
print("=" * 80)
print("   - 如果指定 end_link='Link6'，RobotModel只会加载到Link6")
print("   - 这是串行链模式（single-chain mode）")
print("   - Link7和Link8不在base_link→Link6的路径上")
print()
print("   解决方案：")
print("   - 方案1: 不指定end_link，加载完整树结构 → 8 DOF")
print("   - 方案2: 指定end_link='Link8' → 包含夹爪 → 8 DOF")
print("   - 方案3: 使用return_all_links=True → 计算所有link的FK")
print()

print("=" * 80)
print("总结：")
print("=" * 80)
print("多链FK通过深度优先遍历整个运动树，为每个link计算变换。")
print("固定关节的link通过T_origin获得相对于父link的静态偏移。")
print("可动关节的link额外应用T_motion（基于关节值q）。")
print("=" * 80)

