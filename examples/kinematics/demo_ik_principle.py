#!/usr/bin/env python3
"""pytorch_kinematics逆运动学原理演示

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import torch
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print


def demo_ik_principle():
    """演示pytorch_kinematics逆运动学原理"""
    
    beauty_print("=== pytorch_kinematics逆运动学原理 ===", type="module", centered=True)
    
    # 加载机器人
    model_path = get_robocore_path("assets/robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml")
    robot = RobotModel(model_path)
    
    beauty_print("🤖 Alicia机器人结构:", type="info")
    robot.print_tree()
    
    beauty_print("\n🔍 pytorch_kinematics逆运动学核心原理:", type="info")
    beauty_print("1. 基于雅可比矩阵的迭代求解")
    beauty_print("2. 使用阻尼最小二乘法 (Damped Least Squares)")
    beauty_print("3. 支持批量求解和多次重试")
    beauty_print("4. 支持位置和姿态目标")
    
    beauty_print("\n📊 逆运动学算法流程:", type="info")
    beauty_print("1. 给定目标位姿 T_target")
    beauty_print("2. 初始化关节配置 q_initial")
    beauty_print("3. 迭代求解:")
    beauty_print("   a) 计算当前FK: T_current = FK(q)")
    beauty_print("   b) 计算误差: Δx = T_target - T_current")
    beauty_print("   c) 计算雅可比: J = ∂FK/∂q")
    beauty_print("   d) 求解关节增量: Δq = J⁺Δx (伪逆)")
    beauty_print("   e) 更新关节: q = q + αΔq")
    beauty_print("4. 重复直到收敛")
    
    beauty_print("\n🔧 阻尼最小二乘法 (DLS):", type="info")
    beauty_print("• 标准伪逆: Δq = J⁺Δx")
    beauty_print("• 阻尼伪逆: Δq = Jᵀ(JJᵀ + λ²I)⁻¹Δx")
    beauty_print("• λ是阻尼系数，避免奇异性问题")
    beauty_print("• 当J接近奇异时，λ²I项提供稳定性")
    
    beauty_print("\n💡 多链机器人的IK挑战:", type="info")
    beauty_print("• 对于Alicia机器人，有两个末端执行器 (Link7, Link8)")
    beauty_print("• 传统方法: 分别对每个链求解IK")
    beauty_print("• pytorch_kinematics方法: 使用SerialChain提取特定链")
    
    # 演示SerialChain提取
    beauty_print("\n🚀 SerialChain提取演示:", type="info")
    chains = robot.auto_discover_chains()
    
    for chain_name, end_link in chains.items():
        beauty_print(f"\n{chain_name} (末端: {end_link}):")
        chain = robot.extract_chain(end_link)
        beauty_print(f"  DOF: {chain.num_dof}")
        beauty_print(f"  关节: {chain.joint_list}")
        
        # 模拟IK求解过程
        beauty_print(f"  IK求解过程:")
        beauty_print(f"    1. 提取串行链: base_link → {end_link}")
        beauty_print(f"    2. 使用链的雅可比矩阵求解")
        beauty_print(f"    3. 只考虑链上的关节")
    
    beauty_print("\n🎯 关键优势:", type="success")
    beauty_print("• 批量IK求解: 同时处理多个目标")
    beauty_print("• 多次重试: 从不同初始配置开始")
    beauty_print("• 梯度支持: 端到端可微分")
    beauty_print("• 关节限制: 自动处理关节限制")
    beauty_print("• 收敛检测: 位置和姿态容差")
    
    beauty_print("\n📈 算法特点:", type="info")
    beauty_print("• 迭代次数: 通常30-50次")
    beauty_print("• 学习率: 0.1-0.3")
    beauty_print("• 阻尼系数: 1e-6 到 1e-3")
    beauty_print("• 收敛容差: 位置1mm, 姿态1度")
    
    beauty_print("\n✨ 总结:", type="success")
    beauty_print("pytorch_kinematics的IK通过以下方式处理多链机器人:")
    beauty_print("1. 使用SerialChain提取特定末端执行器链")
    beauty_print("2. 对每个链独立求解IK")
    beauty_print("3. 利用批量计算和多次重试提高成功率")
    beauty_print("4. 支持梯度计算，可用于优化问题")


def demo_ik_vs_traditional():
    """对比传统IK和pytorch_kinematics IK"""
    
    beauty_print("\n=== 传统IK vs pytorch_kinematics IK ===", type="module", centered=True)
    
    beauty_print("🔄 传统IK方法:", type="info")
    beauty_print("• 分别对每个链求解IK")
    beauty_print("• 手动管理多个IK求解器")
    beauty_print("• 难以处理链间约束")
    beauty_print("• 计算效率较低")
    
    beauty_print("\n🚀 pytorch_kinematics IK方法:", type="info")
    beauty_print("• 使用SerialChain自动提取链")
    beauty_print("• 统一的IK接口")
    beauty_print("• 批量求解支持")
    beauty_print("• 梯度计算支持")
    beauty_print("• 更好的数值稳定性")
    
    beauty_print("\n📊 性能对比:", type="info")
    beauty_print("• 批量处理: pytorch_kinematics支持同时求解多个目标")
    beauty_print("• 多次重试: 自动从不同初始配置重试")
    beauty_print("• 收敛速度: 阻尼最小二乘法更稳定")
    beauty_print("• 内存效率: 共享计算图，减少重复计算")
    
    beauty_print("\n🎯 适用场景:", type="success")
    beauty_print("• 单链机器人: 直接使用SerialChain")
    beauty_print("• 多链机器人: 分别提取每个链的SerialChain")
    beauty_print("• 双臂机器人: 两个独立的SerialChain")
    beauty_print("• 人形机器人: 多个SerialChain (手臂、腿部等)")


if __name__ == "__main__":
    demo_ik_principle()
    demo_ik_vs_traditional()
    
    beauty_print("\n=== 演示完成 ===", type="module", centered=True)
