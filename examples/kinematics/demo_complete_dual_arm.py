"""完整双臂协同控制演示（含工作空间预检）

演示完整工作流：
1. 加载模型并派生双臂链
2. 预计算工作空间（一次计算，持久缓存）
3. 测试可达性检查
4. 执行双臂 IK（含自动工作空间预检）
5. 验证最终位姿精度

Run:
    python examples/kinematics/demo_complete_dual_arm.py
"""
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print
from robocore.kinematics.bimanual import dual_fk, dual_ik

def main():
    beauty_print("=== 完整双臂协同控制演示 ===", type="module", centered=True)
    
    # ==================== Step 1: 模型加载 ====================
    beauty_print("\n[Step 1] 加载 Bessica 双臂模型...")
    base = RobotModel(get_robocore_path("assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"))
    
    # 发现末端
    leaves = base.available_leaf_links()
    print(f"  检测到 {len(leaves)} 个叶节点末端")
    
    # 派生左右臂链（零拷贝，共享解析数据）
    left_end = next(l for l in leaves if 'left_arm_gripper_left_finger' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper_left_finger' in l)
    
    left = base.spawn_chain(left_end)
    right = base.spawn_chain(right_end)
    beauty_print(f"✓ 左臂: {left.num_dof()} DOF | 右臂: {right.num_dof()} DOF", type="success")
    
    # ==================== Step 2: 工作空间预计算 ====================
    beauty_print("\n[Step 2] 预计算工作空间（首次耗时，后续零开销）...")
    left.compute_workspace(num_samples=6000, method='monte_carlo', verbose=False)
    right.compute_workspace(num_samples=6000, method='monte_carlo', verbose=False)
    
    left_bounds = left.get_workspace_bounds()
    right_bounds = right.get_workspace_bounds()
    print(f"  左臂工作空间: X=[{left_bounds['x'][0]:.2f}, {left_bounds['x'][1]:.2f}]")
    print(f"  右臂工作空间: X=[{right_bounds['x'][0]:.2f}, {right_bounds['x'][1]:.2f}]")
    
    # ==================== Step 3: 可达性测试 ====================
    beauty_print("\n[Step 3] 测试典型位置可达性...")
    test_points = {
        "中心高位": np.array([0.0, 0.0, 1.2]),
        "左侧中位": np.array([-0.3, 0.3, 0.8]),
        "右侧中位": np.array([0.3, -0.3, 0.8]),
        "前方低位": np.array([0.5, 0.0, 0.5]),
    }
    
    for name, pt in test_points.items():
        left_ok = left.is_point_reachable(pt, tolerance=0.05)
        right_ok = right.is_point_reachable(pt, tolerance=0.05)
        status = "✓" if (left_ok or right_ok) else "✗"
        print(f"  {status} {name} {pt}: L={left_ok} R={right_ok}")
    
    # ==================== Step 4: 双臂 IK（小偏移目标）====================
    beauty_print("\n[Step 4] 执行双臂 IK（从当前位姿微调）...")
    
    # 初始配置（更保守，避免边缘位姿）
    qL_init = [0.0, 0.15, 0.0, 0.5, 0.0, 0.0, 0.0]
    qR_init = [0.0, -0.15, 0.0, -0.5, 0.0, 0.0, 0.0]
    
    # 当前位姿
    T_current = dual_fk(left, right, qL_init, qR_init)
    T_left_cur = T_current['left']
    T_right_cur = T_current['right']
    
    print(f"  当前左臂末端: {T_left_cur[0:3, 3]}")
    print(f"  当前右臂末端: {T_right_cur[0:3, 3]}")
    
    # 创建小偏移目标（1cm量级）
    T_left_goal = T_left_cur.copy()
    T_left_goal[0:3, 3] += np.array([0.01, 0.008, -0.005])
    
    T_right_goal = T_right_cur.copy()
    T_right_goal[0:3, 3] += np.array([-0.008, 0.01, 0.006])
    
    # 预检查目标可达性（使用较宽松tolerance因为工作空间采样不完全覆盖边缘）
    left_target_reachable = left.is_point_reachable(T_left_goal[0:3, 3], tolerance=0.12)
    right_target_reachable = right.is_point_reachable(T_right_goal[0:3, 3], tolerance=0.12)
    
    if not left_target_reachable:
        beauty_print("⚠️ 左臂目标不可达，跳过 IK", type="warning")
        return
    if not right_target_reachable:
        beauty_print("⚠️ 右臂目标不可达，跳过 IK", type="warning")
        return
    
    beauty_print("  ✓ 两个目标均可达，开始 IK 求解...")
    
    # 执行双臂 IK
    result = dual_ik(
        left, right,
        target_left=T_left_goal,
        target_right=T_right_goal,
        q0_left=qL_init,
        q0_right=qR_init,
        max_iters=120,
        pos_tol=2e-3,
        ori_tol=2e-3,
        method='dls',
        check_workspace=False,  # 已在外部预检，避免tolerance冲突
    )
    
    # ==================== Step 5: 结果验证 ====================
    beauty_print("\n[Step 5] IK 结果:")
    if result['success']:
        beauty_print(f"  ✓ 求解成功！迭代次数: {result['iters']}", type="success")
        print(f"  左臂误差: pos={result['pos_err_left']*1000:.2f}mm, ori={result['ori_err_left']*1000:.2f}mrad")
        print(f"  右臂误差: pos={result['pos_err_right']*1000:.2f}mm, ori={result['ori_err_right']*1000:.2f}mrad")
        
        # 验证最终位姿
        T_final = dual_fk(left, right, result['q_left'], result['q_right'])
        pos_diff_L = np.linalg.norm(T_final['left'][0:3, 3] - T_left_goal[0:3, 3])
        pos_diff_R = np.linalg.norm(T_final['right'][0:3, 3] - T_right_goal[0:3, 3])
        
        print(f"\n  最终位置误差验证:")
        print(f"    左臂: {pos_diff_L*1000:.3f} mm")
        print(f"    右臂: {pos_diff_R*1000:.3f} mm")
        
    else:
        print(f"  ✗ 求解失败: {result.get('error', 'convergence_failed')}")
        print(f"  迭代次数: {result['iters']}")
        print(f"  残差: L_pos={result['pos_err_left']:.2e}, R_pos={result['pos_err_right']:.2e}")
    
    # ==================== 总结 ====================
    beauty_print("\n=== 演示完成 ===", type="module", centered=True)
    print("关键要点：")
    print("  1. 工作空间只需计算一次，后续查询< 1ms")
    print("  2. 预检查避免无效 IK 迭代")
    print("  3. 双臂独立求解保证稳定性")
    print("  4. spawn_chain 共享解析，零内存开销")
    print("\n下一步：实现相对位姿约束（Phase 2）")

if __name__ == '__main__':
    main()
