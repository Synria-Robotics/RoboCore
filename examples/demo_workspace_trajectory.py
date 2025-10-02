#!/usr/bin/env python3
"""
Workspace-Constrained Trajectory Planning Demo
===============================================

Demonstrates trajectory planning within workspace constraints:
- Analyze reachable workspace
- Plan trajectories only within reachable regions
- Validate trajectory points against workspace
- Workspace-aware Cartesian path planning

Usage:
    python demo_workspace_trajectory.py --robot bessica --arm left
    python demo_workspace_trajectory.py --robot bessica --arm left --samples 10000 --visualize
"""

import numpy as np
import argparse
from pathlib import Path
import time
import warnings

from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.planning.trajectory import (
    quintic_polynomial_trajectory,
    linear_cartesian_trajectory,
    circular_cartesian_trajectory,
)


def analyze_workspace_for_planning(model, num_samples=5000, safe_threshold=0.01):
    """
    分析工作空间，为轨迹规划准备约束条件。
    
    Returns
    -------
    workspace_data : dict
        包含可达点、边界、安全区域等信息
    """
    print("\n" + "="*70)
    print("步骤 1: 工作空间分析")
    print("="*70)
    
    analyzer = WorkspaceAnalyzer(model, backend='numpy')
    
    # 计算可达工作空间
    print(f"\n正在计算可达工作空间 (样本数: {num_samples})...")
    start = time.time()
    reachable_points = analyzer.compute_reachable_workspace(
        num_samples=num_samples,
        method='monte_carlo',
        seed=42
    )
    elapsed = time.time() - start
    print(f"✓ 完成 ({elapsed:.2f}s)")
    print(f"  可达点数: {len(reachable_points)}")
    
    # 获取边界
    bounds = analyzer.get_workspace_bounds(reachable_points)
    print(f"\n工作空间边界:")
    print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
    print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
    print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
    
    # 计算体积
    volume = analyzer.estimate_workspace_volume(reachable_points, method='convex_hull')
    print(f"\n工作空间体积: {volume:.4f} m³")
    
    # 找出安全区域（无奇异点）
    print(f"\n正在分析安全区域（无奇异点）...")
    start = time.time()
    safe_points = analyzer.find_singularity_free_regions(
        num_samples=num_samples // 2,
        manipulability_threshold=safe_threshold
    )
    elapsed = time.time() - start
    
    if len(safe_points) > 0:
        print(f"✓ 完成 ({elapsed:.2f}s)")
        print(f"  安全点数: {len(safe_points)}")
        print(f"  安全比率: {len(safe_points)/(num_samples//2):.1%}")
        
        safe_bounds = analyzer.get_workspace_bounds(safe_points)
        print(f"\n安全工作空间边界:")
        print(f"  X: [{safe_bounds['x'][0]:.3f}, {safe_bounds['x'][1]:.3f}] m")
        print(f"  Y: [{safe_bounds['y'][0]:.3f}, {safe_bounds['y'][1]:.3f}] m")
        print(f"  Z: [{safe_bounds['z'][0]:.3f}, {safe_bounds['z'][1]:.3f}] m")
    else:
        print("⚠️  未找到安全区域，降低阈值或增加样本")
        safe_bounds = None
    
    # 计算密度（用于找最佳操作区）
    density, (x, y, z) = analyzer.compute_workspace_density(
        reachable_points, 
        grid_resolution=15
    )
    
    # 找最密集区域（最佳操作区）
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    best_region = np.array([x[max_idx[0]], y[max_idx[1]], z[max_idx[2]]])
    
    print(f"\n最佳操作区域（最高密度）:")
    print(f"  位置: ({best_region[0]:.3f}, {best_region[1]:.3f}, {best_region[2]:.3f})")
    print(f"  密度: {density[max_idx]:.1f} points/voxel")
    
    workspace_data = {
        'analyzer': analyzer,
        'reachable_points': reachable_points,
        'bounds': bounds,
        'safe_points': safe_points,
        'safe_bounds': safe_bounds,
        'best_region': best_region,
        'density': density,
        'volume': volume
    }
    
    return workspace_data


def validate_trajectory_workspace(trajectory_points, workspace_data, tolerance=0.05):
    """
    验证轨迹点是否在工作空间内。
    
    Parameters
    ----------
    trajectory_points : np.ndarray
        轨迹点 (N, 3)
    workspace_data : dict
        工作空间数据
    tolerance : float
        容差 (米)
    
    Returns
    -------
    validation : dict
        验证结果
    """
    analyzer = workspace_data['analyzer']
    reachable_points = workspace_data['reachable_points']
    
    valid_count = 0
    invalid_indices = []
    
    for i, point in enumerate(trajectory_points):
        if analyzer.check_point_in_workspace(point, reachable_points, tolerance):
            valid_count += 1
        else:
            invalid_indices.append(i)
    
    validation = {
        'total': len(trajectory_points),
        'valid': valid_count,
        'invalid': len(invalid_indices),
        'success_rate': valid_count / len(trajectory_points),
        'invalid_indices': invalid_indices
    }
    
    return validation


def plan_trajectory_in_workspace(
    model,
    workspace_data,
    q_start=None,
    target_region='best',
    num_waypoints=4,
    use_safe_region=True
):
    """
    在工作空间约束内规划轨迹。
    
    Parameters
    ----------
    model : RobotModel
        机器人模型
    workspace_data : dict
        工作空间数据
    q_start : np.ndarray
        起始关节配置
    target_region : str
        目标区域: 'best', 'safe', 'random'
    num_waypoints : int
        路径点数量
    use_safe_region : bool
        是否使用安全区域
    
    Returns
    -------
    trajectory : dict
        轨迹数据
    """
    print("\n" + "="*70)
    print("步骤 2: 工作空间约束轨迹规划")
    print("="*70)
    
    if q_start is None:
        q_start = np.zeros(model.dof())
    
    # 选择目标区域
    if use_safe_region and workspace_data['safe_points'] is not None:
        bounds = workspace_data['safe_bounds']
        region_name = "安全区域"
        print(f"\n使用区域: {region_name} (无奇异点)")
    else:
        bounds = workspace_data['bounds']
        region_name = "可达区域"
        print(f"\n使用区域: {region_name}")
    
    print(f"边界:")
    print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
    print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
    print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
    
    # 生成目标路径点（在工作空间范围内）
    print(f"\n生成 {num_waypoints} 个目标路径点...")
    
    waypoints_cartesian = []
    waypoints_joint = []
    
    # 起始点
    T_start = forward_kinematics(model, q_start, backend='numpy', return_end=True)
    p_start = T_start[:3, 3]
    waypoints_cartesian.append(p_start)
    waypoints_joint.append(q_start)
    
    print(f"\n路径点 0 (起点): {p_start}")
    
    # 如果指定最佳区域，第一个目标点选择最佳区域附近
    if target_region == 'best':
        best_region = workspace_data['best_region']
        # 在最佳区域附近随机采样
        for i in range(1, num_waypoints):
            offset = np.random.uniform(-0.1, 0.1, 3)
            target = best_region + offset
            
            # 确保在边界内
            target[0] = np.clip(target[0], bounds['x'][0], bounds['x'][1])
            target[1] = np.clip(target[1], bounds['y'][0], bounds['y'][1])
            target[2] = np.clip(target[2], bounds['z'][0], bounds['z'][1])
            
            waypoints_cartesian.append(target)
            print(f"路径点 {i}: {target} (最佳区域附近)")
    
    else:
        # 在边界内随机采样
        for i in range(1, num_waypoints):
            target = np.array([
                np.random.uniform(bounds['x'][0], bounds['x'][1]),
                np.random.uniform(bounds['y'][0], bounds['y'][1]),
                np.random.uniform(bounds['z'][0], bounds['z'][1])
            ])
            waypoints_cartesian.append(target)
            print(f"路径点 {i}: {target} (随机)")
    
    # 为每个路径点求IK
    print(f"\n正在求解逆运动学...")
    q_current = q_start.copy()
    ik_success_count = 0
    
    for i, target_pos in enumerate(waypoints_cartesian[1:], 1):
        # 构造目标位姿（保持当前方向）
        T_current = forward_kinematics(model, q_current, backend='numpy', return_end=True)
        T_target = T_current.copy()
        T_target[:3, 3] = target_pos
        
        # IK求解
        result = inverse_kinematics(
            model, T_target, q_current,
            backend='numpy', method='dls',
            max_iters=200, pos_tol=1e-4, ori_tol=1e-4
        )
        
        if result['success']:
            waypoints_joint.append(result['q'])
            q_current = result['q']
            ik_success_count += 1
            print(f"  路径点 {i}: ✓ IK成功")
        else:
            print(f"  路径点 {i}: ✗ IK失败，使用最接近解")
            waypoints_joint.append(result['q'])
            q_current = result['q']
    
    print(f"\nIK成功率: {ik_success_count}/{num_waypoints-1} = {ik_success_count/(num_waypoints-1):.1%}")
    
    # 生成关节空间轨迹
    print(f"\n生成平滑轨迹（五次多项式）...")
    waypoints_joint = np.array(waypoints_joint)
    
    from robocore.planning.trajectory import multi_waypoint_trajectory
    
    t_traj, q_traj, qd_traj, qdd_traj = multi_waypoint_trajectory(
        waypoints_joint,
        durations=1.0,  # 每段1秒
        num_points_per_segment=50,
        method='quintic'
    )
    
    print(f"✓ 轨迹生成完成")
    print(f"  总时长: {t_traj[-1]:.2f}s")
    print(f"  总点数: {len(t_traj)}")
    print(f"  最大速度: {np.max(np.abs(qd_traj)):.3f} rad/s")
    print(f"  最大加速度: {np.max(np.abs(qdd_traj)):.3f} rad/s²")
    
    # 验证轨迹（通过FK计算笛卡尔位置）
    print(f"\n验证轨迹在工作空间内...")
    trajectory_positions = []
    for q in q_traj:
        T = forward_kinematics(model, q, backend='numpy', return_end=True)
        trajectory_positions.append(T[:3, 3])
    
    trajectory_positions = np.array(trajectory_positions)
    
    validation = validate_trajectory_workspace(
        trajectory_positions,
        workspace_data,
        tolerance=0.05
    )
    
    print(f"\n轨迹验证结果:")
    print(f"  总点数: {validation['total']}")
    print(f"  有效点: {validation['valid']}")
    print(f"  无效点: {validation['invalid']}")
    print(f"  成功率: {validation['success_rate']:.1%}")
    
    if validation['invalid'] > 0:
        print(f"  ⚠️  {validation['invalid']} 个点可能超出工作空间")
    else:
        print(f"  ✓ 所有点都在工作空间内！")
    
    trajectory = {
        'time': t_traj,
        'q': q_traj,
        'qd': qd_traj,
        'qdd': qdd_traj,
        'cartesian_positions': trajectory_positions,
        'waypoints_cartesian': np.array(waypoints_cartesian),
        'waypoints_joint': waypoints_joint,
        'validation': validation,
        'region': region_name
    }
    
    return trajectory


def visualize_workspace_trajectory(workspace_data, trajectory, show=True):
    """
    可视化工作空间和轨迹。
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  需要 matplotlib 进行可视化")
        return
    
    print("\n" + "="*70)
    print("步骤 3: 可视化")
    print("="*70)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1: 工作空间 + 轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 绘制可达工作空间（采样）
    reachable = workspace_data['reachable_points']
    sample_indices = np.random.choice(len(reachable), min(2000, len(reachable)), replace=False)
    ax1.scatter(reachable[sample_indices, 0], 
               reachable[sample_indices, 1], 
               reachable[sample_indices, 2],
               c='lightblue', marker='.', alpha=0.1, s=1, label='可达工作空间')
    
    # 绘制安全区域
    if workspace_data['safe_points'] is not None and len(workspace_data['safe_points']) > 0:
        safe = workspace_data['safe_points']
        sample_indices_safe = np.random.choice(len(safe), min(500, len(safe)), replace=False)
        ax1.scatter(safe[sample_indices_safe, 0],
                   safe[sample_indices_safe, 1],
                   safe[sample_indices_safe, 2],
                   c='lightgreen', marker='.', alpha=0.3, s=2, label='安全区域')
    
    # 绘制轨迹
    traj_pos = trajectory['cartesian_positions']
    ax1.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
            'r-', linewidth=2, label='规划轨迹')
    
    # 绘制路径点
    waypoints = trajectory['waypoints_cartesian']
    ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               c='red', s=100, marker='o', edgecolors='black', linewidths=2, 
               label='路径点', zorder=10)
    
    # 标注起点和终点
    ax1.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2],
               c='green', s=200, marker='s', edgecolors='black', linewidths=2,
               label='起点', zorder=11)
    ax1.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2],
               c='orange', s=200, marker='^', edgecolors='black', linewidths=2,
               label='终点', zorder=11)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('工作空间与轨迹')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 关节轨迹
    ax2 = fig.add_subplot(132)
    t = trajectory['time']
    q = trajectory['q']
    
    for i in range(q.shape[1]):
        ax2.plot(t, q[:, i], label=f'关节 {i}', alpha=0.7)
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('关节角度 (rad)')
    ax2.set_title('关节空间轨迹')
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 速度和加速度
    ax3 = fig.add_subplot(133)
    
    qd_norm = np.linalg.norm(trajectory['qd'], axis=1)
    qdd_norm = np.linalg.norm(trajectory['qdd'], axis=1)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(t, qd_norm, 'b-', linewidth=2, label='速度')
    line2 = ax3_twin.plot(t, qdd_norm, 'r--', linewidth=2, label='加速度')
    
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度 (rad/s)', color='b')
    ax3_twin.set_ylabel('加速度 (rad/s²)', color='r')
    ax3.set_title('速度与加速度')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    
    if show:
        print("\n显示可视化窗口...")
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Workspace-Constrained Trajectory Planning Demo'
    )
    parser.add_argument('--robot', type=str, default='bessica',
                       choices=['alicia', 'bessica'],
                       help='机器人型号')
    parser.add_argument('--arm', type=str, default='left',
                       choices=['left', 'right'],
                       help='手臂选择（bessica）')
    parser.add_argument('--samples', type=int, default=5000,
                       help='工作空间采样数')
    parser.add_argument('--waypoints', type=int, default=4,
                       help='轨迹路径点数')
    parser.add_argument('--visualize', action='store_true',
                       help='显示可视化')
    parser.add_argument('--no-safe', action='store_true',
                       help='不使用安全区域约束')
    parser.add_argument('--target', type=str, default='best',
                       choices=['best', 'random'],
                       help='目标区域选择')
    
    args = parser.parse_args()
    
    print("="*70)
    print("工作空间约束轨迹规划演示")
    print("="*70)
    
    # 加载机器人
    print(f"\n加载机器人模型...")
    if args.robot == 'alicia':
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Alicia-D_v5_4' / 'alicia_duo_with_gripper.urdf'
        dof = 6
    else:
        urdf_path = Path(__file__).parent.parent / 'robocore' / 'assets' / 'robot' / 'urdf' / 'Bessica-D_v1_0' / 'BessicaDCodver.urdf'
        dof = 7
    
    model = RobotModel(str(urdf_path))
    print(f"✓ 已加载 {args.robot} ({dof}-DOF)")
    
    # 步骤1: 分析工作空间
    workspace_data = analyze_workspace_for_planning(
        model,
        num_samples=args.samples,
        safe_threshold=0.01
    )
    
    # 步骤2: 在工作空间内规划轨迹
    q_start = np.zeros(dof)
    
    trajectory = plan_trajectory_in_workspace(
        model,
        workspace_data,
        q_start=q_start,
        target_region=args.target,
        num_waypoints=args.waypoints,
        use_safe_region=not args.no_safe
    )
    
    # 步骤3: 可视化
    if args.visualize:
        visualize_workspace_trajectory(workspace_data, trajectory)
    
    # 总结
    print("\n" + "="*70)
    print("✓ 工作空间约束轨迹规划完成！")
    print("="*70)
    
    print("\n轨迹摘要:")
    print(f"  区域: {trajectory['region']}")
    print(f"  路径点: {len(trajectory['waypoints_cartesian'])}")
    print(f"  总时长: {trajectory['time'][-1]:.2f}s")
    print(f"  总点数: {len(trajectory['q'])}")
    print(f"  工作空间验证: {trajectory['validation']['success_rate']:.1%} 通过")
    
    print("\n建议:")
    print("  • 使用 --visualize 查看3D可视化")
    print("  • 增加 --samples 提高工作空间精度")
    print("  • 使用 --target best 规划到最佳操作区域")
    print("  • 使用 --no-safe 允许在奇异点附近规划")


if __name__ == '__main__':
    main()
