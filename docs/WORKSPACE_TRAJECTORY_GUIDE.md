# Workspace-Constrained Trajectory Planning Guide

## 概述

工作空间约束轨迹规划将**工作空间分析**与**轨迹规划**相结合，确保生成的轨迹在机器人的可达范围内，并避开奇异点区域。这对于实际应用至关重要，因为它可以：

- ✅ 确保目标点可达
- ✅ 避免奇异点配置
- ✅ 在最佳操作区域内规划
- ✅ 提高轨迹可靠性

## 快速开始

### 基本用法

```python
from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
from robocore.planning.trajectory import multi_waypoint_trajectory
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics

# 1. 加载机器人模型
model = RobotModel('path/to/robot.urdf')

# 2. 分析工作空间
analyzer = WorkspaceAnalyzer(model)
reachable_points = analyzer.compute_reachable_workspace(num_samples=5000)
bounds = analyzer.get_workspace_bounds(reachable_points)

# 3. 在工作空间范围内选择目标点
target = np.array([
    np.random.uniform(bounds['x'][0], bounds['x'][1]),
    np.random.uniform(bounds['y'][0], bounds['y'][1]),
    np.random.uniform(bounds['z'][0], bounds['z'][1])
])

# 4. 验证目标点可达性
if analyzer.check_point_in_workspace(target, reachable_points):
    print("✓ 目标点可达")
else:
    print("✗ 目标点不可达")

# 5. 求解IK并生成轨迹
q_start = np.zeros(model.dof())
T_target = np.eye(4)
T_target[:3, 3] = target

result = inverse_kinematics(model, T_target, q_start, backend='numpy')

if result['success']:
    # 生成轨迹
    waypoints = np.array([q_start, result['q']])
    t, q, qd, qdd = multi_waypoint_trajectory(waypoints, durations=2.0)
    print(f"✓ 轨迹生成成功，时长 {t[-1]:.2f}s")
```

### 使用安全区域（避开奇异点）

```python
# 找出无奇异点的安全区域
safe_points = analyzer.find_singularity_free_regions(
    num_samples=5000,
    manipulability_threshold=0.01
)

if len(safe_points) > 0:
    safe_bounds = analyzer.get_workspace_bounds(safe_points)
    
    # 在安全区域内选择目标点
    target_safe = np.array([
        np.random.uniform(safe_bounds['x'][0], safe_bounds['x'][1]),
        np.random.uniform(safe_bounds['y'][0], safe_bounds['y'][1]),
        np.random.uniform(safe_bounds['z'][0], safe_bounds['z'][1])
    ])
    
    print(f"安全目标点: {target_safe}")
```

### 找到最佳操作区域

```python
# 计算密度分布
density, (x, y, z) = analyzer.compute_workspace_density(
    reachable_points,
    grid_resolution=15
)

# 找到密度最高的区域（最佳操作区）
max_idx = np.unravel_index(np.argmax(density), density.shape)
best_region = np.array([x[max_idx[0]], y[max_idx[1]], z[max_idx[2]]])

print(f"最佳操作区域: {best_region}")
print(f"密度: {density[max_idx]} points/voxel")

# 在最佳区域附近规划任务
offset = np.array([0.05, 0.02, -0.03])  # 小偏移
target_optimal = best_region + offset
```

## 完整工作流程

### 步骤 1: 工作空间分析

```python
def analyze_workspace_for_planning(model, num_samples=5000):
    """分析工作空间，为轨迹规划准备约束条件"""
    
    analyzer = WorkspaceAnalyzer(model)
    
    # 可达工作空间
    reachable_points = analyzer.compute_reachable_workspace(
        num_samples=num_samples,
        method='monte_carlo'
    )
    
    # 边界
    bounds = analyzer.get_workspace_bounds(reachable_points)
    
    # 安全区域
    safe_points = analyzer.find_singularity_free_regions(
        num_samples=num_samples // 2,
        manipulability_threshold=0.01
    )
    
    # 最佳区域
    density, (x, y, z) = analyzer.compute_workspace_density(
        reachable_points,
        grid_resolution=15
    )
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    best_region = np.array([x[max_idx[0]], y[max_idx[1]], z[max_idx[2]]])
    
    return {
        'analyzer': analyzer,
        'reachable_points': reachable_points,
        'bounds': bounds,
        'safe_points': safe_points,
        'best_region': best_region
    }
```

### 步骤 2: 生成工作空间约束路径点

```python
def generate_waypoints_in_workspace(workspace_data, num_waypoints=5, use_safe=True):
    """在工作空间范围内生成路径点"""
    
    if use_safe and workspace_data['safe_points'] is not None:
        bounds = analyzer.get_workspace_bounds(workspace_data['safe_points'])
    else:
        bounds = workspace_data['bounds']
    
    waypoints = []
    
    for i in range(num_waypoints):
        point = np.array([
            np.random.uniform(bounds['x'][0], bounds['x'][1]),
            np.random.uniform(bounds['y'][0], bounds['y'][1]),
            np.random.uniform(bounds['z'][0], bounds['z'][1])
        ])
        waypoints.append(point)
    
    return np.array(waypoints)
```

### 步骤 3: 验证轨迹

```python
def validate_trajectory(trajectory_positions, workspace_data, tolerance=0.05):
    """验证轨迹点是否在工作空间内"""
    
    analyzer = workspace_data['analyzer']
    reachable_points = workspace_data['reachable_points']
    
    valid_count = 0
    for point in trajectory_positions:
        if analyzer.check_point_in_workspace(point, reachable_points, tolerance):
            valid_count += 1
    
    success_rate = valid_count / len(trajectory_positions)
    
    return {
        'total': len(trajectory_positions),
        'valid': valid_count,
        'success_rate': success_rate
    }
```

### 步骤 4: 完整示例

```python
# 1. 分析工作空间
workspace_data = analyze_workspace_for_planning(model, num_samples=5000)

print("工作空间边界:")
print(f"  X: {workspace_data['bounds']['x']}")
print(f"  Y: {workspace_data['bounds']['y']}")
print(f"  Z: {workspace_data['bounds']['z']}")

# 2. 生成路径点（在安全区域内）
waypoints_cartesian = generate_waypoints_in_workspace(
    workspace_data,
    num_waypoints=4,
    use_safe=True
)

# 3. 为每个路径点求IK
q_start = np.zeros(model.dof())
waypoints_joint = [q_start]

q_current = q_start.copy()
for target_pos in waypoints_cartesian:
    # 构造目标位姿
    T_target = np.eye(4)
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
        print(f"✓ IK成功: {target_pos}")
    else:
        print(f"✗ IK失败: {target_pos}")
        # 使用最接近的解
        waypoints_joint.append(result['q'])
        q_current = result['q']

# 4. 生成平滑轨迹
waypoints_joint = np.array(waypoints_joint)
t, q, qd, qdd = multi_waypoint_trajectory(
    waypoints_joint,
    durations=1.0,  # 每段1秒
    num_points_per_segment=50
)

# 5. 验证轨迹
trajectory_positions = []
for q_val in q:
    T = forward_kinematics(model, q_val, backend='numpy', return_end=True)
    trajectory_positions.append(T[:3, 3])

validation = validate_trajectory(
    np.array(trajectory_positions),
    workspace_data
)

print(f"\n轨迹验证:")
print(f"  成功率: {validation['success_rate']:.1%}")
print(f"  有效点: {validation['valid']}/{validation['total']}")
```

## 演示脚本

使用提供的演示脚本进行工作空间约束轨迹规划：

```bash
# 基本用法
python examples/demo_workspace_trajectory.py --robot bessica --arm left

# 指定采样数和路径点数
python examples/demo_workspace_trajectory.py --robot bessica --arm left --samples 10000 --waypoints 5

# 带可视化
python examples/demo_workspace_trajectory.py --robot bessica --arm left --visualize

# 不使用安全区域约束
python examples/demo_workspace_trajectory.py --robot bessica --arm left --no-safe

# 使用随机目标区域（而非最佳区域）
python examples/demo_workspace_trajectory.py --robot bessica --arm left --target random
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | `bessica` | 机器人型号 (alicia/bessica) |
| `--arm` | str | `left` | 手臂选择 (left/right) |
| `--samples` | int | `5000` | 工作空间采样数 |
| `--waypoints` | int | `4` | 路径点数量 |
| `--visualize` | flag | - | 显示3D可视化 |
| `--no-safe` | flag | - | 不使用安全区域约束 |
| `--target` | str | `best` | 目标区域 (best/random) |

## 应用场景

### 1. 拾取与放置任务

```python
# 分析工作空间
workspace_data = analyze_workspace_for_planning(model)

# 定义拾取和放置位置
pick_position = np.array([0.3, 0.2, 0.5])
place_position = np.array([0.4, -0.1, 0.6])

# 验证位置可达性
analyzer = workspace_data['analyzer']
reachable = workspace_data['reachable_points']

pick_reachable = analyzer.check_point_in_workspace(pick_position, reachable)
place_reachable = analyzer.check_point_in_workspace(place_position, reachable)

if pick_reachable and place_reachable:
    print("✓ 拾取和放置位置都可达")
    # 规划轨迹...
else:
    print("✗ 有位置不可达，需要调整")
```

### 2. 避开奇异点的装配任务

```python
# 找出安全区域
safe_points = analyzer.find_singularity_free_regions(
    num_samples=10000,
    manipulability_threshold=0.02  # 更严格的阈值
)

# 在安全区域内规划装配路径
safe_bounds = analyzer.get_workspace_bounds(safe_points)

# 装配点应该在安全区域内
assembly_point = np.array([
    (safe_bounds['x'][0] + safe_bounds['x'][1]) / 2,
    (safe_bounds['y'][0] + safe_bounds['y'][1]) / 2,
    (safe_bounds['z'][0] + safe_bounds['z'][1]) / 2
])
```

### 3. 工作站布局优化

```python
# 计算密度分布
density, (x, y, z) = analyzer.compute_workspace_density(
    reachable_points,
    grid_resolution=20
)

# 找到最佳操作区域
max_idx = np.unravel_index(np.argmax(density), density.shape)
optimal_workspace_center = np.array([
    x[max_idx[0]], 
    y[max_idx[1]], 
    z[max_idx[2]]
])

print(f"建议将工作台放置在: {optimal_workspace_center}")
print(f"该区域密度最高，操作最灵活")
```

## 性能优化

### 采样数量选择

```python
# 快速原型（1-2秒）
workspace_data = analyze_workspace_for_planning(model, num_samples=1000)

# 标准精度（3-5秒）
workspace_data = analyze_workspace_for_planning(model, num_samples=5000)

# 高精度（10-20秒）
workspace_data = analyze_workspace_for_planning(model, num_samples=20000)
```

### 使用并行计算

```python
# 并行FK计算
reachable_points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='monte_carlo',
    use_parallel=True  # 启用并行
)
```

### GPU加速（需要PyTorch）

```python
# 使用GPU进行批量FK计算
from robocore.kinematics.fk import batch_fk_torch
import torch

# 生成大量关节配置
q_samples = torch.randn(10000, model.dof(), device='cuda')

# 批量FK（GPU加速）
transforms = batch_fk_torch(model, q_samples, return_end=True)
positions = transforms[:, :3, 3].cpu().numpy()
```

## 最佳实践

### 1. 分层规划策略

```python
# 粗规划：使用较少样本快速验证可行性
coarse_data = analyze_workspace_for_planning(model, num_samples=1000)
if can_reach_target(target, coarse_data):
    # 细规划：使用更多样本进行精确规划
    fine_data = analyze_workspace_for_planning(model, num_samples=10000)
    trajectory = plan_trajectory(target, fine_data)
```

### 2. 缓存工作空间数据

```python
import pickle

# 保存工作空间数据
with open('workspace_cache.pkl', 'wb') as f:
    pickle.dump(workspace_data, f)

# 加载缓存数据
with open('workspace_cache.pkl', 'rb') as f:
    workspace_data = pickle.load(f)
```

### 3. 自适应采样

```python
def adaptive_workspace_analysis(model, target_point, initial_samples=1000):
    """自适应采样：根据目标点调整采样密度"""
    
    # 初始粗采样
    data = analyze_workspace_for_planning(model, num_samples=initial_samples)
    
    # 如果目标点可达性不确定，增加局部采样
    if not analyzer.check_point_in_workspace(target_point, data['reachable_points']):
        # 在目标点附近增加采样
        print("增加局部采样...")
        # ... 实现局部密集采样
    
    return data
```

### 4. IK失败处理

```python
def robust_ik_solve(model, target_pose, q_init, max_attempts=5):
    """鲁棒IK求解：多次尝试不同初始值"""
    
    for attempt in range(max_attempts):
        # 添加随机扰动
        q_start = q_init + np.random.randn(len(q_init)) * 0.1
        
        result = inverse_kinematics(
            model, target_pose, q_start,
            backend='numpy', method='dls',
            max_iters=200, pos_tol=1e-4, ori_tol=1e-4
        )
        
        if result['success']:
            return result
    
    # 所有尝试失败，返回最后一次结果
    return result
```

## 故障排除

### 问题1: IK求解失败率高

**原因**：目标点在工作空间边缘或需要极端配置

**解决方案**：
```python
# 使用安全区域约束
safe_points = analyzer.find_singularity_free_regions(num_samples=5000)
safe_bounds = analyzer.get_workspace_bounds(safe_points)

# 在安全区域内选择目标
# 或者使用multi-start IK
result = inverse_kinematics(
    model, target_pose, q_init,
    backend='numpy',
    multi_start=10,  # 10次随机重启
    multi_noise=0.5
)
```

### 问题2: 轨迹验证成功率低

**原因**：工作空间采样不足或轨迹经过边界

**解决方案**：
```python
# 增加采样数
workspace_data = analyze_workspace_for_planning(model, num_samples=20000)

# 或者在工作空间内部规划（留出安全边距）
bounds = workspace_data['bounds']
margin = 0.05  # 5cm 边距

safe_bounds = {
    'x': [bounds['x'][0] + margin, bounds['x'][1] - margin],
    'y': [bounds['y'][0] + margin, bounds['y'][1] - margin],
    'z': [bounds['z'][0] + margin, bounds['z'][1] - margin]
}
```

### 问题3: 计算时间过长

**原因**：采样数过多或未使用并行化

**解决方案**：
```python
# 启用并行计算
reachable_points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    use_parallel=True
)

# 或者使用更快的采样方法
reachable_points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='sobol'  # Sobol序列收敛更快
)
```

## 进阶主题

### 动态工作空间

```python
# 考虑障碍物的工作空间
def compute_collision_free_workspace(model, obstacles, num_samples=5000):
    """计算无碰撞工作空间"""
    
    analyzer = WorkspaceAnalyzer(model)
    all_points = analyzer.compute_reachable_workspace(num_samples)
    
    # 过滤掉碰撞点
    collision_free = []
    for point in all_points:
        if not check_collision(point, obstacles):
            collision_free.append(point)
    
    return np.array(collision_free)
```

### 时间最优轨迹

```python
# 在工作空间约束下规划时间最优轨迹
def time_optimal_trajectory(waypoints_joint, workspace_data):
    """生成时间最优轨迹（考虑工作空间约束）"""
    
    # 计算每段的最优时间
    durations = []
    for i in range(len(waypoints_joint) - 1):
        # 根据距离和速度限制计算时间
        distance = np.linalg.norm(waypoints_joint[i+1] - waypoints_joint[i])
        optimal_time = distance / max_velocity
        durations.append(optimal_time)
    
    t, q, qd, qdd = multi_waypoint_trajectory(
        waypoints_joint,
        durations=durations
    )
    
    return t, q, qd, qdd
```

## 总结

工作空间约束轨迹规划的关键要点：

1. **先分析，后规划**：始终先分析工作空间，了解机器人能力
2. **使用安全区域**：在无奇异点区域规划可提高可靠性
3. **验证轨迹**：生成后验证所有轨迹点是否在工作空间内
4. **最佳操作区**：在密度最高的区域规划任务
5. **鲁棒性**：使用multi-start IK和自适应采样
6. **性能优化**：根据需求选择采样数，使用并行计算

## 相关文档

- [工作空间分析指南](WORKSPACE_ANALYSIS_GUIDE.md)
- [轨迹规划指南](TRAJECTORY_PLANNING_GUIDE.md)
- [运动学API文档](../robocore/kinematics/README.md)
