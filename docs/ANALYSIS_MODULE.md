# RoboCore 分析模块文档

## 目录

1. [概述](#概述)
2. [模块架构](#模块架构)
3. [功能分类](#功能分类)
4. [功能详解](#功能详解)
5. [分析指标对比](#分析指标对比)
6. [使用指南](#使用指南)
7. [实现状态](#实现状态)

---

## 概述

RoboCore分析模块提供机器人性能分析工具，支持工作空间分析和奇异性分析，帮助评估机器人的可达性、灵活性和性能特征。

### 设计原则

- **全面分析**：提供工作空间和奇异性两个维度的分析
- **高效采样**：支持多种采样方法（蒙特卡洛、网格、Sobol序列）
- **并行计算**：支持并行处理，提高大规模分析效率
- **可视化支持**：提供3D可视化功能
- **灵活配置**：支持自定义关节限制、采样参数等

---

## 模块架构

### 目录结构

```
robocore/analysis/
├── __init__.py                    # 模块导出
├── workspace_analyzer.py          # 工作空间分析器
└── singularity_analyzer.py        # 奇异性分析器
```

### 统一接口

所有分析功能通过统一的高级接口访问：

```python
from robocore.analysis import (
    WorkspaceAnalyzer,              # 工作空间分析器
    SingularityAnalyzer,             # 奇异性分析器
    analyze_workspace_comparison,    # 工作空间对比分析
)
```

---

## 功能分类

### 工作空间分析（Workspace Analysis）

**特点**：分析机器人末端执行器的可达空间

| 功能 | 状态 | 说明 |
|------|------|------|
| 可达工作空间 | ✅ 已实现 | 计算所有可达点 |
| 灵巧工作空间 | ✅ 已实现 | 计算可多姿态到达的点 |
| 工作空间边界 | ✅ 已实现 | 计算工作空间边界框 |
| 工作空间体积 | ✅ 已实现 | 估计工作空间体积 |
| 可达性检查 | ✅ 已实现 | 检查点是否在工作空间内 |
| 工作空间密度 | ✅ 已实现 | 计算工作空间密度分布 |
| 无奇异性区域 | ✅ 已实现 | 查找无奇异性工作空间区域 |
| 可视化 | ✅ 已实现 | 3D可视化工作空间 |

### 奇异性分析（Singularity Analysis）

**特点**：分析机器人配置的奇异性特征

| 功能 | 状态 | 说明 |
|------|------|------|
| 配置分析 | ✅ 已实现 | 分析单个配置的奇异性指标 |
| 工作空间采样 | ✅ 已实现 | 采样分析奇异性分布 |
| 可操作性度量 | ✅ 已实现 | 计算可操作性指标 |
| 条件数分析 | ✅ 已实现 | 计算雅可比条件数 |

---

## 功能详解

### 1. 工作空间分析器（WorkspaceAnalyzer）

#### 初始化

**函数**：`WorkspaceAnalyzer(model)`

**参数**：
- `model`: RobotModel实例

**使用示例**：

```python
from robocore.analysis import WorkspaceAnalyzer
from robocore.modeling import RobotModel

robot = RobotModel("path/to/robot.urdf")
analyzer = WorkspaceAnalyzer(robot)
```

---

#### 计算可达工作空间

**函数**：`compute_reachable_workspace(...)`

**功能**：计算机器人末端执行器可以到达的所有点

**参数**：
- `num_samples`: 采样数量（默认10000）
- `method`: 采样方法
  - `'monte_carlo'`: 蒙特卡洛随机采样（默认）
  - `'grid'`: 网格采样
  - `'sobol'`: Sobol序列（准随机）
- `q_limits`: 关节限制，形状 (dof, 2)（默认：[-π, π]）
- `use_parallel`: 是否使用并行处理（默认False）
- `num_workers`: 并行工作线程数（默认None）
- `seed`: 随机种子（默认None）

**返回**：
- `points`: 可达点数组，形状 (num_valid, 3)

**使用示例**：

```python
# 基本使用
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='monte_carlo'
)

# 使用并行处理（大规模采样）
points = analyzer.compute_reachable_workspace(
    num_samples=100000,
    method='monte_carlo',
    use_parallel=True,
    num_workers=4
)

# 使用Sobol序列（更均匀的采样）
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='sobol',
    seed=42
)

# 自定义关节限制
q_limits = np.array([
    [-np.pi, np.pi],    # 关节1
    [-np.pi/2, np.pi/2], # 关节2
    # ...
])
points = analyzer.compute_reachable_workspace(
    num_samples=10000,
    q_limits=q_limits
)
```

**适用场景**：
- 机器人设计评估
- 工作空间可视化
- 可达性分析
- 任务规划

---

#### 计算灵巧工作空间

**函数**：`compute_dexterous_workspace(...)`

**功能**：计算可以以多种姿态到达的点（灵巧工作空间）

**参数**：
- `num_samples`: 位置采样数量（默认10000）
- `num_orientations`: 所需的最小姿态数量（默认8）
- `q_limits`: 关节限制（可选）
- `tolerance`: 位置容差（米，默认0.01）
- `use_parallel`: 是否使用并行处理（默认False）

**返回**：
- `points`: 灵巧工作空间点数组

**使用示例**：

```python
# 计算至少可以8种不同姿态到达的点
dex_points = analyzer.compute_dexterous_workspace(
    num_samples=20000,
    num_orientations=8,
    tolerance=0.01
)
```

**适用场景**：
- 需要灵活操作的任务
- 抓取规划
- 装配任务

---

#### 获取工作空间边界

**函数**：`get_workspace_bounds(points=None)`

**功能**：计算工作空间的边界框

**参数**：
- `points`: 工作空间点数组（可选，默认使用缓存）

**返回**：
- `bounds`: 字典，包含 'x', 'y', 'z' 键，每个值为 (min, max) 元组

**使用示例**：

```python
bounds = analyzer.get_workspace_bounds()
print(f"X范围: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
print(f"Y范围: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
print(f"Z范围: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
```

---

#### 估计工作空间体积

**函数**：`estimate_workspace_volume(points=None, method='convex_hull')`

**功能**：估计工作空间的体积

**参数**：
- `points`: 工作空间点数组（可选）
- `method`: 体积估计方法
  - `'convex_hull'`: 凸包体积（上界，默认）
  - `'voxel'`: 体素估计
  - `'alpha_shape'`: Alpha形状体积（更准确，待实现）

**返回**：
- `volume`: 体积（立方米）

**使用示例**：

```python
# 凸包体积（上界）
volume_convex = analyzer.estimate_workspace_volume(method='convex_hull')

# 体素估计
volume_voxel = analyzer.estimate_workspace_volume(method='voxel')

print(f"工作空间体积: {volume_convex:.6f} m³")
```

---

#### 检查点可达性

**函数**：`check_point_in_workspace(point, points=None, tolerance=0.05)`

**功能**：检查给定点是否在工作空间内

**参数**：
- `point`: 3D点，形状 (3,)
- `points`: 工作空间点数组（可选）
- `tolerance`: 距离阈值（米，默认0.05）

**返回**：
- `in_workspace`: 布尔值，True表示可达

**使用示例**：

```python
target_point = np.array([0.5, 0.2, 0.3])
reachable = analyzer.check_point_in_workspace(target_point, tolerance=0.05)

if reachable:
    print("目标点可达")
else:
    print("目标点不可达")
```

---

#### 计算工作空间密度

**函数**：`compute_workspace_density(points=None, grid_resolution=20)`

**功能**：计算工作空间的密度分布

**参数**：
- `points`: 工作空间点数组（可选）
- `grid_resolution`: 网格分辨率（每轴，默认20）

**返回**：
- `density`: 3D密度数组，形状 (grid_resolution, grid_resolution, grid_resolution)
- `grid_coords`: 网格坐标数组 (x, y, z)

**使用示例**：

```python
density, (x, y, z) = analyzer.compute_workspace_density(grid_resolution=30)

# 找到密度最高的区域
max_idx = np.unravel_index(np.argmax(density), density.shape)
print(f"最密集区域: ({x[max_idx[0]]}, {y[max_idx[1]]}, {z[max_idx[2]]})")
```

---

#### 查找无奇异性区域

**函数**：`find_singularity_free_regions(...)`

**功能**：查找工作空间中无奇异性的区域

**参数**：
- `num_samples`: 采样数量（默认5000）
- `manipulability_threshold`: 最小可操作性阈值（默认0.01）
- `q_limits`: 关节限制（可选）

**返回**：
- `points`: 无奇异性工作空间点数组

**使用示例**：

```python
# 查找可操作性大于0.05的区域
safe_points = analyzer.find_singularity_free_regions(
    num_samples=10000,
    manipulability_threshold=0.05
)
```

---

#### 可视化工作空间

**函数**：`visualize_workspace(...)`

**功能**：3D可视化工作空间

**参数**：
- `points`: 工作空间点数组（可选）
- `show_bounds`: 是否显示边界框（默认True）
- `show_density`: 是否显示密度热图（默认False）
- `alpha`: 点透明度（默认0.3）
- `figsize`: 图形大小（默认(10, 8)）

**使用示例**：

```python
# 基本可视化
analyzer.visualize_workspace(show_bounds=True, alpha=0.1)

# 显示密度分布
analyzer.visualize_workspace(show_density=True, alpha=0.2)
```

---

### 2. 奇异性分析器（SingularityAnalyzer）

#### 初始化

**函数**：`SingularityAnalyzer(model)`

**参数**：
- `model`: RobotModel实例

**使用示例**：

```python
from robocore.analysis import SingularityAnalyzer

analyzer = SingularityAnalyzer(robot)
```

---

#### 分析配置

**函数**：`analyze_configuration(q)`

**功能**：分析单个关节配置的奇异性指标

**参数**：
- `q`: 关节配置

**返回**：
- `result`: 字典，包含以下键：
  - `manipulability`: 可操作性度量（Yoshikawa）
  - `condition_number`: 条件数（σ_max / σ_min）
  - `min_singular_value`: 最小奇异值
  - `max_singular_value`: 最大奇异值
  - `singular_values`: 所有奇异值列表
  - `is_singular`: 是否为奇异性配置（σ_min < 1e-3）

**使用示例**：

```python
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
result = analyzer.analyze_configuration(q)

print(f"可操作性: {result['manipulability']:.6f}")
print(f"条件数: {result['condition_number']:.2f}")
print(f"最小奇异值: {result['min_singular_value']:.6f}")
print(f"是否奇异: {result['is_singular']}")

# 分析多个配置
configs = {
    "零位": [0.0] * 6,
    "伸展": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
    "折叠": [0.0, 1.57, 0.0, 0.0, 0.0, 0.0],
}

for name, q in configs.items():
    result = analyzer.analyze_configuration(q)
    print(f"{name}: 可操作性={result['manipulability']:.6f}, "
          f"条件数={result['condition_number']:.2f}")
```

**指标说明**：
- **可操作性（Manipulability）**：`√det(J·J^T)`，衡量机器人在该配置下的灵活性
- **条件数（Condition Number）**：`σ_max / σ_min`，衡量雅可比矩阵的条件，值越大越接近奇异
- **奇异值（Singular Values）**：雅可比矩阵的奇异值，反映不同方向的可操作性

---

#### 工作空间采样分析

**函数**：`sample_workspace(n_samples=1000, seed=42)`

**功能**：随机采样工作空间，分析奇异性分布

**参数**：
- `n_samples`: 采样数量（默认1000）
- `seed`: 随机种子（默认42）

**返回**：
- `stats`: 字典，包含以下键：
  - `n_samples`: 有效样本数
  - `singular_configs`: 奇异性配置数量
  - `singular_ratio`: 奇异性配置比例
  - `manipulability_mean`: 平均可操作性
  - `manipulability_min`: 最小可操作性
  - `manipulability_max`: 最大可操作性
  - `condition_number_mean`: 平均条件数
  - `condition_number_max`: 最大条件数

**使用示例**：

```python
stats = analyzer.sample_workspace(n_samples=5000, seed=42)

print(f"采样数量: {stats['n_samples']}")
print(f"奇异性配置: {stats['singular_configs']} "
      f"({stats['singular_ratio']*100:.2f}%)")
print(f"可操作性: 平均={stats['manipulability_mean']:.6f}, "
      f"最小={stats['manipulability_min']:.6f}, "
      f"最大={stats['manipulability_max']:.6f}")
print(f"条件数: 平均={stats['condition_number_mean']:.2f}, "
      f"最大={stats['condition_number_max']:.2f}")
```

---

### 3. 工作空间对比分析

**函数**：`analyze_workspace_comparison(model1, model2, ...)`

**功能**：对比两个机器人的工作空间

**参数**：
- `model1, model2`: 两个RobotModel实例
- `num_samples`: 每个机器人的采样数量（默认10000）
- `names`: 机器人名称元组（可选）

**返回**：
- `comparison`: 字典，包含：
  - `names`: 机器人名称
  - `num_points`: 点数量元组
  - `bounds`: 边界元组
  - `volumes`: 体积元组
  - `volume_ratio`: 体积比
  - `points`: 点数组元组

**使用示例**：

```python
from robocore.analysis import analyze_workspace_comparison

model_6dof = RobotModel('robot_6dof.urdf')
model_7dof = RobotModel('robot_7dof.urdf')

comparison = analyze_workspace_comparison(
    model_6dof, model_7dof,
    num_samples=10000,
    names=("6-DOF Robot", "7-DOF Robot")
)

print(f"体积比: {comparison['volume_ratio']:.2f}")
print(f"6-DOF体积: {comparison['volumes'][0]:.6f} m³")
print(f"7-DOF体积: {comparison['volumes'][1]:.6f} m³")
```

---

## 分析指标对比

### 工作空间指标

| 指标 | 说明 | 计算方法 | 用途 |
|------|------|----------|------|
| **可达工作空间** | 所有可达点 | 正向运动学采样 | 评估机器人可达范围 |
| **灵巧工作空间** | 可多姿态到达的点 | 聚类分析 | 评估灵活性 |
| **工作空间体积** | 工作空间大小 | 凸包/体素估计 | 量化比较 |
| **工作空间密度** | 点分布密度 | 3D直方图 | 识别高密度区域 |
| **边界框** | 工作空间范围 | 最小/最大坐标 | 快速边界估计 |

### 奇异性指标

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| **可操作性** | `√det(J·J^T)` | [0, ∞) | 值越大越灵活 |
| **条件数** | `σ_max / σ_min` | [1, ∞) | 值越大越接近奇异 |
| **最小奇异值** | `σ_min` | [0, ∞) | 值越小越接近奇异 |
| **最大奇异值** | `σ_max` | [0, ∞) | 最大方向的可操作性 |

### 采样方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **蒙特卡洛** | 简单，快速 | 可能不均匀 | 快速估计 |
| **网格** | 均匀覆盖 | 维度灾难 | 低维机器人 |
| **Sobol序列** | 准随机，均匀 | 需要scipy | 高质量采样 |

---

## 使用指南

### 基本使用流程

1. **导入模块**
```python
from robocore.analysis import WorkspaceAnalyzer, SingularityAnalyzer
from robocore.modeling import RobotModel
```

2. **创建工作空间分析器**
```python
robot = RobotModel("path/to/robot.urdf")
workspace_analyzer = WorkspaceAnalyzer(robot)
```

3. **计算工作空间**
```python
points = workspace_analyzer.compute_reachable_workspace(
    num_samples=10000,
    method='monte_carlo'
)
```

4. **获取分析结果**
```python
bounds = workspace_analyzer.get_workspace_bounds()
volume = workspace_analyzer.estimate_workspace_volume()
```

5. **可视化**
```python
workspace_analyzer.visualize_workspace(show_bounds=True)
```

### 奇异性分析流程

1. **创建奇异性分析器**
```python
singularity_analyzer = SingularityAnalyzer(robot)
```

2. **分析单个配置**
```python
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
result = singularity_analyzer.analyze_configuration(q)
```

3. **工作空间采样分析**
```python
stats = singularity_analyzer.sample_workspace(n_samples=5000)
```

### 参数调优指南

#### 工作空间分析参数

**采样数量**：
- 快速估计：1000-5000
- 标准分析：10000-50000
- 高精度分析：100000+

**采样方法选择**：
- 快速预览：`monte_carlo`
- 均匀覆盖：`sobol`
- 低维机器人：`grid`

**并行处理**：
- 采样数量 > 10000 时启用
- 工作线程数：通常设为CPU核心数

#### 奇异性分析参数

**可操作性阈值**：
- 严格：0.1（避免接近奇异的配置）
- 标准：0.01（识别明显奇异性）
- 宽松：0.001（仅识别严重奇异性）

**采样数量**：
- 快速评估：500-1000
- 标准分析：1000-5000
- 详细分析：10000+

### 常见模式

#### 模式1：完整工作空间分析

```python
# 计算工作空间
points = analyzer.compute_reachable_workspace(
    num_samples=50000,
    method='sobol',
    use_parallel=True
)

# 获取指标
bounds = analyzer.get_workspace_bounds(points)
volume = analyzer.estimate_workspace_volume(points)

# 可视化
analyzer.visualize_workspace(points, show_density=True)
```

#### 模式2：奇异性检查

```python
# 分析配置
result = singularity_analyzer.analyze_configuration(q)

if result['is_singular']:
    print("警告：配置接近奇异")
    print(f"条件数: {result['condition_number']:.2f}")
else:
    print(f"可操作性: {result['manipulability']:.6f}")
```

#### 模式3：工作空间对比

```python
comparison = analyze_workspace_comparison(
    model1, model2,
    num_samples=20000
)

print(f"体积比: {comparison['volume_ratio']:.2f}")
```

---

## 实现状态

### Phase 1: 基础分析（已完成 ✅）

- [x] 可达工作空间计算
- [x] 工作空间边界和体积估计
- [x] 可达性检查
- [x] 奇异性配置分析
- [x] 可操作性度量

### Phase 2: 高级功能（已完成 ✅）

- [x] 灵巧工作空间计算
- [x] 工作空间密度分析
- [x] 无奇异性区域查找
- [x] 工作空间采样分析
- [x] 工作空间对比分析
- [x] 3D可视化

### Phase 3: 优化与扩展（部分完成 ⏳）

- [x] 多种采样方法（蒙特卡洛、网格、Sobol）
- [x] 并行处理支持
- [ ] Alpha形状体积估计
- [ ] 工作空间切片分析
- [ ] 可操作性椭球可视化
- [ ] 奇异性路径分析

### Phase 4: 高级分析（待实现 ⏳）

- [ ] 工作空间质量度量
- [ ] 可操作性分布分析
- [ ] 奇异性避免路径规划
- [ ] 多机器人工作空间分析
- [ ] 动态工作空间分析

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Robot Workspace Analysis** (Kumar & Waldron) - 工作空间分析专著

### 参考库

- **Pinocchio** - 高性能机器人学库（工作空间分析）
- **Robotics Toolbox** - MATLAB机器人工具箱
- **MoveIt!** - ROS运动规划框架（工作空间分析）

---

**文档版本**: 1.0  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

