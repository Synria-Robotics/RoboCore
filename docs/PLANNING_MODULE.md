# RoboCore 轨迹规划模块文档

## 目录

1. [概述](#概述)
2. [规划空间分类](#规划空间分类)
3. [关节空间轨迹规划](#关节空间轨迹规划)
4. [笛卡尔空间轨迹规划](#笛卡尔空间轨迹规划)
5. [速度曲线生成](#速度曲线生成)
6. [规划器对比](#规划器对比)
7. [使用指南](#使用指南)
8. [实现状态](#实现状态)

---

## 概述

RoboCore轨迹规划模块提供完整的机器人轨迹规划算法，支持关节空间（Joint Space）和笛卡尔空间（Cartesian Space）两大类规划方法。

### 设计原则

- **按规划空间分类**：清晰区分关节空间和笛卡尔空间规划
- **统一接口**：所有规划器继承自`BaseTrajectoryPlanner`，提供一致的接口
- **双后端支持**：支持NumPy（CPU）和PyTorch（GPU）后端
- **模块化设计**：不同规划方法独立实现，易于扩展

---

## 规划空间分类

### 关节空间规划 (Joint Space Planning)

**特点**：
- 直接在关节角度空间中生成轨迹
- 优点：计算简单，不会遇到奇异点
- 缺点：无法直接控制末端执行器路径

**适用场景**：
- 关节空间运动
- 避障规划
- 不需要精确控制末端路径的场景

### 笛卡尔空间规划 (Cartesian Space Planning)

**特点**：
- 在末端执行器的位置和姿态空间中生成轨迹
- 优点：可以精确控制末端路径
- 缺点：需要实时逆运动学求解，可能遇到奇异点

**适用场景**：
- 精确的末端路径控制
- 焊接、切割等需要特定路径的应用
- 操作空间任务

---

## 关节空间轨迹规划

### 1. 三次多项式插值 (Cubic Polynomial)

**类名**：`CubicPolynomialPlanner`

**特点**：
- C1连续（位置和速度连续）
- 需要起点和终点的位置和速度

**公式**：
```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3
```

**使用示例**：
```python
from robocore.planning import CubicPolynomialPlanner

planner = CubicPolynomialPlanner()

result = planner.plan(
    start=np.array([0, 0, 0]),
    end=np.array([1, 0.5, 0.8]),
    duration=2.0,
    num_points=100,
    qd_start=np.array([0, 0, 0]),  # 可选，默认零速度
    qd_end=np.array([0, 0, 0])     # 可选，默认零速度
)

# 结果包含：t, q, qd, qdd
t = result['t']
q = result['q']      # 位置 [num_points, n_joints]
qd = result['qd']    # 速度 [num_points, n_joints]
qdd = result['qdd']  # 加速度 [num_points, n_joints]
```

**适用场景**：
- 简单的点到点运动
- 需要速度连续但不需要加速度连续的场景

---

### 2. 五次多项式插值 (Quintic Polynomial)

**类名**：`QuinticPolynomialPlanner`

**特点**：
- C2连续（位置、速度、加速度都连续）
- 需要起点和终点的位置、速度、加速度

**公式**：
```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
```

**使用示例**：
```python
from robocore.planning import QuinticPolynomialPlanner

planner = QuinticPolynomialPlanner()

result = planner.plan(
    start=np.array([0, 0, 0]),
    end=np.array([1, 0.5, 0.8]),
    duration=2.0,
    num_points=100,
    qd_start=np.array([0, 0, 0]),      # 可选
    qd_end=np.array([0, 0, 0]),        # 可选
    qdd_start=np.array([0, 0, 0]),     # 可选
    qdd_end=np.array([0, 0, 0])        # 可选
)
```

**适用场景**：
- 需要平滑运动的场景
- 对加速度连续性有要求的应用
- 大多数轨迹规划任务

---

### 3. 七次多项式插值 (Septic Polynomial)

**类名**：`SepticPolynomialPlanner`

**特点**：
- C3连续（位置、速度、加速度、加加速度都连续）
- 需要起点和终点的位置、速度、加速度、加加速度(jerk)

**公式**：
```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 + a6*t^6 + a7*t^7
```

**使用示例**：
```python
from robocore.planning import SepticPolynomialPlanner

planner = SepticPolynomialPlanner()

result = planner.plan(
    start=np.array([0, 0, 0]),
    end=np.array([1, 0.5, 0.8]),
    duration=2.0,
    num_points=100,
    qd_start=np.array([0, 0, 0]),
    qd_end=np.array([0, 0, 0]),
    qdd_start=np.array([0, 0, 0]),
    qdd_end=np.array([0, 0, 0]),
    qddd_start=np.array([0, 0, 0]),   # 加加速度
    qddd_end=np.array([0, 0, 0])      # 加加速度
)

# 结果还包含 qddd (加加速度)
qddd = result['qddd']
```

**适用场景**：
- 高精度、高平滑度要求
- 对振动敏感的应用
- 需要最小化加加速度的场景

---

### 4. B样条插值 (B-Spline)

**类名**：`BSplinePlanner`

**特点**：
- 通过多个中间点生成平滑曲线
- 局部控制性好
- 支持三次（degree=3）和五次（degree=5）B样条

**使用示例**：
```python
from robocore.planning import BSplinePlanner

# 三次B样条
planner = BSplinePlanner(degree=3)

# 五次B样条
planner_quintic = BSplinePlanner(degree=5)

# 通过多个路径点
waypoints = np.array([
    [0, 0, 0],
    [0.3, 0.2, 0.1],
    [0.6, 0.4, 0.3],
    [1, 0.5, 0.8]
])

result = planner.plan(
    waypoints=waypoints,
    duration=3.0,
    num_points=150
)
```

**适用场景**：
- 通过多个中间点的复杂路径
- 需要局部控制轨迹形状
- 复杂路径规划

---

### 5. 多段轨迹规划 (Multi-Segment)

**类名**：`MultiSegmentPlanner`

**特点**：
- 通过多个路径点的分段规划
- 每段使用多项式插值（三次或五次）
- 保证段间连续性

**使用示例**：
```python
from robocore.planning import MultiSegmentPlanner

# 使用五次多项式连接各段
planner = MultiSegmentPlanner(method='quintic')

waypoints = np.array([
    [0, 0, 0],
    [0.3, 0.2, 0.1],
    [0.6, 0.4, 0.3],
    [1, 0.5, 0.8]
])

result = planner.plan(
    waypoints=waypoints,
    durations=1.0,  # 每段1秒，或提供数组 [1.0, 1.5, 1.0]
    num_points_per_segment=50
)
```

**适用场景**：
- 通过多个路径点的轨迹
- 需要段间连续性的场景
- 分段优化的轨迹

---

## 笛卡尔空间轨迹规划

### 1. 直线位置插值 (Linear Position)

**类名**：`LinearPositionPlanner`

**特点**：
- 末端执行器沿直线运动
- 位置线性插值

**使用示例**：
```python
from robocore.planning import LinearPositionPlanner

planner = LinearPositionPlanner()

# 输入可以是位置 [3] 或变换矩阵 [4, 4]
result = planner.plan(
    start=np.array([0.3, 0.2, 0.1]),  # 或 4x4 变换矩阵
    end=np.array([0.5, 0.4, 0.3]),
    duration=2.0,
    num_points=100
)

# 结果包含
positions = result['positions']        # [num_points, 3]
velocities = result['velocities']     # [num_points, 3]
accelerations = result['accelerations'] # [num_points, 3]
```

**适用场景**：
- 简单的直线运动
- 抓取前移动
- 不需要平滑曲线的场景

---

### 2. SLERP姿态插值 (Spherical Linear Interpolation)

**类名**：`SLERPPlanner`

**特点**：
- 球面线性插值，保证姿态旋转的最短路径
- 使用四元数表示姿态

**使用示例**：
```python
from robocore.planning import SLERPPlanner

planner = SLERPPlanner()

# 输入可以是四元数 [4]、旋转矩阵 [3, 3] 或变换矩阵 [4, 4]
R_start = np.eye(3)  # 或四元数、变换矩阵
R_end = ...  # 目标旋转

result = planner.plan(
    start=R_start,
    end=R_end,
    duration=2.0,
    num_points=100
)

# 结果包含
orientations = result['orientations']           # 四元数 [num_points, 4]
angular_velocities = result['angular_velocities']     # [num_points, 3]
angular_accelerations = result['angular_accelerations'] # [num_points, 3]
```

**适用场景**：
- 姿态插值
- 需要最短旋转路径的场景
- 与位置规划组合使用

---

### 3. 圆弧插值 (Circular Arc)

**类名**：`CircularArcPlanner`

**特点**：
- 通过三个点（起点、中间点、终点）定义圆弧
- 位置沿圆弧运动，姿态使用SLERP插值

**使用示例**：
```python
from robocore.planning import CircularArcPlanner

planner = CircularArcPlanner()

# 三个点定义圆弧
result = planner.plan(
    start=np.array([0.3, 0.2, 0.1]),      # 起点
    via=np.array([0.4, 0.3, 0.25]),       # 中间点
    end=np.array([0.5, 0.4, 0.3]),        # 终点
    duration=2.0,
    num_points=100
)

# 结果包含完整的位姿信息
poses = result['poses']                   # [num_points, 4, 4] 变换矩阵
positions = result['positions']           # [num_points, 3]
orientations = result['orientations']     # [num_points, 3, 3]
velocities = result['velocities']         # [num_points, 6] (线速度3 + 角速度3)
accelerations = result['accelerations']   # [num_points, 6]
```

**适用场景**：
- 焊接、切割等需要圆弧路径的应用
- 绕过障碍物的路径
- 需要平滑曲线的操作

---

### 4. 样条曲线插值 (Spline Curve)

**类名**：`SplineCurvePlanner`

**特点**：
- 通过多个路径点生成平滑样条曲线
- 位置使用三次样条插值，姿态使用SLERP

**使用示例**：
```python
from robocore.planning import SplineCurvePlanner

planner = SplineCurvePlanner()

# 多个路径点（可以是位置或变换矩阵）
waypoints = np.array([
    [0.3, 0.2, 0.1],
    [0.35, 0.25, 0.15],
    [0.4, 0.3, 0.2],
    [0.5, 0.4, 0.3]
])

# 或使用变换矩阵
# waypoints = np.array([T1, T2, T3, T4])  # shape: [4, 4, 4]

result = planner.plan(
    waypoints=waypoints,
    duration=3.0,
    num_points=150
)
```

**适用场景**：
- 通过多个路径点的复杂轨迹
- 需要平滑曲线的操作
- 复杂路径规划

---

## 速度曲线生成

### 1. 梯形速度曲线 (Trapezoidal Velocity Profile)

**类名**：`TrapezoidalVelocityProfile`

**特点**：
- 加速-匀速-减速三段式
- 计算简单，易于实现
- 加速度不连续

**使用示例**：
```python
from robocore.planning import TrapezoidalVelocityProfile

planner = TrapezoidalVelocityProfile()

result = planner.plan(
    start=0.0,
    end=1.0,
    duration=2.0,  # 或提供 v_max 和 a_max
    num_points=100,
    v_max=0.5,     # 可选
    a_max=1.0      # 可选
)

# 结果包含
t = result['t']
s = result['s']    # 位置
v = result['v']    # 速度
a = result['a']    # 加速度
```

**适用场景**：
- 简单的速度规划
- 不需要加速度连续的场景
- 快速实现

---

### 2. S曲线速度曲线 (S-Curve Velocity Profile)

**类名**：`SCurveVelocityProfile`

**特点**：
- 加加速度(jerk)受限的平滑曲线
- 加速度连续变化
- 适用于对振动敏感的应用

**使用示例**：
```python
from robocore.planning import SCurveVelocityProfile

planner = SCurveVelocityProfile()

result = planner.plan(
    start=0.0,
    end=1.0,
    duration=2.0,
    num_points=100,
    v_max=0.5,
    a_max=1.0,
    j_max=5.0  # 最大加加速度
)

# 结果还包含加加速度
j = result['j']  # 加加速度
```

**适用场景**：
- 对振动敏感的应用
- 需要平滑加速度变化的场景
- 高精度运动控制

---

## 规划器对比

### 关节空间规划器对比

| 规划器 | 连续性 | 需要参数 | 计算复杂度 | 适用场景 |
|--------|--------|----------|------------|----------|
| **三次多项式** | C1 | 位置、速度 | ⭐ | 简单点到点运动 |
| **五次多项式** | C2 | 位置、速度、加速度 | ⭐⭐ | 大多数轨迹规划 |
| **七次多项式** | C3 | 位置、速度、加速度、加加速度 | ⭐⭐⭐ | 高精度、高平滑度 |
| **B样条** | C2/C3 | 多个路径点 | ⭐⭐⭐ | 复杂路径 |
| **多段轨迹** | C1/C2 | 多个路径点 | ⭐⭐ | 分段优化 |

### 笛卡尔空间规划器对比

| 规划器 | 路径类型 | 需要参数 | 计算复杂度 | 适用场景 |
|--------|----------|----------|------------|----------|
| **直线插值** | 直线 | 起点、终点 | ⭐ | 简单直线运动 |
| **SLERP** | 球面 | 起点、终点姿态 | ⭐ | 姿态插值 |
| **圆弧插值** | 圆弧 | 起点、中间点、终点 | ⭐⭐ | 焊接、切割 |
| **样条曲线** | 样条 | 多个路径点 | ⭐⭐⭐ | 复杂路径 |

### 速度曲线对比

| 曲线类型 | 加速度连续性 | 加加速度限制 | 计算复杂度 | 适用场景 |
|----------|--------------|--------------|------------|----------|
| **梯形** | ❌ | ❌ | ⭐ | 简单快速 |
| **S曲线** | ✅ | ✅ | ⭐⭐ | 平滑运动 |

---

## 使用指南

### 基本使用流程

1. **选择规划器**
```python
from robocore.planning import QuinticPolynomialPlanner
```

2. **创建规划器实例**
```python
planner = QuinticPolynomialPlanner()
```

3. **生成轨迹**
```python
result = planner.plan(
    start=q_start,
    end=q_end,
    duration=2.0,
    num_points=100
)
```

4. **使用轨迹数据**
```python
t = result['t']
q = result['q']
qd = result['qd']
qdd = result['qdd']

# 可以用于控制器
for i in range(len(t)):
    tau = controller.compute(
        q=current_q,
        qd=current_qd,
        qd_desired=q[i],
        qdd_desired=qd[i]
    )
```

### 组合使用示例

**关节空间轨迹 + 控制器**：
```python
from robocore.planning import QuinticPolynomialPlanner
from robocore.control import JointTrajectoryController

# 生成轨迹
planner = QuinticPolynomialPlanner()
trajectory = planner.plan(
    start=np.array([0, 0, 0]),
    end=np.array([1, 0.5, 0.8]),
    duration=2.0,
    num_points=100
)

# 设置控制器轨迹
controller = JointTrajectoryController(Kp=100.0, Kd=10.0)
controller.set_trajectory(trajectory_data={
    't': trajectory['t'],
    'q': trajectory['q'],
    'qd': trajectory['qd'],
    'qdd': trajectory['qdd']
})

# 在控制循环中使用
tau = controller.compute(q=current_q, qd=current_qd, t=current_time)
```

**笛卡尔空间轨迹**：
```python
from robocore.planning import CircularArcPlanner
from robocore.kinematics import inverse_kinematics

# 生成圆弧轨迹
planner = CircularArcPlanner()
trajectory = planner.plan(
    start=T_start,
    via=T_via,
    end=T_end,
    duration=2.0,
    num_points=100
)

# 转换为关节空间（需要IK）
q_trajectory = []
for pose in trajectory['poses']:
    ik_result = inverse_kinematics(robot_model, pose, q_init)
    q_trajectory.append(ik_result['q'])
```

---

## 实现状态

### Phase 1: 基础规划（已完成 ✅）

- [x] 三次多项式插值
- [x] 五次多项式插值
- [x] 直线位置插值
- [x] SLERP姿态插值
- [x] 梯形速度曲线
- [x] S曲线速度曲线

### Phase 2: 高级插值方法（已完成 ✅）

- [x] 七次多项式插值
- [x] B样条插值
- [x] 多段轨迹规划
- [x] 圆弧插值
- [x] 样条曲线插值

### Phase 3: 冗余机械臂支持（待实现 ⏳）

- [ ] 零空间规划
- [ ] 任务优先级规划
- [ ] 扩展任务空间规划

### Phase 4: 混合空间和优化（待实现 ⏳）

- [ ] 混合空间规划
- [ ] 轨迹优化（时间最优、能耗最优等）

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Trajectory Planning for Automatic Machines and Robots** (Bianchi & Riviere) - 轨迹规划专著

### 参考库

- **Pinocchio** - 高性能动力学和轨迹规划库
- **MoveIt!** - ROS运动规划框架
- **OMPL** - 开源运动规划库

---

**文档版本**: 1.0  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

