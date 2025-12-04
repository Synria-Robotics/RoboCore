# RoboCore 控制器模块文档

## 目录

1. [概述](#概述)
2. [控制器架构](#控制器架构)
3. [控制器分类](#控制器分类)
4. [控制器详解](#控制器详解)
5. [控制器对比](#控制器对比)
6. [前馈控制的意义](#前馈控制的意义)
7. [MIT控制模式](#mit控制模式)
8. [算法层 vs 执行器层](#算法层-vs-执行器层)
9. [使用指南](#使用指南)
10. [实现状态](#实现状态)

---

## 概述

RoboCore控制器模块提供完整的机器人控制算法，支持关节空间（Joint Space）和笛卡尔空间（Cartesian Space）两大类控制器。

### 设计原则

- **算法层实现**：RoboCore提供控制算法，计算控制律，输出控制指令（力矩）
- **执行器层分离**：硬件驱动和执行器控制由硬件厂商/驱动器提供
- **双后端支持**：支持NumPy（CPU）和PyTorch（GPU）后端
- **统一接口**：所有控制器继承自`BaseController`，提供一致的接口

---

## 控制器架构

### 目录结构

```
robocore/control/
├── __init__.py                    # 模块导出
├── base.py                        # 基础控制器接口
├── joint/                         # 关节空间控制器
│   ├── __init__.py
│   ├── position.py                # Joint位置控制器（PD/PID）
│   ├── velocity.py                # Joint速度控制器
│   ├── trajectory.py              # Joint轨迹跟踪控制器（待实现）
│   ├── computed_torque.py         # Joint计算力矩控制器（待实现）
│   ├── impedance.py               # Joint阻抗控制器（待实现）
│   └── mpc.py                     # Joint MPC控制器（待实现）
├── cartesian/                     # 笛卡尔空间控制器
│   ├── __init__.py
│   ├── position.py                # Cartesian位置控制器
│   ├── velocity.py                # Cartesian速度控制器
│   ├── trajectory.py              # Cartesian轨迹跟踪控制器（待实现）
│   ├── impedance.py               # Cartesian阻抗控制器（待实现）
│   ├── admittance.py              # Cartesian导纳控制器（待实现）
│   ├── force.py                   # Cartesian力/力矩控制器（待实现）
│   ├── hybrid.py                  # Cartesian混合位置/力控制（待实现）
│   ├── operational_space.py       # Cartesian操作空间控制器（待实现）
│   └── mpc.py                     # Cartesian MPC控制器（待实现）
└── utils.py                       # 控制器工具函数（待实现）
```

### 基础接口

所有控制器继承自`BaseController`，提供统一接口：

```python
class BaseController(ABC):
    def compute(self, *args, **kwargs) -> Any:
        """计算控制力矩，返回 τ [n×1]"""
        raise NotImplementedError
    
    def reset(self):
        """重置控制器状态"""
        pass
    
    def update_params(self, **params):
        """更新控制参数"""
        pass
```

---

## 控制器分类

### Joint Space Controllers（关节空间控制器）

**特点**：输入输出都是关节空间量（q, qd, qdd, τ）

| 控制器 | 状态 | 说明 |
|--------|------|------|
| Joint位置控制器（PD/PID） | ✅ 已实现 | 支持PD和PID模式，可配置 |
| Joint速度控制器 | ✅ 已实现 | 纯速度反馈控制 |
| Joint轨迹跟踪控制器 | ⏳ 待实现 | 前馈+反馈，支持简化版和完整版 |
| Joint计算力矩控制器 | ⏳ 待实现 | 基于动力学模型的精确控制 |
| Joint阻抗控制器 | ⏳ 待实现 | 关节空间柔顺控制 |
| Joint MPC控制器 | ⏳ 待实现 | 模型预测控制 |

### Cartesian Space Controllers（笛卡尔空间控制器）

**特点**：输入是笛卡尔空间量（x, xd, xdd），输出是关节力矩（通过雅可比转置映射）

| 控制器 | 状态 | 说明 |
|--------|------|------|
| Cartesian位置控制器 | ✅ 已实现 | 通过雅可比转置映射到关节空间 |
| Cartesian速度控制器 | ✅ 已实现 | 笛卡尔空间速度跟踪 |
| Cartesian轨迹跟踪控制器 | ⏳ 待实现 | 操作空间轨迹跟踪 |
| Cartesian阻抗控制器 | ⏳ 待实现 | 操作空间柔顺控制 |
| Cartesian导纳控制器 | ⏳ 待实现 | 力到位置的映射 |
| Cartesian力/力矩控制器 | ⏳ 待实现 | 直接力控制 |
| Cartesian混合位置/力控制 | ⏳ 待实现 | 不同方向不同控制策略 |
| Cartesian操作空间控制器 | ⏳ 待实现 | 操作空间动力学控制 |
| Cartesian MPC控制器 | ⏳ 待实现 | 笛卡尔空间MPC |

---

## 控制器详解

### 1. Joint位置控制器（PD/PID）

**公式**：
- **PD模式**：`τ = Kp·(qd - q) + Kd·(q̇d - q̇)`
- **PID模式**：`τ = Kp·(qd - q) + Ki·∫(qd - q)dt + Kd·(q̇d - q̇)`

**参数**：
- `Kp`: 位置增益矩阵 [n×n] 或向量 [n×1]
- `Kd`: 速度增益矩阵 [n×n] 或向量 [n×1]
- `Ki`: 积分增益矩阵 [n×n] 或向量 [n×1]（PID模式）
- `use_integral`: 是否启用积分项（PID模式）
- `anti_windup`: 是否启用积分抗饱和

**使用示例**：
```python
from robocore.control import JointPositionController

# PD控制器
controller = JointPositionController(
    Kp=100.0,  # 可以是标量、向量或矩阵
    Kd=10.0,
    use_integral=False
)

# PID控制器
controller_pid = JointPositionController(
    Kp=100.0,
    Kd=10.0,
    Ki=1.0,
    use_integral=True,
    anti_windup=True
)

# 计算控制力矩
tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    qd_desired=desired_joint_positions,
    qdd_desired=desired_joint_velocities  # 可选
)
```

**适用场景**：
- 定点控制
- 简单轨迹跟踪
- PID模式适用于需要消除稳态误差的场景

---

### 2. Joint速度控制器

**公式**：`τ = Kp·(q̇d - q̇)`

**参数**：
- `Kp`: 速度增益矩阵 [n×n] 或向量 [n×1]

**使用示例**：
```python
from robocore.control import JointVelocityController

controller = JointVelocityController(Kp=50.0)

tau = controller.compute(
    q=current_joint_positions,  # 不使用，仅为接口一致性
    qd=current_joint_velocities,
    qdd_desired=desired_joint_velocities
)
```

**适用场景**：
- 速度跟踪任务
- 连续运动控制
- 不关心位置的场景

---

### 3. Cartesian位置控制器

**公式**：`τ = J^T·[Kp·(xd - x) + Kd·(ẋd - ẋ)] + g(q)`

其中：
- `J`: 雅可比矩阵 [6×n]
- `x`: 当前末端执行器位姿 [7×1] (位置3 + 四元数4)
- `xd`: 期望末端执行器位姿 [7×1]
- `ẋ`: 当前末端执行器速度 [6×1] (线速度3 + 角速度3)
- `ẋd`: 期望末端执行器速度 [6×1]
- `g(q)`: 重力向量 [n×1]（待实现动力学模块后支持）

**参数**：
- `robot_model`: RobotModel实例
- `Kp`: 位置增益矩阵 [6×6] 或对角向量 [6×1]
- `Kd`: 速度增益矩阵 [6×6] 或对角向量 [6×1]
- `use_gravity_compensation`: 是否启用重力补偿（待实现）
- `end_link`: 末端执行器链接名称

**使用示例**：
```python
from robocore.control import CartesianPositionController
from robocore.modeling import RobotModel

robot = RobotModel("path/to/robot.urdf")

controller = CartesianPositionController(
    robot_model=robot,
    Kp=np.diag([100, 100, 100, 50, 50, 50]),  # 位置和姿态增益
    Kd=np.diag([10, 10, 10, 5, 5, 5])
)

# 期望位姿：位置 [x, y, z] + 四元数 [x, y, z, w]
xd_desired = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.707, 0.707])

tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    xd_desired=xd_desired,
    xdd_desired=None  # 可选，期望速度
)
```

**适用场景**：
- 末端执行器定位
- 抓取前移动
- 操作空间控制

---

### 4. Cartesian速度控制器

**公式**：`τ = J^T·Kp·(ẋd - ẋ) + g(q)`

**参数**：
- `robot_model`: RobotModel实例
- `Kp`: 速度增益矩阵 [6×6] 或对角向量 [6×1]
- `use_gravity_compensation`: 是否启用重力补偿（待实现）
- `end_link`: 末端执行器链接名称

**使用示例**：
```python
from robocore.control import CartesianVelocityController

controller = CartesianVelocityController(
    robot_model=robot,
    Kp=np.diag([50, 50, 50, 30, 30, 30])
)

# 期望速度：线速度 [vx, vy, vz] + 角速度 [wx, wy, wz]
xdd_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    xdd_desired=xdd_desired
)
```

**适用场景**：
- 速度跟踪任务
- 连续运动控制
- 不关心位置的场景

---

### 5. Joint轨迹跟踪控制器

**公式（简化版，无动力学）**：
```
τ = Kp·(qd - q) + Kd·(q̇d - q̇) + Kff·q̈d
```

**公式（完整版，有动力学）**：
```
τ = M·q̈d + C·q̇d + g + Kp·(qd - q) + Kd·(q̇d - q̇)
```

其中：
- `Kp, Kd`: 反馈增益（位置和速度）
- `Kff`: 前馈增益（加速度补偿）
- `M, C, g`: 动力学项（质量矩阵、科里奥利力、重力）

**参数**：
- `Kp`: 位置增益矩阵 [n×n] 或向量 [n×1]
- `Kd`: 速度增益矩阵 [n×n] 或向量 [n×1]
- `Kff`: 前馈增益矩阵 [n×n] 或向量 [n×1]（可选，默认等于Kp）
- `use_dynamics`: 是否使用完整动力学模型（待实现）
- `robot_model`: RobotModel实例（use_dynamics=True时需要）

**使用示例**：

**方式1：直接提供期望状态**
```python
from robocore.control import JointTrajectoryController

controller = JointTrajectoryController(
    Kp=100.0,
    Kd=10.0,
    Kff=50.0  # 前馈增益
)

# 直接提供期望状态
tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    qd_desired=desired_joint_positions,
    qdd_desired=desired_joint_velocities,
    qddd_desired=desired_joint_accelerations
)
```

**方式2：使用轨迹数据**
```python
from robocore.planning.trajectory import cubic_polynomial_trajectory

# 生成轨迹
t, q, qd, qdd = cubic_polynomial_trajectory(
    q_start=np.array([0, 0, 0]),
    q_end=np.array([1, 0.5, 0.8]),
    duration=2.0
)

# 设置轨迹
controller.set_trajectory(trajectory_data={
    't': t,
    'q': q,
    'qd': qd,
    'qdd': qdd
})

# 时间跟踪模式
tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    t=current_time
)
```

**方式3：使用轨迹函数**
```python
def trajectory_func(t):
    # 自定义轨迹函数，返回 (qd, qdd, qddd)
    qd = compute_desired_position(t)
    qdd = compute_desired_velocity(t)
    qddd = compute_desired_acceleration(t)
    return (qd, qdd, qddd)

controller.set_trajectory(trajectory_func=trajectory_func)

tau = controller.compute(q=current_q, qd=current_qd, t=current_time)
```

**适用场景**：
- 跟踪已知轨迹
- 需要高精度轨迹跟踪
- 有前馈补偿的场景

**特点**：
- 支持时间参数化轨迹
- 前馈项补偿期望加速度
- 反馈项消除跟踪误差
- 简化版不需要动力学模型，计算快
- 完整版精度高，但需要动力学模型

---

### 6. Cartesian轨迹跟踪控制器

**公式（简化版，无动力学）**：
```
τ = J^T·[Kp·(xd - x) + Kd·(ẋd - ẋ) + Kff·ẍd] + g(q)
```

**公式（完整版，有动力学）**：
```
τ = J^T·Λ·[ẍd + Kp·(xd - x) + Kd·(ẋd - ẋ)] + μ + p
```

其中：
- `Λ = (J·M^-1·J^T)^-1`: 操作空间惯性矩阵 [6×6]
- `μ`: 操作空间科里奥利力 [6×1]
- `p`: 操作空间重力 [6×1]

**参数**：
- `robot_model`: RobotModel实例
- `Kp`: 位置增益矩阵 [6×6] 或对角向量 [6×1]
- `Kd`: 速度增益矩阵 [6×6] 或对角向量 [6×1]
- `Kff`: 前馈增益矩阵 [6×6] 或对角向量 [6×1]（可选，默认等于Kp）
- `use_dynamics`: 是否使用完整动力学模型（待实现）
- `use_gravity_compensation`: 是否启用重力补偿（待实现）
- `end_link`: 末端执行器链接名称

**使用示例**：

**方式1：直接提供期望状态**
```python
from robocore.control import CartesianTrajectoryController

controller = CartesianTrajectoryController(
    robot_model=robot,
    Kp=np.diag([100, 100, 100, 50, 50, 50]),
    Kd=np.diag([10, 10, 10, 5, 5, 5]),
    Kff=np.diag([50, 50, 50, 25, 25, 25])
)

# 直接提供期望状态
tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    xd_desired=desired_pose,  # [7×1] 位置+四元数
    xdd_desired=desired_velocity,  # [6×1]
    xddd_desired=desired_acceleration  # [6×1]
)
```

**方式2：使用轨迹数据**
```python
from robocore.planning.trajectory import linear_cartesian_trajectory

# 生成轨迹
t, poses, q = linear_cartesian_trajectory(
    robot_model=robot,
    pose_start=T_start,
    pose_end=T_end,
    duration=2.0
)

# 设置轨迹（poses是4×4矩阵数组）
controller.set_trajectory(trajectory_data={
    't': t,
    'poses': poses,
    'velocities': None,  # 可选，从poses计算
    'accelerations': None  # 可选
})

# 时间跟踪模式
tau = controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    t=current_time
)
```

**适用场景**：
- 笛卡尔空间轨迹跟踪
- 需要高精度操作空间控制
- 复杂轨迹跟踪任务

**特点**：
- 支持时间参数化轨迹
- 前馈项补偿期望加速度
- 反馈项消除跟踪误差
- 简化版不需要动力学模型
- 完整版精度高，但需要动力学模型

---

## 控制器对比

### Joint空间控制器对比表

| 控制器 | 公式 | 控制目标 | 需要动力学 | 需要传感器 | 参数含义 | 适用场景 | 特点 | 缺点 |
|--------|------|----------|------------|------------|----------|----------|------|------|
| **Joint位置控制器（PD/PID）** | `τ = Kp·e + Kd·ė + Ki·∫e·dt` | 到达并保持目标位置 | ❌ | ❌ | Kp:位置增益<br>Kd:速度增益<br>Ki:积分增益 | 定点控制<br>简单轨迹跟踪 | 简单易用<br>计算快<br>PID可消除稳态误差 | 无前馈补偿<br>跟踪精度一般 |
| **Joint速度控制器** | `τ = Kp·(q̇d - q̇)` | 维持目标速度 | ❌ | ❌ | Kp:速度增益 | 速度跟踪任务<br>连续运动 | 简单直接 | 不控制位置<br>可能累积误差 |
| **Joint轨迹跟踪（简化）** | `τ = Kp·e + Kd·ė + Kff·q̈d` | 跟踪已知轨迹 | ❌ | ❌ | Kp,Kd:反馈增益<br>Kff:前馈增益 | 轨迹跟踪<br>无动力学模型 | 有前馈补偿<br>计算快 | 精度有限<br>无动力学补偿 |
| **Joint轨迹跟踪（完整）** | `τ = M·q̈d + C·q̇d + g + Kp·e + Kd·ė` | 高精度轨迹跟踪 | ✅ | ❌ | Kp,Kd:反馈增益<br>M,C,g:动力学项 | 高精度轨迹跟踪 | 动力学前馈<br>精度高 | 需要动力学模型<br>计算复杂 |
| **Joint计算力矩** | `τ = M·[q̈d + Kp·e + Kd·ė] + C·q̇ + g` | 精确跟踪，线性化系统 | ✅ | ❌ | Kp,Kd:误差增益<br>M,C,g:动力学项 | 高精度控制<br>精确装配 | 完全补偿非线性<br>系统线性化 | 需要精确模型<br>对模型误差敏感 |
| **Joint阻抗控制** | `τ = M·(q̈d - q̈) + Dd·ė + Kd·e + C·q̇ + g` | 柔顺控制，允许位置偏差 | ✅ | ⚠️(可选) | Kd:刚度矩阵<br>Dd:阻尼矩阵<br>Md:惯性矩阵 | 接触任务<br>人机交互 | 柔顺响应<br>安全交互 | 需要加速度<br>参数调优复杂 |

### Cartesian空间控制器对比表

| 控制器 | 公式 | 控制目标 | 需要动力学 | 需要传感器 | 参数含义 | 适用场景 | 特点 | 缺点 |
|--------|------|----------|------------|------------|----------|----------|------|------|
| **Cartesian位置控制器** | `τ = J^T·[Kp·(xd-x) + Kd·(ẋd-ẋ)] + g` | 到达并保持目标位姿 | ❌ | ❌ | Kp:位置增益[6×6]<br>Kd:速度增益[6×6] | 末端执行器定位<br>抓取前移动 | 直观易用<br>操作空间控制 | 需要雅可比<br>奇异性问题 |
| **Cartesian速度控制器** | `τ = J^T·Kp·(ẋd - ẋ) + g` | 维持目标速度 | ❌ | ❌ | Kp:速度增益[6×6] | 速度跟踪<br>连续运动 | 简单直接 | 不控制位置<br>可能累积误差 |
| **Cartesian轨迹跟踪（简化）** | `τ = J^T·[Kp·e + Kd·ė + Kff·ẍd] + g` | 跟踪笛卡尔轨迹 | ❌ | ❌ | Kp,Kd:反馈增益<br>Kff:前馈增益 | 笛卡尔轨迹跟踪 | 有前馈补偿 | 精度有限 |
| **Cartesian轨迹跟踪（完整）** | `τ = J^T·Λ·[ẍd + Kp·e + Kd·ė] + μ + p` | 高精度笛卡尔跟踪 | ✅ | ❌ | Kp,Kd:反馈增益<br>Λ:操作空间惯性<br>μ:操作空间科氏力<br>p:操作空间重力 | 高精度操作<br>复杂轨迹 | 操作空间动力学<br>精度高 | 计算复杂<br>需要完整动力学 |
| **Cartesian阻抗控制** | `τ = J^T·[Md·(ẍd-ẍ) + Dd·(ẋd-ẋ) + Kd·(xd-x)] + g` | 柔顺控制，响应外力 | ✅ | ⚠️(可选) | Kd:刚度[6×6]<br>Dd:阻尼[6×6]<br>Md:惯性[6×6] | 接触任务<br>打磨抛光<br>人机交互 | 柔顺响应<br>安全交互 | 需要加速度<br>参数调优复杂 |
| **Cartesian导纳控制** | `ẍd = Md^-1·[Fext - Dd·(ẋd-ẋr) - Kd·(xd-xr)]`<br>`τ = J^T·[Md·ẍd] + g` | 力到位置的映射 | ❌ | ✅(力传感器) | Kd:刚度<br>Dd:阻尼<br>Md:惯性 | 力引导操作<br>人机协作 | 力到位置映射<br>与阻抗对偶 | 需要力传感器<br>响应可能滞后 |
| **Cartesian力/力矩控制** | `τ = J^T·Fd + g`<br>或<br>`τ = J^T·[Fd + Kp·(xd-x)] + g` | 直接控制力/力矩 | ❌ | ✅(力传感器) | Fd:期望力[6×1]<br>Kp:位置增益(可选) | 力控制任务<br>装配操作 | 直接力控制<br>精确力跟踪 | 需要力传感器<br>稳定性要求高 |
| **Cartesian混合位置/力控制** | `τ = J^T·[S·Fd + (I-S)·(Kp·e + Kd·ė)] + g` | 不同方向不同策略 | ❌ | ✅(力传感器) | S:选择矩阵[6×6]<br>Kp,Kd:位置增益 | 约束任务<br>沿表面移动 | 灵活控制策略 | 需要任务坐标系<br>选择矩阵设计复杂 |
| **Cartesian操作空间控制** | `τ = J^T·Λ·[ẍd + Kp·e + Kd·ė] + μ + p` | 操作空间精确控制 | ✅ | ❌ | Kp,Kd:反馈增益<br>Λ:操作空间惯性<br>μ:操作空间科氏力<br>p:操作空间重力 | 高精度操作<br>复杂任务 | 操作空间设计<br>性能最优 | 计算复杂度高<br>需要完整动力学 |

### 控制器特性总结

| 特性维度 | Joint位置 | Joint速度 | Joint轨迹跟踪 | Joint计算力矩 | Joint阻抗 | Cartesian位置 | Cartesian阻抗 | Cartesian力控制 |
|---------|-----------|-----------|---------------|---------------|-----------|---------------|----------------|-----------------|
| **复杂度** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **计算速度** | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡ | ⚡⚡ | ⚡ | ⚡⚡ |
| **精度** | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **柔顺性** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **安全性** | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **模型要求** | 无 | 无 | 可选 | 必须 | 必须 | 无 | 必须 | 无 |
| **传感器要求** | 无 | 无 | 无 | 无 | 可选 | 无 | 可选 | 必须 |
| **适用任务** | 定点 | 速度跟踪 | 轨迹跟踪 | 精确装配 | 接触任务 | 定位 | 柔顺操作 | 力控制 |

### 选择指南

**按任务类型选择：**
- **定点控制** → Joint位置控制器（PD/PID）
- **速度跟踪** → Joint/Cartesian速度控制器
- **轨迹跟踪** → Joint/Cartesian轨迹跟踪控制器
- **高精度控制** → Joint计算力矩 / Cartesian操作空间控制
- **接触任务** → Joint/Cartesian阻抗控制
- **力控制** → Cartesian力/力矩控制器
- **约束任务** → Cartesian混合位置/力控制
- **复杂优化** → MPC控制器

**按模型可用性选择：**
- **无动力学模型** → 位置/速度/轨迹跟踪（简化版）
- **有动力学模型** → 计算力矩/阻抗/操作空间控制

**按传感器可用性选择：**
- **无力传感器** → 位置/速度/轨迹跟踪/计算力矩
- **有力传感器** → 阻抗/导纳/力控制/混合控制

---

## 前馈控制的意义

### 为什么轨迹跟踪需要前馈？

#### 1. 反馈控制的局限性

**纯反馈控制（PD控制器）**：
```
τ = Kp·(qd - q) + Kd·(q̇d - q̇)
```

**问题**：
- 只能等误差出现后再修正
- 存在滞后：误差 → 检测 → 计算 → 执行 → 修正
- 对于快速变化的轨迹，总是"追着跑"

**例子**：跟踪一个快速移动的目标
```
时刻 t0: 目标在位置A，机器人在位置A（误差=0）
时刻 t1: 目标移动到位置B，机器人还在A（误差出现）
时刻 t2: 控制器检测到误差，开始修正
时刻 t3: 机器人开始向B移动
时刻 t4: 目标已经移动到位置C了（永远追不上！）
```

#### 2. 前馈控制的作用

**前馈+反馈控制**：
```
τ = Kp·(qd - q) + Kd·(q̇d - q̇) + Kff·q̈d
     ↑反馈项（修正误差）        ↑前馈项（预测补偿）
```

**前馈的意义**：
1. **预测性**：提前知道目标要加速，提前施加力矩
2. **主动补偿**：不等误差出现，直接补偿期望加速度
3. **减少滞后**：让机器人"提前准备"，而不是"被动追赶"

**例子**：同样的快速移动目标
```
时刻 t0: 目标在A，机器人也在A
时刻 t1: 目标要移动到B（我们知道轨迹！）
         → 前馈项：提前计算需要的加速度 q̈d
         → 提前施加力矩，让机器人开始加速
时刻 t2: 机器人已经在向B移动了（而不是等误差出现）
时刻 t3: 机器人到达B，误差很小（因为提前准备了）
```

#### 3. 物理直觉理解

**纯反馈控制（像"追车"）**：
```
你开车追前面的车：
- 看到距离远了 → 加速（反馈）
- 看到距离近了 → 减速（反馈）
- 总是慢一拍，永远在调整
```

**前馈+反馈控制（像"跟车"）**：
```
你开车跟着前面的车，并且知道它的行驶计划：
- 知道它要加速 → 提前加速（前馈）
- 知道它要转弯 → 提前准备（前馈）
- 同时观察距离 → 微调（反馈）
- 跟得很紧，很平滑
```

#### 4. 数学分析

**机器人动力学方程**：
```
M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ
```

**纯反馈控制**：
```
τ = Kp·e + Kd·ė
   = Kp·(qd - q) + Kd·(q̇d - q̇)

代入动力学：
M·q̈ + C·q̇ + g = Kp·(qd - q) + Kd·(q̇d - q̇)

问题：如果 qd 在快速变化（q̈d ≠ 0），
     机器人需要加速才能跟上，但反馈项不知道要加速多少！
```

**前馈+反馈控制**：
```
τ = Kp·e + Kd·ė + M·q̈d
   = Kp·(qd - q) + Kd·(q̇d - q̇) + M·q̈d

代入动力学：
M·q̈ + C·q̇ + g = Kp·(qd - q) + Kd·(q̇d - q̇) + M·q̈d

关键：前馈项 M·q̈d 直接提供了期望加速度所需的力矩！
     如果模型准确，M·q̈d 正好抵消 M·q̈d 项
     误差动力学变成：M·ë = Kp·e + Kd·ė（更简单！）
```

#### 5. 前馈项的两种形式

**简化前馈（无动力学模型）**：
```
τ = Kp·e + Kd·ė + Kff·q̈d
                    ↑
              简单加速度前馈
```
- 优点：不需要动力学模型，计算快
- 缺点：精度有限（没有补偿科氏力、重力等）

**完整前馈（有动力学模型）**：
```
τ = M·q̈d + C·q̇d + g + Kp·e + Kd·ė
   ↑前馈项（完整动力学补偿）  ↑反馈项
```
- 优点：精度高，完全补偿非线性
- 缺点：需要动力学模型，计算复杂

#### 6. 总结

**前馈的意义**：
1. **预测性补偿**：提前知道要做什么，提前准备
2. **减少滞后**：不等误差出现，主动补偿
3. **提高精度**：特别是对于快速变化的轨迹
4. **更平滑**：减少震荡和超调

**前馈的本质**：
- 利用已知的期望轨迹信息（q̈d）
- 提前计算需要的控制量
- 让系统"主动跟随"而不是"被动追赶"

**类比**：
- 纯反馈 = 看后视镜开车（总是慢一拍）
- 前馈+反馈 = 看导航+后视镜（提前知道路线，提前准备）

---

## MIT控制模式

### 什么是MIT控制？

MIT控制模式（MIT Mode）是一种**执行器级别**的控制方法，由MIT开发，用于精确力矩控制。它混合了位置、速度和电流（力矩）控制。

### MIT控制的公式

```
c_ref = Kp·(qd - q) + Kd·(q̇d - q̇) + τff
        ↑位置反馈    ↑速度反馈    ↑前馈力矩
```

其中：
- `c_ref`: 参考电流（控制输出）
- `Kp`: 位置增益（刚度）
- `Kd`: 速度增益（阻尼）
- `qd, q̇d`: 期望位置和速度
- `q, q̇`: 实际位置和速度
- `τff`: 前馈力矩（可选）

### MIT控制 vs Joint轨迹跟踪控制器

**MIT控制模式 ≈ Joint轨迹跟踪控制器**

**相同点**：
- 算法结构相同：PD反馈 + 前馈
- 公式形式相似

**区别**：
1. **输出单位**：电流 vs 力矩（可通过电机常数转换）
2. **前馈项形式**：直接力矩 vs 从加速度/动力学计算
3. **实现层级**：执行器级 vs 算法级

**结论**：
- MIT控制模式本质上就是Joint轨迹跟踪控制器，只是输出形式不同
- 在我们的规划中，Joint轨迹跟踪控制器可以支持MIT控制模式（通过前馈项参数）

---

## 算法层 vs 执行器层

### 系统层次划分

```
┌─────────────────────────────────────┐
│  应用层 (Application Layer)         │  ← 任务规划、轨迹生成
│  - 任务规划                         │
│  - 轨迹规划                         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  算法层 (Algorithm Layer)            │  ← RoboCore在这里
│  - 控制律计算                       │
│  - 输出：控制指令（力矩/位置/速度）  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  执行器层 (Actuator Layer)          │  ← 电机驱动器/硬件
│  - 接收控制指令                     │
│  - 电流控制、PWM生成                │
│  - 硬件驱动                         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  物理层 (Physical Layer)             │  ← 实际机器人
│  - 电机、关节                       │
│  - 传感器反馈                       │
└─────────────────────────────────────┘
```

### 算法层 vs 执行器层对比

| 特性 | 算法层（Algorithm Level） | 执行器层（Actuator Level） |
|------|--------------------------|---------------------------|
| **位置** | 软件/控制算法 | 硬件/驱动器 |
| **输入** | 期望状态、当前状态 | 控制指令（力矩/位置/速度） |
| **输出** | 控制指令（τ, qd, q̇d） | 电流/PWM信号 |
| **计算内容** | 控制律（PD、PID、计算力矩等） | 电流环、位置环、速度环 |
| **实现方式** | Python/C++算法代码 | 固件/FPGA/ASIC |
| **更新频率** | 1-10kHz（控制循环） | 10-100kHz（电流环） |
| **典型代表** | RoboCore控制器 | 电机驱动器（如MIT控制模式） |

### 我们的控制器分类

#### 算法层控制器（RoboCore提供）

所有我们规划的控制器都在算法层，计算控制律，输出控制指令：

- Joint位置控制器（PD/PID） → 输出：`τ` 或 `qd`
- Joint速度控制器 → 输出：`τ` 或 `q̇d`
- Joint轨迹跟踪控制器 → 输出：`τ`
- Joint计算力矩控制器 → 输出：`τ`
- Joint阻抗控制器 → 输出：`τ`
- Cartesian位置控制器 → 输出：`τ`
- Cartesian速度控制器 → 输出：`τ`
- 所有Cartesian控制器 → 输出：`τ`

#### 执行器层控制器（硬件/驱动器提供）

这些在执行器层，接收指令，驱动硬件：

- MIT控制模式 → 输入：`qd, q̇d, τff`，输出：`c_ref`（电流）
- 位置伺服控制 → 输入：`qd`，输出：PWM/电流
- 速度伺服控制 → 输入：`q̇d`，输出：PWM/电流
- 电流环控制 → 输入：`τ` 或 `c_ref`，输出：实际电流

### 数据流示例

#### 场景1：使用算法层控制器

```
算法层（RoboCore）：
  Joint位置控制器.compute()
    → 输出：τ = [10, 5, 3, ...] N·m
  
执行器层（驱动器）：
  接收：τ = [10, 5, 3, ...] N·m
  转换：c_ref = τ / Kt  (Kt是力矩常数)
  输出：实际电流到电机
  
物理层：
  电机产生力矩，机器人运动
```

#### 场景2：使用执行器层控制器（MIT模式）

```
算法层（RoboCore）：
  计算前馈力矩：τff = inverse_dynamics(...)
  输出：qd, q̇d, τff
  
执行器层（驱动器，MIT模式）：
  接收：qd, q̇d, τff
  计算：c_ref = Kp·(qd - q) + Kd·(q̇d - q̇) + τff
  输出：实际电流到电机
  
物理层：
  电机产生力矩，机器人运动
```

### 总结

- **算法层（RoboCore）**：所有我们规划的控制器，计算控制律，输出控制指令
- **执行器层（硬件/驱动器）**：MIT控制模式、位置/速度/电流伺服环，接收指令，驱动硬件
- **关键区别**：算法层决定"做什么"（计算控制律），执行器层决定"怎么做"（驱动硬件）

RoboCore专注于算法层，提供控制算法；执行器层由硬件/驱动器厂商提供。

---

## 使用指南

### 基本使用流程

1. **导入控制器**
```python
from robocore.control import (
    JointPositionController,
    JointVelocityController,
    CartesianPositionController,
    CartesianVelocityController
)
```

2. **创建控制器实例**
```python
# Joint位置控制器
joint_controller = JointPositionController(
    Kp=100.0,
    Kd=10.0,
    use_integral=False  # PD模式
)

# Cartesian位置控制器
cartesian_controller = CartesianPositionController(
    robot_model=robot,
    Kp=np.diag([100, 100, 100, 50, 50, 50]),
    Kd=np.diag([10, 10, 10, 5, 5, 5])
)
```

3. **计算控制力矩**
```python
# Joint控制
tau = joint_controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    qd_desired=desired_joint_positions,
    qdd_desired=desired_joint_velocities
)

# Cartesian控制
tau = cartesian_controller.compute(
    q=current_joint_positions,
    qd=current_joint_velocities,
    xd_desired=desired_end_effector_pose
)
```

4. **应用控制力矩**
```python
# 发送到机器人（具体实现取决于硬件接口）
robot.set_torques(tau)
```

### 参数调优指南

#### Joint位置控制器

**初始参数设置**：
- `Kp`: 从较小值开始（如10-50），逐步增加直到响应快速但不过冲
- `Kd`: 通常设为Kp的0.1-0.2倍，用于阻尼
- `Ki`: 如果需要消除稳态误差，从Kp的0.01-0.1倍开始

**调优步骤**：
1. 先调Kp，使系统响应快速
2. 再调Kd，消除震荡
3. 最后调Ki（如果需要），消除稳态误差

#### Cartesian位置控制器

**初始参数设置**：
- 位置增益（前3个）：100-500
- 姿态增益（后3个）：50-200（通常比位置增益小）
- 速度增益：位置增益的0.1-0.2倍

**注意事项**：
- 奇异性附近需要降低增益
- 不同方向可以设置不同的增益（使用对角矩阵）

---

## 实现状态

### Phase 1: 基础控制器（已完成 ✅）

- [x] Joint位置控制器（PD/PID，可配置）
- [x] Joint速度控制器
- [x] Cartesian位置控制器
- [x] Cartesian速度控制器

### Phase 2: 轨迹跟踪（已完成 ✅）

- [x] Joint轨迹跟踪控制器（简化版，无动力学）
- [ ] Joint轨迹跟踪控制器（完整版，有动力学）- 待动力学模块
- [x] Cartesian轨迹跟踪控制器（简化版，无动力学）
- [ ] Cartesian轨迹跟踪控制器（完整版，有动力学）- 待动力学模块

### Phase 3: 高级控制器（待实现 ⏳）

- [ ] Joint计算力矩控制器
- [ ] Joint阻抗控制器
- [ ] Cartesian操作空间控制器
- [ ] Cartesian阻抗控制器

### Phase 4: 力控制（待实现 ⏳）

- [ ] Cartesian力/力矩控制器
- [ ] Cartesian导纳控制器
- [ ] Cartesian混合位置/力控制

### Phase 5: 优化控制（待实现 ⏳）

- [ ] Joint MPC控制器
- [ ] Cartesian MPC控制器

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Introduction to Robotics** (Craig) - 经典教材

### 参考库

- **Pinocchio** - 高性能动力学库
- **Drake** - MIT机器人工具箱
- **Franka Control Interface** - Franka机器人控制接口

---

**文档版本**: 1.0  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

