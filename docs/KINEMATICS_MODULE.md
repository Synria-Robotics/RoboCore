# RoboCore 运动学模块文档

## 目录

1. [概述](#概述)
2. [模块架构](#模块架构)
3. [功能分类](#功能分类)
4. [功能详解](#功能详解)
5. [功能对比](#功能对比)
6. [双臂运动学](#双臂运动学)
7. [任务抽象](#任务抽象)
8. [使用指南](#使用指南)
9. [实现状态](#实现状态)
10. [最新更新](#最新更新2025-01)

---

## 概述

RoboCore运动学模块提供完整的机器人运动学计算功能，支持正向运动学（Forward Kinematics）、逆向运动学（Inverse Kinematics）、雅可比矩阵计算，以及双臂协调运动学。

### 设计原则

- **统一接口**：提供高级统一API，隐藏后端实现细节
- **双后端支持**：支持NumPy（CPU）和PyTorch（GPU）后端
- **单链与多链**：支持单链运动学和多链运动学（多末端执行器）
- **双臂协调**：支持双臂独立、相对位姿、镜像等多种协调模式
- **灵活配置**：支持部分任务、零空间优化、多起点求解等高级功能

---

## 模块架构

### 目录结构

```
robocore/kinematics/
├── __init__.py                    # 模块导出
├── fk.py                          # 正向运动学统一接口
├── ik.py                          # 逆向运动学统一接口
├── jacobian.py                    # 雅可比矩阵统一接口
├── bimanual.py                    # 双臂运动学接口
├── task.py                        # 任务抽象（多任务控制）
├── utils.py                       # 工具函数
├── fk_utils/                      # 正向运动学求解器
│   ├── fk_solver_numpy.py         # NumPy后端FK求解器
│   ├── fk_solver_torch.py         # PyTorch后端FK求解器
│   ├── bimanual_fk_solver_numpy.py # 双臂FK求解器（NumPy）
│   └── bimanual_fk_solver_torch.py # 双臂FK求解器（PyTorch）
├── ik_utils/                      # 逆向运动学求解器
│   ├── ik_solver_numpy.py         # NumPy后端IK求解器
│   ├── ik_solver_torch.py         # PyTorch后端IK求解器
│   ├── bimanual_ik_solver_numpy.py # 双臂IK求解器（NumPy）
│   └── bimanual_ik_solver_torch.py # 双臂IK求解器（PyTorch）
├── jacobian_utils/                # 雅可比矩阵求解器
│   ├── jacobian_solver_numpy.py   # NumPy后端雅可比求解器
│   ├── jacobian_solver_torch.py   # PyTorch后端雅可比求解器
│   ├── bimanual_jacobian_solver_numpy.py # 双臂雅可比求解器（NumPy）
│   └── bimanual_jacobian_solver_torch.py # 双臂雅可比求解器（PyTorch）
└── solvers/                       # 高级求解器
    └── multi_chain_solver.py      # 多链IK求解器
```

### 统一接口

所有运动学功能通过统一的高级接口访问：

```python
from robocore.kinematics import (
    forward_kinematics,      # 正向运动学
    inverse_kinematics,      # 逆向运动学
    jacobian,                # 雅可比矩阵
    bimanual_forward_kinematics,  # 双臂正向运动学
    bimanual_inverse_kinematics,  # 双臂逆向运动学
    bimanual_jacobian,       # 双臂雅可比矩阵
)
```

---

## 功能分类

### 单链运动学（Single Chain Kinematics）

**特点**：输入输出都是单条运动链（一个末端执行器）

| 功能 | 状态 | 说明 |
|------|------|------|
| 正向运动学（FK） | ✅ 已实现 | 从关节角度计算末端执行器位姿 |
| 逆向运动学（IK） | ✅ 已实现 | 从目标位姿计算关节角度 |
| 雅可比矩阵 | ✅ 已实现 | 计算速度雅可比矩阵 |

### 多链运动学（Multi-Chain Kinematics）

**特点**：支持多个末端执行器或计算所有链路的位姿

| 功能 | 状态 | 说明 |
|------|------|------|
| 多链正向运动学 | ✅ 已实现 | 计算所有链路或指定链路的位姿 |
| 多链逆向运动学 | ⏳ 待实现 | 多任务协调控制 |

### 双臂运动学（Bimanual Kinematics）

**特点**：支持双臂协调运动学计算

| 功能 | 状态 | 说明 |
|------|------|------|
| 双臂正向运动学 | ✅ 已实现 | 支持独立、相对、镜像模式 |
| 双臂逆向运动学 | ✅ 已实现 | 支持多种协调约束 |
| 双臂雅可比矩阵 | ✅ 已实现 | 支持独立和相对模式 |

---

## 功能详解

### 1. 正向运动学（Forward Kinematics）

**功能**：从关节配置 `q` 计算末端执行器位姿 `T`

**公式**：
```
T = T_base @ T_1(q1) @ T_2(q2) @ ... @ T_n(qn)
```

其中 `T_i(qi)` 是第 `i` 个关节的变换矩阵。

**参数**：
- `model`: RobotModel实例
- `q`: 关节配置（数组或字典）
- `return_end`: 是否只返回末端执行器位姿（传统模式）
- `return_all_links`: 是否返回所有链路的位姿（多链模式）
- `link_names`: 指定要计算的链路名称（仅多链模式）
- `device`: PyTorch设备（仅PyTorch后端）
- `dtype`: PyTorch数据类型（仅PyTorch后端）

**使用示例**：

```python
from robocore.kinematics import forward_kinematics
from robocore.modeling import RobotModel
import numpy as np

robot = RobotModel("path/to/robot.urdf")

# 单链模式：只计算末端执行器（单个配置）
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T_end = forward_kinematics(robot, q, return_end=True)
# 返回: 4x4 位姿矩阵

# 批处理模式：多个配置并行计算
q_batch = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
])
T_batch = forward_kinematics(robot, q_batch, return_end=True)
# 返回: [3, 4, 4] 数组，包含3个配置的末端执行器位姿

# 单链模式：返回所有链路的位姿
poses = forward_kinematics(robot, q, return_end=False)
# 返回: {'base': T_base, 'link1': T_1, ..., 'end': T_end}

# 多链模式：计算所有链路的位姿
all_poses = forward_kinematics(robot, q, return_all_links=True)
# 返回: {link_name: T_link for all links in kinematic tree}

# 多链模式：计算指定链路的位姿
selected_poses = forward_kinematics(
    robot, q, 
    return_all_links=True,
    link_names=['link1', 'link3', 'end']
)
# 返回: {'link1': T_1, 'link3': T_3, 'end': T_end}
```

**适用场景**：
- 末端执行器定位
- 碰撞检测（需要所有链路位姿）
- 可视化
- 工作空间分析

---

### 2. 逆向运动学（Inverse Kinematics）

**功能**：从目标位姿 `T_target` 计算关节配置 `q`

**方法**：
- `'dls'`: Damped Least Squares（阻尼最小二乘，默认）
- `'pinv'`: Pseudo-Inverse（伪逆）
- `'transpose'`: Jacobian Transpose（雅可比转置）

**参数**：
- `model`: RobotModel实例
- `target_pose`: 目标位姿（4x4矩阵）或批处理数组 [B, 4, 4]
- `q0`: 初始关节配置（可选，用于某些初始猜测策略的基准）
- `method`: 求解方法（'dls'|'pinv'|'transpose'，默认'dls'）
- `num_initial_guesses`: 初始猜测数量（默认1，提高成功率可设为5-20）
- `initial_guess_strategy`: 初始猜测策略（'zero'|'random'|'sobol'|'latin'|'center'|'uniform'，默认'random'）
- `initial_guess_scale`: 关节限制缩放因子（0.0到1.0，默认1.0）
- `random_seed`: 随机种子（用于可重复性）
- `target_link`: 目标链路名称（部分任务）
- `row_mask`: 行掩码（部分任务，如只控制位置）
- `nullspace_gain`: 零空间增益（冗余度利用，默认0.0）
- `joint_centering`: 是否启用关节居中优化（默认True）
- `joint_center_gain`: 关节居中增益（默认0.2）
- `joint_center_weights`: 关节居中权重（可选）
- `max_iters`: 最大迭代次数（默认200，优化后提高成功率）
- `pos_tol`: 位置容差（默认1e-3，优化后平衡精度和成功率）
- `ori_tol`: 姿态容差（默认1e-3，优化后平衡精度和成功率）
- `adaptive_damping`: 是否启用自适应阻尼（默认True）
- `adaptive_step`: 是否启用自适应步长（默认True）
- `pos_weight`: 位置权重（默认1.0）
- `ori_weight`: 姿态权重（默认1.0）

**使用示例**：

```python
from robocore.kinematics import inverse_kinematics
import numpy as np

# 基本使用（单个目标位姿）
target_pose = np.array([
    [1, 0, 0, 0.5],
    [0, 1, 0, 0.3],
    [0, 0, 1, 0.2],
    [0, 0, 0, 1]
])
q0 = [0.0] * 6  # 初始配置

result = inverse_kinematics(
    robot, target_pose, q0,
    method='dls'
)
# result['success']: 是否成功
# result['q']: 求解得到的关节配置
# result['err_norm']: 误差范数

# 批处理模式：多个目标位姿并行求解（推荐方式）
target_poses = np.array([
    [[1, 0, 0, 0.5], [0, 1, 0, 0.3], [0, 0, 1, 0.2], [0, 0, 0, 1]],
    [[1, 0, 0, 0.6], [0, 1, 0, 0.4], [0, 0, 1, 0.3], [0, 0, 0, 1]],
    [[1, 0, 0, 0.4], [0, 1, 0, 0.2], [0, 0, 1, 0.1], [0, 0, 0, 1]]
])  # [3, 4, 4]

# 不传入 q0，系统会自动生成初始猜测（推荐）
results = inverse_kinematics(
    robot, target_poses,
    method='dls',
    initial_guess_strategy='random',  # 使用随机初始猜测
    random_seed=42
)
# 返回: 包含3个结果的列表
# results[0]['success'], results[0]['q'], ...

# 或者传入 q0 作为基准（用于某些策略）
results = inverse_kinematics(
    robot, target_poses, q0=[0.0] * 6,  # 作为基准配置
    method='dls',
    initial_guess_strategy='random'
)

# 多初始猜测求解（提高成功率）
result = inverse_kinematics(
    robot, target_pose, q0,
    method='dls',
    num_initial_guesses=10,  # 尝试10个不同的初始猜测
    initial_guess_strategy='sobol',  # 使用Sobol序列（需要scipy）
    initial_guess_scale=1.0,  # 使用完整关节范围
    random_seed=42  # 可重复性
)

# 部分任务：只控制位置（不控制姿态）
result = inverse_kinematics(
    robot, target_pose, q0,
    row_mask=[True, True, True, False, False, False]  # 只控制前3维（位置）
)

# 零空间优化：在满足任务的同时优化关节配置
result = inverse_kinematics(
    robot, target_pose, q0,
    nullspace_gain=0.1,  # 零空间增益
    joint_centering=True,  # 启用关节居中
    joint_center_gain=0.2  # 居中增益
)
```

**适用场景**：
- 末端执行器定位
- 轨迹规划
- 抓取规划
- 冗余机器人优化

---

### 3. 雅可比矩阵（Jacobian）

**功能**：计算速度雅可比矩阵 `J`，建立关节速度与末端速度的关系

**公式**：
```
ẋ = J(q) · q̇
```

其中：
- `ẋ`: 末端执行器速度（6维：线速度3 + 角速度3）
- `J(q)`: 雅可比矩阵（6×n）
- `q̇`: 关节速度（n维）

**方法**：
- `'analytic'`: 解析法（默认，最快最准确）
- `'numeric'`: 数值微分法（用于验证）
- `'autograd'`: 自动微分法（仅PyTorch后端）

**参数**：
- `model`: RobotModel实例
- `q`: 关节配置
- `method`: 计算方法（'analytic'|'numeric'|'autograd'）
- `target_link`: 目标链路名称
- `joint_indices`: 选择的关节索引（部分雅可比）
- `row_mask`: 行掩码（部分雅可比，如只计算位置）
- `epsilon`: 数值微分的步长（仅数值法）
- `use_central_diff`: 是否使用中心差分（仅数值法）
- `device`: PyTorch设备（仅PyTorch后端）
- `dtype`: PyTorch数据类型（仅PyTorch后端）

**使用示例**：

```python
from robocore.kinematics import jacobian
import numpy as np

# 基本使用：计算完整雅可比矩阵（单个配置）
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
J = jacobian(robot, q, method='analytic')
# 返回: 6×n 雅可比矩阵

# 批处理模式：多个配置并行计算
q_batch = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
])  # [3, 6]
J_batch = jacobian(robot, q_batch, method='analytic')
# 返回: [3, 6, 6] 数组，包含3个配置的雅可比矩阵

# 部分雅可比：只计算位置（前3行）
J_pos = jacobian(
    robot, q,
    row_mask=[True, True, True, False, False, False]
)
# 返回: 3×n 位置雅可比矩阵

# 部分雅可比：只计算前3个关节的贡献
J_partial = jacobian(
    robot, q,
    joint_indices=[0, 1, 2]
)
# 返回: 6×3 部分雅可比矩阵

# 数值法验证（用于调试）
J_numeric = jacobian(
    robot, q,
    method='numeric',
    epsilon=1e-5,
    use_central_diff=True
)
```

**适用场景**：
- 速度控制
- 奇异性分析
- 可操作性分析
- 力控制（雅可比转置）

**重要说明：坐标系约定**

RoboCore的雅可比矩阵采用以下坐标系约定（与`pytorch_kinematics`一致）：

- **位置雅可比（前3行）**：在世界坐标系（基坐标系）中表达
- **角速度雅可比（后3行）**：在世界坐标系（基坐标系）中表达

这种约定符合标准机器人学实践，其中线速度和角速度都在基坐标系中表达，便于与速度控制、力控制等应用直接对接。

---

### 4. 双臂正向运动学

**功能**：计算双臂系统的末端执行器位姿

**模式**：
- `'indep'`: 独立模式（两臂独立计算）
- `'relative'`: 相对模式（计算相对位姿）
- `'mirror'`: 镜像模式（镜像对称）

**使用示例**：

```python
from robocore.kinematics import bimanual_forward_kinematics

# 独立模式
result = bimanual_forward_kinematics(
    left_model, right_model,
    q_left, q_right,
    mode='indep'
)
# 返回: {'left': T_left, 'right': T_right}

# 相对模式
result = bimanual_forward_kinematics(
    left_model, right_model,
    q_left, q_right,
    mode='relative'
)
# 返回: {'left': T_left, 'right': T_right, 'relative': T_rel}

# 镜像模式
result = bimanual_forward_kinematics(
    left_model, right_model,
    q_left, q_right,
    mode='mirror'
)
# 返回: {'left': T_left, 'right': T_right, 'mirror': T_mirror}
```

---

### 5. 双臂逆向运动学

**功能**：求解双臂系统的关节配置

**协调模式**：
- `'indep'`: 独立模式（两臂独立求解）
- `'relative_pose'`: 相对位姿约束（完整6D约束）
- `'relative_pos'`: 相对位置约束（仅3D位置）
- `'relative_ori'`: 相对姿态约束（仅3D姿态）
- `'mirror'`: 镜像模式（镜像对称）

**使用示例**：

```python
from robocore.kinematics import bimanual_inverse_kinematics

# 独立模式
result = bimanual_inverse_kinematics(
    left_model, right_model,
    target_left=T_left_target,
    target_right=T_right_target,
    q0_left=q0_left,
    q0_right=q0_right,
    coordination='indep',
    method='dls'
)

# 相对位姿约束（如双手抓取）
result = bimanual_inverse_kinematics(
    left_model, right_model,
    target_left=T_left_target,  # 主目标
    q0_left=q0_left,
    q0_right=q0_right,
    coordination='relative_pose',
    T_rel_grasp=T_relative  # 期望的相对变换
)
```

---

### 6. 双臂雅可比矩阵

**功能**：计算双臂系统的雅可比矩阵

**模式**：
- `'indep'`: 独立模式（12×n，两臂独立）
- `'relative'`: 相对模式（6×n，相对位姿雅可比）

**使用示例**：

```python
from robocore.kinematics import bimanual_jacobian

# 独立模式
J = bimanual_jacobian(
    left_model, right_model,
    q_left, q_right,
    mode='indep'
)
# 返回: 12×(nL+nR) 雅可比矩阵

# 相对模式
J_rel = bimanual_jacobian(
    left_model, right_model,
    q_left, q_right,
    mode='relative'
)
# 返回: 6×(nL+nR) 相对雅可比矩阵
```

---

## 功能对比

### 正向运动学 vs 逆向运动学

| 特性 | 正向运动学（FK） | 逆向运动学（IK） |
|------|-----------------|-----------------|
| **输入** | 关节配置 `q` | 目标位姿 `T` |
| **输出** | 末端位姿 `T` | 关节配置 `q` |
| **唯一性** | 唯一解 | 多解（可能无解） |
| **计算复杂度** | O(n) | O(n²) 迭代 |
| **适用场景** | 定位、可视化 | 轨迹规划、抓取 |
| **奇异性** | 无 | 存在奇异性问题 |

### 雅可比计算方法对比

| 方法 | 精度 | 速度 | 适用场景 | 后端支持 |
|------|------|------|----------|----------|
| **解析法（analytic）** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 生产环境（默认） | NumPy, PyTorch |
| **数值法（numeric）** | ⭐⭐⭐ | ⚡⚡ | 验证、调试 | NumPy, PyTorch |
| **自动微分（autograd）** | ⭐⭐⭐⭐⭐ | ⚡⚡ | 梯度优化 | PyTorch only |

### IK求解方法对比

| 方法 | 公式 | 特点 | 适用场景 |
|------|------|------|----------|
| **DLS（阻尼最小二乘）** | `dq = J^T (JJ^T + λ²I)^(-1) e` | 稳定，抗奇异性，自适应阻尼 | 通用（默认，推荐） |
| **伪逆（Pseudo-Inverse）** | `dq = V @ diag(S/(S²+λ²)) @ U^T @ e` | 快速，自适应阻尼，SVD分解 | 非奇异区域 |
| **转置（Transpose）** | `dq = α J^T @ e` | 最简单，自适应增益 | 教学、简单任务 |

**自适应特性**：
- **自适应阻尼**：根据雅可比条件数和误差大小动态调整阻尼（默认启用）
- **自适应步长**：根据误差大小动态调整步长（默认启用）
- **Plateau检测**：检测收敛停滞并调整参数
- **精化阶段**：收敛后进行精化迭代以提高精度（可选）

### 双臂协调模式对比

| 模式 | 约束类型 | 自由度 | 适用场景 |
|------|----------|--------|----------|
| **独立（indep）** | 无约束 | 最大 | 独立任务 |
| **相对位姿（relative_pose）** | 6D约束 | 减少6 | 双手抓取 |
| **相对位置（relative_pos）** | 3D约束 | 减少3 | 位置协调 |
| **相对姿态（relative_ori）** | 3D约束 | 减少3 | 姿态协调 |
| **镜像（mirror）** | 对称约束 | 减少6 | 对称操作 |

---

## 双臂运动学

### 协调模式详解

#### 1. 独立模式（Independent Mode）

**特点**：两臂完全独立，无约束

**应用场景**：
- 两臂执行不同任务
- 独立轨迹跟踪
- 无协调要求的场景

#### 2. 相对位姿模式（Relative Pose Mode）

**特点**：约束两臂末端执行器的相对位姿

**公式**：
```
T_rel = T_left^(-1) @ T_right = T_rel_desired
```

**应用场景**：
- 双手抓取物体
- 双手协作操作
- 保持相对位姿不变

#### 3. 镜像模式（Mirror Mode）

**特点**：两臂镜像对称运动

**应用场景**：
- 对称操作
- 双手对称轨迹
- 美学要求

---

## 任务抽象

### Task类

`Task`类提供统一的任务表示，支持多任务协调控制：

**任务类型**：
- `'absolute'`: 绝对位姿跟踪（末端执行器到目标位姿）
- `'relative'`: 相对位姿约束（两个组之间的相对变换）
- `'centering'`: 关节居中（优化关节配置）
- `'contact'`: 接触约束（保持接触点静止）

**使用示例**：

```python
from robocore.kinematics.task import (
    absolute_task,
    relative_task,
    centering_task,
    contact_task
)

# 绝对任务：末端执行器到目标位姿
task1 = absolute_task(
    group='left_arm',
    target=T_target,
    weight=1.0,
    priority=0
)

# 相对任务：保持两臂相对位姿
task2 = relative_task(
    group_a='left_arm',
    group_b='right_arm',
    target=T_relative,
    weight=0.5,
    priority=1
)

# 居中任务：优化关节配置
task3 = centering_task(
    group='left_arm',
    weight=0.1,
    priority=2
)

# 接触任务：保持接触点静止
task4 = contact_task(
    group='right_arm',
    weight=1.0,
    priority=0
)
```

---

## 使用指南

### 基本使用流程

1. **导入模块**
```python
from robocore.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian
)
from robocore.modeling import RobotModel
import numpy as np
```

2. **创建机器人模型**
```python
robot = RobotModel("path/to/robot.urdf")
```

3. **正向运动学**
```python
# 单个配置
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T = forward_kinematics(robot, q, return_end=True)

# 批处理（自动检测）
q_batch = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
T_batch = forward_kinematics(robot, q_batch, return_end=True)  # [2, 4, 4]
```

4. **逆向运动学**
```python
# 单个目标
target_pose = np.array([...])  # 4x4矩阵
q0 = [0.0] * 6
result = inverse_kinematics(robot, target_pose, q0)
if result['success']:
    q_solution = result['q']

# 批处理（推荐：不传入 q0，系统自动生成初始猜测）
target_poses = np.array([...])  # [B, 4, 4]
results = inverse_kinematics(
    robot, target_poses,
    initial_guess_strategy='random',
    random_seed=42
)  # 返回列表
```

5. **雅可比矩阵**
```python
# 单个配置
J = jacobian(robot, q, method='analytic')

# 批处理
q_batch = np.array([...])  # [B, n]
J_batch = jacobian(robot, q_batch, method='analytic')  # [B, 6, n]
```

### 批处理功能

所有运动学函数（`forward_kinematics`、`inverse_kinematics`、`jacobian`）都支持批处理模式：

- **自动检测**：输入为 1D 数组时自动包装为 batch=1，2D/3D 数组时直接批处理
- **统一接口**：单个配置和批处理使用相同的函数调用
- **真正的并行**：两个后端都实现了真正的批量并行处理
- **向后兼容**：现有单配置代码无需修改即可工作

**批处理性能优势**：

| 功能 | NumPy 后端 | PyTorch 后端 (CPU) | PyTorch 后端 (GPU) |
|------|-----------|-------------------|-------------------|
| **Forward Kinematics** | ✅ 批量并行 | ✅ 批量并行 | ✅ 批量并行 + GPU加速 |
| **Jacobian** | ✅ 批量并行 | ✅ 批量并行 (~30x) | ✅ 批量并行 + GPU加速 |
| **Inverse Kinematics** | ✅ 批量并行 | ✅ 批量并行 | ✅ 批量并行 + GPU加速 |

**批处理并行实现详解**：

1. **Forward Kinematics (FK)**
   - ✅ **完全向量化**：所有样本的变换矩阵同时计算
   - ✅ **无循环**：真正的批量并行

2. **Jacobian 计算**
   - ✅ **完全向量化**：所有样本的雅可比矩阵同时计算
   - ✅ **无循环**：真正的批量并行
   - 🚀 **Torch 优势**：在 CPU 上比 NumPy 快 ~30x

3. **Inverse Kinematics (IK)**
   - ✅ **批量并行部分**：
     - FK 计算：批量并行
     - Jacobian 计算：批量并行
     - 误差计算：向量化计算
   - ⚠️ **循环处理部分**：
     - 自适应阻尼计算：每个样本的条件数不同，需要循环
     - DLS/Pinv 求解：每个样本的 damping 不同，需要循环
   - 💡 **设计权衡**：自适应阻尼显著提高成功率（~89-94%），但需要循环处理

**性能表现**（Alicia-D 10 DOF，测试配置）：
- **NumPy batch IK** (n=100): ~4.5ms/sample，成功率 ~89-94%
- **Torch batch IK (CPU)** (n=100): ~10-11ms/sample，成功率 ~81-90%
- **Torch batch IK (GPU)**: 预期显著加速（推荐用于大规模批处理，n>1000）

**性能建议**：
- **小规模批处理** (n<50): NumPy 通常更快
- **中等规模** (50<n<500): 根据硬件选择，CPU 上 NumPy 更快
- **大规模批处理** (n>500): 推荐使用 Torch + GPU，优势明显

### 高级功能

#### 多初始猜测IK求解

```python
# 提高IK成功率 - 使用多种初始猜测策略
result = inverse_kinematics(
    robot, target_pose, q0,
    num_initial_guesses=20,  # 尝试20个初始猜测
    initial_guess_strategy='sobol',  # 使用Sobol序列（均匀分布）
    initial_guess_scale=1.0,  # 使用完整关节范围
    random_seed=42
)

# 其他初始猜测策略：
# - 'random': 随机采样（默认，快速）
# - 'sobol': Sobol序列（需要scipy，均匀覆盖）
# - 'latin': 拉丁超立方采样（需要scipy，均匀覆盖）
# - 'uniform': 均匀随机采样
# - 'center': 关节范围中心
# - 'zero': 零配置
```

#### 部分任务控制

```python
# 只控制位置，不控制姿态
result = inverse_kinematics(
    robot, target_pose, q0,
    row_mask=[True, True, True, False, False, False]
)

# 只控制前3个关节
J_partial = jacobian(
    robot, q,
    joint_indices=[0, 1, 2]
)
```

#### 零空间优化

```python
# 在满足任务的同时优化关节配置
result = inverse_kinematics(
    robot, target_pose, q0,
    nullspace_gain=0.1,
    joint_centering=True,
    joint_center_gain=0.2
)
```

### 参数调优指南

#### IK求解参数

**容差设置**：
- `pos_tol`: 位置容差，默认1e-3（优化后平衡精度和成功率）
- `ori_tol`: 姿态容差，默认1e-3（优化后平衡精度和成功率）
- 容差越小，精度越高，但可能降低成功率
- 对于高精度需求，可设为1e-4，但成功率可能下降

**迭代次数**：
- `max_iters`: 最大迭代次数，默认200（优化后提高成功率）
- 对于复杂机器人或困难目标，可以增加到300-500

**初始猜测参数**：
- `num_initial_guesses`: 初始猜测数量，默认1，通常设为5-20可显著提高成功率
- `initial_guess_strategy`: 初始猜测策略
  - `'random'`: 随机采样（默认，快速）
  - `'sobol'`: Sobol序列（需要scipy，均匀覆盖，推荐用于多猜测）
  - `'latin'`: 拉丁超立方采样（需要scipy，均匀覆盖）
  - `'uniform'`: 均匀随机采样
  - `'center'`: 关节范围中心
  - `'zero'`: 零配置
- `initial_guess_scale`: 缩放因子（0.0-1.0），控制初始猜测的范围

**零空间参数**：
- `nullspace_gain`: 零空间增益，通常设为0.01-0.1
- `joint_center_gain`: 关节居中增益，通常设为0.1-0.5
- `joint_centering`: 是否启用关节居中（默认True）
- 增益太大可能影响主任务，太小可能效果不明显

**自适应参数**：
- `adaptive_damping`: 是否启用自适应阻尼（默认True，推荐）
- `adaptive_step`: 是否启用自适应步长（默认True，推荐）
- `pos_weight`: 位置权重（默认1.0）
- `ori_weight`: 姿态权重（默认1.0）
- 自适应机制根据雅可比条件数和误差大小动态调整参数，提高收敛稳定性

---

## 实现状态

### Phase 1: 基础运动学（已完成 ✅）

- [x] 正向运动学（单链）
- [x] 逆向运动学（单链）
- [x] 雅可比矩阵计算（解析法、数值法）
- [x] NumPy后端支持
- [x] PyTorch后端支持

### Phase 2: 多链运动学（已完成 ✅）

- [x] 多链正向运动学（所有链路）
- [x] 多链正向运动学（指定链路）
- [ ] 多链逆向运动学（多任务协调）

### Phase 3: 双臂运动学（已完成 ✅）

- [x] 双臂正向运动学（独立、相对、镜像模式）
- [x] 双臂逆向运动学（独立、相对位姿、相对位置、相对姿态、镜像模式）
- [x] 双臂雅可比矩阵（独立、相对模式）

### Phase 4: 高级功能（已完成 ✅）

- [x] 部分任务控制（row_mask, joint_indices）
- [x] 零空间优化（nullspace_gain, joint_centering）
- [x] 统一的初始猜测系统（num_initial_guesses, initial_guess_strategy）
- [x] 自适应阻尼和自适应步长（adaptive_damping, adaptive_step）
- [x] 位置/姿态权重（pos_weight, ori_weight）
- [x] 精化阶段（refine）
- [x] 任务抽象（Task类）
- [ ] 多任务协调求解器
- [ ] 层次化任务求解

### Phase 5: 优化与扩展（部分完成 ⏳）

- [x] 批处理性能优化（NumPy 和 Torch 都实现真正的批量并行）
- [x] 自适应参数优化（默认参数优化，成功率提升）
- [x] 公式一致性验证（NumPy 和 Torch 完全一致）
- [ ] 解析IK（针对特定机器人）
- [ ] IK种子生成（基于工作空间分析）
- [ ] 奇异性检测与处理
- [ ] 可操作性分析
- [ ] 工作空间分析工具

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Introduction to Robotics** (Craig) - 经典教材

### 参考库

- **Pinocchio** - 高性能运动学和动力学库
- **Drake** - MIT机器人工具箱
- **PyBullet** - 物理仿真库（运动学部分）

---

---

## 最新更新（2025-01）

### IK 求解器重大更新

#### 1. 统一的初始猜测系统 ✅

**改进**：替换了旧的 `multi_start`/`multi_noise` 参数系统

**新参数**：
- `num_initial_guesses`: 初始猜测数量（默认1）
- `initial_guess_strategy`: 初始猜测策略
  - `'zero'`: 零配置
  - `'random'`: 随机采样（默认）
  - `'sobol'`: Sobol序列（需要scipy，均匀覆盖）
  - `'latin'`: 拉丁超立方采样（需要scipy）
  - `'center'`: 关节范围中心
  - `'uniform'`: 均匀随机采样
- `initial_guess_scale`: 关节限制缩放因子（0.0-1.0）

**优势**：
- 初始猜测生成与求解逻辑完全解耦
- 支持更多采样策略，提高成功率
- 批处理模式下自动为每个样本生成初始猜测

#### 2. 批处理性能优化 ✅

**NumPy 后端**：
- ✅ 修复了 batch 模式，现在使用真正的批量并行处理
- ✅ 使用 `_solve_batch` 方法，FK 和 Jacobian 都是批量计算
- ✅ 性能提升：从循环处理改为批量处理，速度提升 ~3x

**Torch 后端**：
- ✅ 优化了结果转换，批量转换 tensor 到 numpy（避免循环中的 `.item()` 调用）
- ✅ 优化了 `eye6` 和 `lam` 的处理，减少重复开销
- ✅ 两个后端都实现了真正的批量并行（FK、Jacobian、误差计算）

#### 3. 自适应逻辑完善 ✅

**Torch 后端**：
- ✅ 实现了完整的自适应逻辑（与 NumPy 完全一致）
- ✅ 包括：自适应阻尼、自适应步长、plateau 检测、精化阶段
- ✅ 支持所有高级功能：位置/姿态权重、行掩码、零空间优化

**功能对等**：
- ✅ NumPy 和 Torch 后端的自适应逻辑完全一致
- ✅ 两个后端的成功率接近（NumPy ~89-94%, Torch ~81-90%）

#### 4. 默认参数优化 ✅

**参数调整**：
- `max_iters`: 100 → **200**（提高成功率）
- `pos_tol`: 1e-4 → **1e-3**（平衡精度和成功率）
- `ori_tol`: 1e-4 → **1e-3**（平衡精度和成功率）

**优化结果**（Alicia-D 10 DOF，n=1000）：
- NumPy: **89.1%** 成功率
- Torch: **81.7%** 成功率

#### 5. 公式一致性验证 ✅

**验证内容**：
- ✅ DLS 公式：`dq = J^T (JJ^T + λ²I)^(-1) e` - 两个后端完全一致
- ✅ Pinv 公式：`dq = V @ diag(S/(S²+λ²)) @ U^T @ e` - 两个后端完全一致
- ✅ 阻尼参数：都使用 `λ²`（平方）
- ✅ Jacobian 权重应用：修复了 NumPy batch 版本，现在两个后端一致

#### 6. 性能表现

**批处理性能**（Alicia-D 10 DOF）：
- **NumPy batch IK**: ~4.5ms/sample (n=50-100)，成功率 ~89-94%
- **Torch batch IK (CPU)**: ~10-11ms/sample (n=50-100)，成功率 ~81-90%
- **Torch batch IK (GPU)**: 预期显著加速（推荐用于大规模批处理）

**性能建议**：
- 小规模批处理 (n<50): NumPy 通常更快
- 中等规模 (50<n<500): 根据硬件选择
- 大规模批处理 (n>500): 推荐使用 Torch + GPU

#### 7. 代码质量改进

- ✅ 精简了实现，移除了冗余代码
- ✅ 统一了接口，两个后端行为一致
- ✅ 优化了性能瓶颈（结果转换、循环优化）
- ✅ 完善了文档和注释

---

**文档版本**: 1.1  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

