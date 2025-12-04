# RoboCore 变换模块文档

## 目录

1. [概述](#概述)
2. [模块架构](#模块架构)
3. [功能分类](#功能分类)
4. [功能详解](#功能详解)
5. [表示方法对比](#表示方法对比)
6. [使用指南](#使用指南)
7. [实现状态](#实现状态)

---

## 概述

RoboCore变换模块提供完整的3D变换计算功能，支持旋转（SO(3)）和刚体变换（SE(3)）的各种表示方法及其相互转换。

### 设计原则

- **统一接口**：提供统一的API，支持NumPy和PyTorch后端
- **双后端支持**：支持NumPy（CPU）和PyTorch（CPU/GPU）后端
- **批量处理**：支持单值和批量计算
- **数值稳定**：采用数值稳定的算法实现
- **完整转换**：支持所有常见旋转和变换表示方法之间的转换

---

## 模块架构

### 目录结构

```
robocore/transform/
├── __init__.py                    # 模块导出
├── se3.py                         # SE(3) 刚体变换操作（4×4矩阵）
├── so3.py                         # SO(3) 旋转操作（3×3矩阵）
├── conversions.py                 # 旋转表示转换函数
└── utils.py                       # 高级工具函数
```

### 统一接口

所有变换功能通过统一的高级接口访问：

```python
from robocore.transform import (
    # SE(3) 变换
    make_transform, translation_transform, rotation_transform,
    get_rotation, get_translation,
    transform_multiply, transform_inverse, transform_apply, transform_interpolate,
    # SO(3) 旋转
    rpy_to_matrix, axis_angle_to_matrix, quaternion_to_matrix, euler_to_matrix,
    rotation_x, rotation_y, rotation_z,
    rotation_multiply, rotation_inverse, rotation_apply,
    skew_symmetric, rotation_from_vectors,
    # 转换函数
    matrix_to_rpy, matrix_to_quaternion, matrix_to_axis_angle, matrix_to_euler,
    quaternion_normalize, quaternion_conjugate, quaternion_inverse, quaternion_multiply,
    rpy_to_quaternion, rpy_to_axis_angle,
    axis_angle_to_quaternion, axis_angle_to_rpy,
    # 工具函数
    slerp, rotation_interpolate,
    rotation_distance, rotation_error, orientation_error,
    is_rotation_matrix, is_transform_matrix, look_at,
)
```

---

## 功能分类

### SE(3) 刚体变换（4×4齐次变换矩阵）

**特点**：表示位置和姿态的完整变换

| 功能 | 状态 | 说明 |
|------|------|------|
| 创建变换 | ✅ 已实现 | 从旋转和平移创建变换矩阵 |
| 提取旋转/平移 | ✅ 已实现 | 从变换矩阵提取旋转和平移 |
| 变换组合 | ✅ 已实现 | 变换矩阵乘法 |
| 变换求逆 | ✅ 已实现 | 计算变换矩阵的逆 |
| 变换应用 | ✅ 已实现 | 将变换应用到点 |
| 变换插值 | ✅ 已实现 | 使用SLERP进行旋转插值 |

### SO(3) 旋转（3×3旋转矩阵）

**特点**：表示3D空间中的旋转

| 功能 | 状态 | 说明 |
|------|------|------|
| 基础旋转 | ✅ 已实现 | 绕X/Y/Z轴的旋转 |
| 旋转组合 | ✅ 已实现 | 旋转矩阵乘法 |
| 旋转求逆 | ✅ 已实现 | 旋转矩阵转置 |
| 旋转应用 | ✅ 已实现 | 将旋转应用到向量 |
| 从向量构造 | ✅ 已实现 | 从两个向量构造旋转矩阵 |

### 旋转表示转换

**特点**：支持多种旋转表示方法之间的转换

| 表示方法 | 状态 | 说明 |
|----------|------|------|
| 旋转矩阵 | ✅ 已实现 | 3×3正交矩阵 |
| RPY（Roll-Pitch-Yaw） | ✅ 已实现 | ZYX欧拉角（机器人学约定） |
| 四元数 | ✅ 已实现 | [x, y, z, w] 格式 |
| 轴角 | ✅ 已实现 | 轴向量 + 角度 |
| 紧凑轴角 | ✅ 已实现 | 轴向量 × 角度 |
| 欧拉角 | ✅ 已实现 | 支持多种序列（xyz, zyx等） |

### 工具函数

| 功能 | 状态 | 说明 |
|------|------|------|
| SLERP插值 | ✅ 已实现 | 球面线性插值 |
| 旋转距离 | ✅ 已实现 | 计算两个旋转之间的角度距离 |
| 旋转误差 | ✅ 已实现 | 计算旋转误差向量 |
| 矩阵验证 | ✅ 已实现 | 验证旋转矩阵和变换矩阵的有效性 |
| Look-At矩阵 | ✅ 已实现 | 创建相机视图矩阵 |

---

## 功能详解

### 1. SE(3) 刚体变换

#### 创建变换矩阵

**函数**：`make_transform(R, t)`

**公式**：
```
T = [R  t]
    [0  1]
```

其中：
- `R`: 3×3旋转矩阵
- `t`: 3×1平移向量

**使用示例**：

```python
from robocore.transform import make_transform, rpy_to_matrix
import numpy as np

# 创建旋转和平移
R = rpy_to_matrix(0.1, 0.2, 0.3)
t = np.array([1.0, 2.0, 3.0])

# 创建变换矩阵
T = make_transform(R, t)
# 返回: 4×4 变换矩阵
```

#### 变换组合

**函数**：`transform_multiply(T1, T2)`

**公式**：`T = T1 @ T2`

**物理意义**：先应用T2，再应用T1

**使用示例**：

```python
from robocore.transform import transform_multiply

T1 = make_transform(R1, t1)
T2 = make_transform(R2, t2)

# 组合变换
T_composed = transform_multiply(T1, T2)
```

#### 变换求逆

**函数**：`transform_inverse(T)`

**公式**：
```
T^(-1) = [R^T  -R^T @ t]
         [0    1        ]
```

**使用示例**：

```python
from robocore.transform import transform_inverse

T_inv = transform_inverse(T)
# 验证: T @ T_inv = I
```

#### 变换应用

**函数**：`transform_apply(T, points)`

**公式**：`p' = R @ p + t`

**使用示例**：

```python
from robocore.transform import transform_apply

# 单个点
point = np.array([1.0, 0.0, 0.0])
point_transformed = transform_apply(T, point)

# 多个点
points = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
points_transformed = transform_apply(T, points)
```

#### 变换插值

**函数**：`transform_interpolate(T1, T2, t)`

**方法**：
- 旋转：使用SLERP（球面线性插值）
- 平移：使用线性插值

**使用示例**：

```python
from robocore.transform import transform_interpolate

T_start = make_transform(R1, t1)
T_end = make_transform(R2, t2)

# 在t=0.5处插值
T_mid = transform_interpolate(T_start, T_end, 0.5)
```

---

### 2. SO(3) 旋转

#### 基础旋转

**函数**：`rotation_x(theta)`, `rotation_y(theta)`, `rotation_z(theta)`

**使用示例**：

```python
from robocore.transform import rotation_x, rotation_y, rotation_z
import numpy as np

# 绕X轴旋转45度
Rx = rotation_x(np.pi / 4)

# 绕Y轴旋转30度
Ry = rotation_y(np.pi / 6)

# 绕Z轴旋转60度
Rz = rotation_z(np.pi / 3)
```

#### RPY到旋转矩阵

**函数**：`rpy_to_matrix(roll, pitch, yaw)`

**公式**：`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`

**约定**：机器人学约定（ZYX欧拉角）

**使用示例**：

```python
from robocore.transform import rpy_to_matrix

roll, pitch, yaw = 0.1, 0.2, 0.3
R = rpy_to_matrix(roll, pitch, yaw)
```

#### 旋转组合

**函数**：`rotation_multiply(R1, R2)`

**公式**：`R = R1 @ R2`

**使用示例**：

```python
from robocore.transform import rotation_multiply

R = rotation_multiply(Rz, rotation_multiply(Ry, Rx))
```

#### 旋转应用

**函数**：`rotation_apply(R, vectors)`

**公式**：`v' = R @ v`

**使用示例**：

```python
from robocore.transform import rotation_apply

v = np.array([1.0, 0.0, 0.0])
v_rotated = rotation_apply(R, v)
```

---

### 3. 旋转表示转换

#### 旋转矩阵 ↔ RPY

**函数**：
- `rpy_to_matrix(roll, pitch, yaw)`: RPY → 旋转矩阵
- `matrix_to_rpy(R)`: 旋转矩阵 → RPY

**使用示例**：

```python
from robocore.transform import rpy_to_matrix, matrix_to_rpy

# RPY → 旋转矩阵
roll, pitch, yaw = 0.1, 0.2, 0.3
R = rpy_to_matrix(roll, pitch, yaw)

# 旋转矩阵 → RPY
rpy = matrix_to_rpy(R)
```

#### 旋转矩阵 ↔ 四元数

**函数**：
- `quaternion_to_matrix(q)`: 四元数 → 旋转矩阵
- `matrix_to_quaternion(R)`: 旋转矩阵 → 四元数

**四元数格式**：`[x, y, z, w]`

**使用示例**：

```python
from robocore.transform import (
    quaternion_to_matrix, matrix_to_quaternion,
    rpy_to_quaternion
)

# RPY → 四元数
q = rpy_to_quaternion(0.1, 0.2, 0.3)

# 四元数 → 旋转矩阵
R = quaternion_to_matrix(q)

# 旋转矩阵 → 四元数
q_back = matrix_to_quaternion(R)
```

#### 旋转矩阵 ↔ 轴角

**函数**：
- `axis_angle_to_matrix(axis, angle)`: 轴角 → 旋转矩阵
- `matrix_to_axis_angle(R)`: 旋转矩阵 → 轴角

**使用示例**：

```python
from robocore.transform import (
    axis_angle_to_matrix, matrix_to_axis_angle
)

# 轴角 → 旋转矩阵
axis = np.array([0.0, 0.0, 1.0])  # Z轴
angle = np.pi / 4  # 45度
R = axis_angle_to_matrix(axis, angle)

# 旋转矩阵 → 轴角
axis, angle = matrix_to_axis_angle(R)
```

#### 旋转矩阵 ↔ 欧拉角

**函数**：
- `euler_to_matrix(alpha, beta, gamma, seq='xyz')`: 欧拉角 → 旋转矩阵
- `matrix_to_euler(R, seq='xyz')`: 旋转矩阵 → 欧拉角

**支持的序列**：
- 内旋（小写）：`xyz`, `zyx`, `xzy`, `yxz`, `yzx`, `zxy`
- 外旋（大写）：`XYZ`, `ZYX`, `XZY`, `YXZ`, `YZX`, `ZXY`

**使用示例**：

```python
from robocore.transform import euler_to_matrix, matrix_to_euler

# 内旋（小写）
R = euler_to_matrix(0.1, 0.2, 0.3, seq='xyz')
euler = matrix_to_euler(R, seq='xyz')

# 外旋（大写）
R = euler_to_matrix(0.1, 0.2, 0.3, seq='XYZ')
euler = matrix_to_euler(R, seq='XYZ')
```

#### 四元数操作

**函数**：
- `quaternion_normalize(q)`: 归一化四元数
- `quaternion_conjugate(q)`: 四元数共轭
- `quaternion_inverse(q)`: 四元数逆
- `quaternion_multiply(q1, q2)`: 四元数乘法

**使用示例**：

```python
from robocore.transform import (
    quaternion_normalize, quaternion_conjugate,
    quaternion_inverse, quaternion_multiply
)

q1 = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
q2 = rpy_to_quaternion(0.0, 0.0, np.pi/2)

# 归一化
q_norm = quaternion_normalize(q2)

# 共轭
q_conj = quaternion_conjugate(q2)

# 逆（单位四元数的逆等于共轭）
q_inv = quaternion_inverse(q2)

# 乘法（组合旋转）
q_composed = quaternion_multiply(q1, q2)
```

---

### 4. 工具函数

#### SLERP插值

**函数**：`slerp(q1, q2, t)`

**功能**：球面线性插值，用于平滑旋转插值

**使用示例**：

```python
from robocore.transform import slerp, rpy_to_quaternion

q_start = np.array([0.0, 0.0, 0.0, 1.0])
q_end = rpy_to_quaternion(0.0, 0.0, np.pi/2)

# 在t=0.5处插值
q_mid = slerp(q_start, q_end, 0.5)
```

#### 旋转距离

**函数**：`rotation_distance(R1, R2)`

**功能**：计算两个旋转之间的角度距离（弧度）

**使用示例**：

```python
from robocore.transform import rotation_distance, rotation_z
import numpy as np

R1 = rotation_z(0.0)
R2 = rotation_z(np.pi / 4)

dist = rotation_distance(R1, R2)
print(f"角度距离: {np.degrees(dist):.2f}度")
```

#### 旋转误差

**函数**：`rotation_error(R_current, R_target)`

**功能**：计算当前旋转到目标旋转的误差向量（轴角紧凑形式）

**使用示例**：

```python
from robocore.transform import rotation_error

R_current = rotation_z(0.1)
R_target = rotation_z(0.5)

error = rotation_error(R_current, R_target)
# 返回: 3维误差向量（轴 × 角度）
```

#### 矩阵验证

**函数**：
- `is_rotation_matrix(R, tol=1e-6)`: 验证旋转矩阵
- `is_transform_matrix(T, tol=1e-6)`: 验证变换矩阵

**使用示例**：

```python
from robocore.transform import is_rotation_matrix, is_transform_matrix

# 验证旋转矩阵
R = rpy_to_matrix(0.1, 0.2, 0.3)
is_valid = is_rotation_matrix(R)

# 验证变换矩阵
T = make_transform(R, t)
is_valid = is_transform_matrix(T)
```

#### Look-At矩阵

**函数**：`look_at(eye, target, up)`

**功能**：创建相机视图矩阵（从eye看向target）

**使用示例**：

```python
from robocore.transform import look_at

eye = np.array([3.0, 4.0, 5.0])
target = np.array([0.0, 0.0, 0.0])
up = np.array([0.0, 0.0, 1.0])

T = look_at(eye, target, up)
```

---

## 表示方法对比

### 旋转表示方法对比

| 表示方法 | 参数数量 | 优点 | 缺点 | 适用场景 |
|---------|---------|------|------|----------|
| **旋转矩阵** | 9 (3×3) | 无奇异性，计算高效 | 冗余（6个约束），存储空间大 | 通用计算 |
| **RPY** | 3 | 直观，易于理解 | 万向锁问题 | 机器人学，简单控制 |
| **四元数** | 4 | 无奇异性，插值平滑 | 不直观，需要归一化 | 动画，SLERP插值 |
| **轴角** | 4 (轴3 + 角度1) | 直观，无奇异性 | 表示不唯一 | 误差表示，控制 |
| **紧凑轴角** | 3 | 紧凑，无奇异性 | 角度为0时轴未定义 | 误差表示，优化 |
| **欧拉角** | 3 | 直观，多种约定 | 万向锁问题 | 特定应用场景 |

### 变换表示方法对比

| 表示方法 | 参数数量 | 优点 | 缺点 | 适用场景 |
|---------|---------|------|------|----------|
| **齐次变换矩阵** | 16 (4×4) | 统一表示，计算高效 | 冗余，存储空间大 | 通用计算 |
| **旋转+平移** | 12 (R 9 + t 3) | 分离表示，直观 | 需要分别处理 | 简单应用 |
| **四元数+平移** | 7 (q 4 + t 3) | 紧凑，无奇异性 | 需要归一化 | 优化，插值 |

### 选择指南

**按应用场景选择**：
- **通用计算** → 旋转矩阵 / 齐次变换矩阵
- **机器人控制** → RPY / 轴角
- **动画/插值** → 四元数
- **优化问题** → 紧凑轴角 / 四元数+平移
- **误差表示** → 轴角 / 紧凑轴角

**按数值稳定性选择**：
- **避免万向锁** → 四元数 / 轴角
- **需要插值** → 四元数（SLERP）
- **简单计算** → 旋转矩阵

---

## 使用指南

### 基本使用流程

1. **导入模块**
```python
from robocore.transform import (
    rpy_to_matrix, make_transform,
    transform_multiply, transform_apply
)
import numpy as np
```

2. **创建变换**
```python
# 从RPY创建旋转
R = rpy_to_matrix(0.1, 0.2, 0.3)

# 创建变换矩阵
t = np.array([1.0, 2.0, 3.0])
T = make_transform(R, t)
```

3. **组合变换**
```python
T1 = make_transform(R1, t1)
T2 = make_transform(R2, t2)
T_composed = transform_multiply(T1, T2)
```

4. **应用变换**
```python
point = np.array([1.0, 0.0, 0.0])
point_transformed = transform_apply(T, point)
```

### 批量处理

所有函数都支持批量处理：

```python
# 批量RPY
rolls = np.array([0.1, 0.2, 0.3])
pitches = np.array([0.2, 0.3, 0.4])
yaws = np.array([0.3, 0.4, 0.5])

# 批量创建旋转矩阵
R_batch = rpy_to_matrix(rolls, pitches, yaws)
# 返回: (3, 3, 3) 形状

# 批量转换
q_batch = matrix_to_quaternion(R_batch)
# 返回: (3, 4) 形状
```

### 后端切换

```python
from robocore.utils.backend import set_backend

# 使用NumPy后端
set_backend('numpy')
R = rpy_to_matrix(0.1, 0.2, 0.3)

# 切换到PyTorch后端
set_backend('torch', device='cpu')
R = rpy_to_matrix(0.1, 0.2, 0.3)

# 使用GPU（如果可用）
set_backend('torch', device='cuda:0')
R = rpy_to_matrix(0.1, 0.2, 0.3)
```

### 常见模式

#### 模式1：RPY → 变换矩阵 → 应用

```python
from robocore.transform import rpy_to_matrix, make_transform, transform_apply

# 创建变换
R = rpy_to_matrix(roll, pitch, yaw)
T = make_transform(R, translation)

# 应用变换
points_transformed = transform_apply(T, points)
```

#### 模式2：四元数插值

```python
from robocore.transform import slerp, quaternion_to_matrix

# 插值
q_interp = slerp(q_start, q_end, t)

# 转换为旋转矩阵
R_interp = quaternion_to_matrix(q_interp)
```

#### 模式3：旋转误差计算

```python
from robocore.transform import rotation_error

# 计算误差
error = rotation_error(R_current, R_target)

# 误差向量可用于控制
# error是3维向量（轴 × 角度）
```

---

## 实现状态

### Phase 1: 基础变换（已完成 ✅）

- [x] SE(3) 变换操作（创建、组合、求逆、应用）
- [x] SO(3) 旋转操作（基础旋转、组合、求逆、应用）
- [x] 旋转矩阵 ↔ RPY 转换
- [x] 旋转矩阵 ↔ 四元数 转换
- [x] 旋转矩阵 ↔ 轴角 转换
- [x] NumPy后端支持

### Phase 2: 高级功能（已完成 ✅）

- [x] 旋转矩阵 ↔ 欧拉角 转换（多种序列）
- [x] 四元数操作（归一化、共轭、逆、乘法）
- [x] 紧凑轴角表示
- [x] SLERP插值
- [x] 旋转距离和误差计算
- [x] 矩阵验证函数
- [x] Look-At矩阵
- [x] PyTorch后端支持

### Phase 3: 批量处理（已完成 ✅）

- [x] 批量旋转矩阵操作
- [x] 批量转换函数
- [x] 批量变换操作

### Phase 4: 优化与扩展（待实现 ⏳）

- [ ] 对数映射和指数映射（SE(3) / SO(3)）
- [ ] 李代数表示
- [ ] 更高效的批量操作
- [ ] 自动微分支持（PyTorch）

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Quaternions and Rotation Sequences** (Kuipers) - 四元数理论

### 参考库

- **SciPy** - 科学计算库（旋转相关函数）
- **PyTorch3D** - 3D深度学习库
- **Pinocchio** - 高性能机器人学库

---

**文档版本**: 1.0  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

