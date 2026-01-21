# RoboCore 建模模块文档

## 目录

1. [概述](#概述)
2. [模块架构](#模块架构)
3. [核心功能](#核心功能)
4. [RobotModel详解](#robotmodel详解)
5. [多链机器人支持](#多链机器人支持)
6. [双手机器人支持](#双手机器人支持)
7. [解析器系统](#解析器系统)
8. [使用指南](#使用指南)
9. [实现状态](#实现状态)

---

## 概述

RoboCore建模模块提供统一的机器人模型抽象，支持从URDF和MJCF文件加载机器人模型，并提供完整的运动学计算接口。

### 设计原则

- **统一接口**：`RobotModel`类提供统一的接口，隐藏底层解析器细节
- **多格式支持**：支持URDF和MJCF两种主流机器人描述格式
- **多链支持**：支持复杂机器人（如双手机器人）的多链运动学计算
- **高效计算**：使用多链索引系统，支持高效的批量前向运动学计算
- **缓存机制**：解析结果缓存，避免重复解析相同文件
- **双后端支持**：支持NumPy（CPU）和PyTorch（GPU）后端

---

## 模块架构

### 目录结构

```
robocore/modeling/
├── __init__.py                    # 模块导出
├── robot_model.py                 # RobotModel核心类
└── parser/                        # 解析器模块
    ├── __init__.py
    ├── urdf_parser/               # URDF解析器
    │   ├── __init__.py
    │   ├── parser.py              # URDF解析主文件
    │   ├── urdf.py                # URDF数据结构
    │   ├── sdf.py                 # SDF支持
    │   └── xml_reflection/        # XML反射工具
    ├── mjcf_parser/               # MJCF解析器
    │   ├── __init__.py
    │   ├── parser.py              # MJCF解析主文件
    │   ├── mjcf.py                # MJCF数据结构
    │   ├── element.py             # 元素处理
    │   ├── physics.py             # 物理属性
    │   └── ...                    # 其他MJCF工具
    └── utils.py                   # 解析器工具函数
```

### 核心类

```python
class RobotModel:
    """通用串链机器人模型"""
    
    def __init__(self, model_path, base_link=None, end_link=None, ...):
        """初始化机器人模型"""
    
    def fk(self, q, ...):
        """前向运动学"""
    
    def ik(self, target_pose, ...):
        """逆向运动学"""
    
    def jacobian(self, q, ...):
        """雅可比矩阵计算"""
    
    def spawn_chain(self, end_link, ...):
        """创建子链视图"""
    
    def available_leaf_links(self):
        """查找所有叶子链接"""
```

---

## 核心功能

### 1. 模型加载

**支持的格式**：
- **URDF** (`.urdf`)：Unified Robot Description Format
- **MJCF** (`.xml`)：MuJoCo XML Format

**自动检测**：根据文件扩展名自动选择解析器

**缓存机制**：相同路径的模型文件只解析一次，后续加载复用解析结果

**使用示例**：
```python
from robocore.modeling import RobotModel

# 加载URDF模型
robot = RobotModel("path/to/robot.urdf")

# 加载MJCF模型
robot = RobotModel("path/to/robot.xml")

# 指定基座和末端执行器
robot = RobotModel(
    "robot.urdf",
    base_link="base_link",
    end_link="end_effector"
)
```

### 2. 前向运动学（Forward Kinematics）

**功能**：计算给定关节配置下末端执行器的位姿

**公式**：`T_end = fk(q)`

其中：
- `q`: 关节配置 [n×1]
- `T_end`: 末端执行器位姿 [4×4] 齐次变换矩阵

**使用示例**：
```python
# 基本使用
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
result = robot.fk(q)

# 返回字典：link_name -> 4×4变换矩阵
T_end = result[robot.end_link]

# 只返回末端执行器位姿
T_end = robot.fk(q, return_end=True)

# 使用PyTorch后端（批量计算）
import torch
q_batch = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
result = robot.fk(q_batch, device='cuda')
```

**特点**：
- 支持单配置和批量配置
- 支持NumPy和PyTorch后端
- 可返回所有链路的变换矩阵或仅末端执行器

### 3. 逆向运动学（Inverse Kinematics）

**功能**：计算达到目标位姿所需的关节配置

**公式**：`q = ik(T_target, q_initial)`

其中：
- `T_target`: 目标位姿 [4×4]
- `q_initial`: 初始猜测 [n×1]
- `q`: 求解的关节配置 [n×1]

**支持的算法**：
- `pinv`: 伪逆法（默认）
- `dls`: 阻尼最小二乘法
- `transpose`: 转置法

**使用示例**：
```python
import numpy as np

# 目标位姿（4×4齐次变换矩阵）
T_target = np.array([
    [1, 0, 0, 0.5],
    [0, 1, 0, 0.3],
    [0, 0, 1, 0.2],
    [0, 0, 0, 1]
])

# 基本使用
result = robot.ik(T_target)

# 检查是否成功
if result['success']:
    q = result['q']
    print(f"Position error: {result['pos_err']}")
    print(f"Orientation error: {result['ori_err']}")
else:
    print("IK failed")

# 提供初始猜测
q_initial = [0.0] * robot.num_chain_dof
result = robot.ik(T_target, q_initial=q_initial)

# 使用多起点策略提高成功率
result = robot.ik(
    T_target,
    method='dls',
    max_iters=200,
    multi_start=10,  # 10个随机起点
    multi_noise=0.3  # 噪声幅度
)
```

**返回结果**：
```python
{
    'q': [n×1],           # 求解的关节配置
    'success': bool,      # 是否成功
    'pos_err': float,     # 位置误差（米）
    'ori_err': float,     # 姿态误差（弧度）
    'iters': int          # 迭代次数
}
```

### 4. 雅可比矩阵（Jacobian）

**功能**：计算末端执行器速度与关节速度之间的映射关系

**公式**：`v = J(q) · q̇`

其中：
- `J`: 6×n雅可比矩阵（3行线速度 + 3行角速度）
- `q`: 关节配置 [n×1]
- `q̇`: 关节速度 [n×1]
- `v`: 末端执行器空间速度 [6×1]

**计算方法**：
- `analytic`: 解析法（默认，最快最准确）
- `numeric`: 数值微分法
- `autograd`: 自动微分法（PyTorch）

**使用示例**：
```python
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# 基本使用（解析法）
J = robot.jacobian(q)  # 6×n矩阵

# 数值微分法
J = robot.jacobian(q, method='numeric', epsilon=1e-5)

# 计算特定链路的雅可比
J = robot.jacobian(q, target_link='link3')

# 部分雅可比（只计算某些关节）
J = robot.jacobian(q, joint_indices=[0, 1, 2])

# 部分雅可比（只计算某些行，如仅位置）
J = robot.jacobian(q, row_mask=[True, True, True, False, False, False])
```

**应用场景**：
- 速度控制：`q̇ = J^T · F`（力到关节力矩的映射）
- 奇异性检测：`det(J·J^T)`接近0时接近奇异
- 可操作性分析：可操作性椭球分析

### 5. 工作空间分析

**功能**：分析机器人的可达工作空间

**方法**：
- `monte_carlo`: 蒙特卡洛采样（默认）
- `grid`: 网格采样

**使用示例**：
```python
# 计算工作空间（蒙特卡洛，5000个采样点）
robot.compute_workspace(num_samples=5000)

# 检查点是否可达
point = np.array([0.5, 0.3, 0.2])
is_reachable = robot.is_point_reachable(point, tolerance=0.05)

# 获取工作空间边界
bounds = robot.get_workspace_bounds()
# 返回: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
```

### 6. 网格加载与变换

**功能**：加载和变换机器人网格模型

**使用示例**：
```python
# 加载网格（初始化时）
robot = RobotModel("robot.urdf", load_mesh_flag=True)

# 获取变换后的网格
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
vertices, shapes, trans_dict = robot.forward(q)

# 获取trimesh对象
mesh_list = robot.get_forward_robot_mesh(q, base_trans=None)
```

---

## RobotModel详解

### 初始化参数

```python
RobotModel(
    model_path: str | Path,           # 模型文件路径
    base_link: Optional[str] = None, # 基座链路名称（默认：第一个真实链路）
    end_link: Optional[str] = None,  # 末端执行器链路名称（默认：最后一个真实链路）
    load_mesh_flag: bool = False     # 是否加载网格
)
```

### 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `num_dof` | int | 总自由度（所有关节） |
| `num_chain_dof` | int | 链自由度（活动链关节数） |
| `joint_list` | List[str] | 所有关节名称列表 |
| `chain_joint_list` | List[JointSpec] | 活动链关节列表 |
| `joint_limit` | np.ndarray | 关节限位 [n×2] |
| `base_link` | str | 基座链路名称 |
| `end_link` | str | 末端执行器链路名称 |
| `real_link` | List[str] | 真实链路列表 |
| `all_link` | List[str] | 所有链路列表（包括虚拟链路） |

### 主要方法

#### fk - 前向运动学

```python
def fk(
    q: Sequence[float] | Any,
    *,
    return_end: bool = False,
    device: Any | None = None,
    dtype: Any | None = None
) -> Dict[str, Any] | Any
```

**参数**：
- `q`: 关节配置，长度必须等于`num_chain_dof`
- `return_end`: 如果为True，只返回末端执行器位姿（4×4矩阵）
- `device`: PyTorch设备（使用全局后端设置）
- `dtype`: PyTorch数据类型（使用全局后端设置）

**返回**：
- 如果`return_end=False`: 字典 `{link_name: 4×4变换矩阵}`
- 如果`return_end=True`: 单个4×4变换矩阵

#### ik - 逆向运动学

```python
def ik(
    target_pose: List[List[float]],
    q_initial: Optional[Sequence[float]] = None,
    method: str = 'pinv',
    max_iters: int = 120,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-4,
    multi_start: int = 0,
    multi_noise: float = 0.3,
    random_seed: Optional[int] = None,
    **solver_kwargs
) -> Dict[str, Any]
```

**参数**：
- `target_pose`: 目标位姿 [4×4] 嵌套列表
- `q_initial`: 初始猜测（默认：零向量）
- `method`: 算法 `'pinv'|'dls'|'transpose'`
- `max_iters`: 最大迭代次数
- `pos_tol`: 位置容差（米）
- `ori_tol`: 姿态容差（弧度）
- `multi_start`: 多起点数量（0禁用）
- `multi_noise`: 多起点噪声幅度（弧度）
- `random_seed`: 随机种子

**返回**：
```python
{
    'q': List[float],      # 求解的关节配置
    'success': bool,       # 是否成功
    'pos_err': float,      # 位置误差
    'ori_err': float,      # 姿态误差
    'iters': int           # 迭代次数
}
```

#### jacobian - 雅可比矩阵

```python
def jacobian(
    q: Sequence[float] | Any,
    *,
    method: str = 'analytic',
    epsilon: float = 5e-5,
    use_central_diff: bool = True,
    device: Any | None = None,
    dtype: Any | None = None,
    target_link: str | None = None,
    joint_indices: Sequence[int] | None = None,
    row_mask: Sequence[int | bool] | None = None
) -> Any
```

**参数**：
- `q`: 关节配置
- `method`: 计算方法 `'analytic'|'numeric'|'autograd'`
- `epsilon`: 数值微分步长（仅numeric方法）
- `use_central_diff`: 使用中心差分（更准确）
- `target_link`: 目标链路（默认：末端执行器）
- `joint_indices`: 只计算指定关节的列
- `row_mask`: 只计算指定行（如仅位置或仅姿态）

**返回**：6×n雅可比矩阵（numpy.ndarray或torch.Tensor）

#### random_q - 随机关节配置

```python
def random_q(
    seed: int = None,
    scale: float = 0.5
) -> List[float]
```

**参数**：
- `seed`: 随机种子（默认：None，使用随机种子）
- `scale`: 关节范围缩放因子（0.0-1.0，默认0.5表示中间50%范围）

**返回**：随机关节配置列表（长度 = num_chain_dof）

**相关方法**：
- `random_q_batch(batch_size, seed, scale)` - 批量生成链配置
- `random_q_full(seed, scale)` - 生成完整配置空间（所有DOF）
- `random_q_full_batch(batch_size, seed, scale)` - 批量生成完整配置
- `random_pose(seed, scale)` - 生成随机位姿
- `random_pose_batch(batch_size, seed, scale)` - 批量生成随机位姿

---

## 多链机器人支持

### 概念

对于复杂机器人（如双手机器人、移动机械臂），一个机器人模型可能包含多个运动学链。RoboCore支持通过`spawn_chain`来创建子链视图，或使用统一配置空间进行多链协调求解。

### spawn_chain - 创建子链视图

**功能**：从完整机器人模型中创建一个轻量级的子链视图

**特点**：
- 共享相同的解析结果（不重复解析）
- 独立的DOF和关节顺序
- 独立的运动学计算接口

**使用示例**：
```python
# 加载完整机器人模型
full_robot = RobotModel("bimanual_robot.urdf")

# 创建左臂子链
left_arm = full_robot.spawn_chain(
    end_link="left_end_effector",
    base_link="base_link"
)

# 创建右臂子链
right_arm = full_robot.spawn_chain(
    end_link="right_end_effector",
    base_link="base_link"
)

# 独立使用
q_left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T_left = left_arm.fk(q_left)

q_right = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
T_right = right_arm.fk(q_right)
```

### available_leaf_links - 查找叶子链接

**功能**：查找机器人模型中的所有叶子链接（没有子链接的链接）

**使用场景**：用于发现机器人的多个末端执行器（如左右手爪）

**使用示例**：
```python
# 查找所有叶子链接
leaf_links = robot.available_leaf_links()
print(f"Available end-effectors: {leaf_links}")

# 通常用于双手机器人
left_end = next(l for l in leaf_links if 'left' in l.lower())
right_end = next(l for l in leaf_links if 'right' in l.lower())

# 创建子链
left_arm = robot.spawn_chain(left_end)
right_arm = robot.spawn_chain(right_end)
```

### 多链逆向运动学

**功能**：同时满足多个链路的位姿约束，使用统一配置空间

**使用示例**：
```python
import numpy as np

# 加载完整机器人模型（包含多个运动链）
robot = RobotModel("bimanual_robot.urdf")

# 定义多个目标位姿
T_left_target = np.eye(4)
T_left_target[:3, 3] = [0.5, 0.3, 0.2]

T_right_target = np.eye(4)
T_right_target[:3, 3] = [0.5, -0.3, 0.2]

# 使用多链模式求解（统一配置空间）
result = robot.ik(
    targets={
        'left_end_effector': T_left_target,
        'right_end_effector': T_right_target,
    },
    end_links=['left_end_effector', 'right_end_effector'],
    base_link='base_link',
    method='dls',
    max_iters=200,
    pos_tol=1e-3,
    ori_tol=1e-3
)

if result['success']:
    q_full = result['q']  # 完整配置空间 [num_dof]
    print(f"Full configuration: {q_full}")
    
    # 提取各链的关节值
    left_indices = robot._get_joint_indices('base_link', 'left_end_effector')
    right_indices = robot._get_joint_indices('base_link', 'right_end_effector')
    
    q_left = q_full[left_indices]
    q_right = q_full[right_indices]
    print(f"Left arm: {q_left}")
    print(f"Right arm: {q_right}")
```

---

## 双手机器人支持

### 统一配置空间方法（推荐）

**功能**：使用统一配置空间同时控制多个运动链

**使用示例**：
```python
import numpy as np
from robocore.modeling import RobotModel

# 加载双手机器人模型
robot = RobotModel("bimanual_robot.urdf")

# 获取左右臂的关节索引
left_indices = robot._get_joint_indices('base_link', 'left_end_effector')
right_indices = robot._get_joint_indices('base_link', 'right_end_effector')

# 构建完整配置向量
q_left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
q_right = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
q_full = np.zeros(robot.num_dof)
q_full[left_indices] = q_left
q_full[right_indices] = q_right

# 同时计算左右臂前向运动学
result = robot.fk(
    q_full,
    base_link='base_link',
    end_link=None  # 返回所有链路的位姿
)
T_left = result['left_end_effector']
T_right = result['right_end_effector']

# 同时计算左右臂逆向运动学
ik_result = robot.ik(
    targets={
        'left_end_effector': T_left_target,
        'right_end_effector': T_right_target,
    },
    end_links=['left_end_effector', 'right_end_effector'],
    base_link='base_link',
    method='dls',
    max_iters=200
)

if ik_result['success']:
    q_full = ik_result['q']
    q_left = q_full[left_indices]
    q_right = q_full[right_indices]
```

### 使用 bimanual 模块（兼容性接口）

**功能**：提供便捷的双臂运动学接口（向后兼容）

**使用示例**：
```python
from robocore.kinematics.bimanual import (
    bimanual_forward_kinematics,
    bimanual_inverse_kinematics,
    bimanual_jacobian
)

# 创建子链视图
left_arm = robot.spawn_chain("left_end_effector")
right_arm = robot.spawn_chain("right_end_effector")

# 前向运动学
result = bimanual_forward_kinematics(
    left_arm, right_arm,
    q_left, q_right,
    mode='indep'  # 'indep'|'relative'|'mirror'
)

# 逆向运动学
ik_result = bimanual_inverse_kinematics(
    left_arm, right_arm,
    target_left=T_left_target,
    target_right=T_right_target,
    q0_left=q_left_init,
    q0_right=q_right_init
)

# 雅可比矩阵
J = bimanual_jacobian(
    left_arm, right_arm,
    q_left, q_right,
    mode='indep'
)
```

**注意**：`bimanual` 模块中的函数已标记为向后兼容接口，推荐使用统一配置空间方法。

---

## 解析器系统

### URDF解析器

**功能**：解析URDF（Unified Robot Description Format）文件

**支持的元素**：
- Links（链路）
- Joints（关节）
- Visuals（视觉）
- Collisions（碰撞）
- Materials（材料）
- Transmissions（传动）

**使用示例**：
```python
from robocore.modeling.parser.urdf_parser import URDFParser

parser = URDFParser("robot.urdf")
joint_names = parser.get_joint_names()
link_names = parser.get_link_names()
joint_limits = parser.get_joint_limits()
```

### MJCF解析器

**功能**：解析MJCF（MuJoCo XML Format）文件

**特点**：
- 支持MuJoCo物理仿真格式
- 支持更丰富的物理属性
- 支持资产（mesh、texture等）管理

**使用示例**：
```python
from robocore.modeling.parser.mjcf_parser import MJCFParser

parser = MJCFParser("robot.xml")
# 接口与URDFParser相同
```

### 解析器接口

所有解析器实现统一的接口：

```python
class Parser:
    def get_joint_names(self) -> List[str]
    def get_link_names(self) -> List[str]
    def get_real_link_names(self) -> List[str]
    def get_joint_limits(self) -> np.ndarray
    def get_link_mesh_map(self) -> Dict[str, Dict]
    def get_link_virtual_map(self) -> Tuple[Dict, Dict]
```

---

## 使用指南

### 基本使用流程

1. **加载模型**
```python
from robocore.modeling import RobotModel

robot = RobotModel("robot.urdf")
print(f"DOF: {robot.num_chain_dof}")
print(f"End link: {robot.end_link}")
```

2. **前向运动学**
```python
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T_end = robot.fk(q, return_end=True)
print(f"End-effector pose:\n{T_end}")
```

3. **逆向运动学**
```python
import numpy as np

# 目标位姿
T_target = np.eye(4)
T_target[:3, 3] = [0.5, 0.3, 0.2]

# 求解
result = robot.ik(T_target)
if result['success']:
    q = result['q']
    print(f"Joint configuration: {q}")
```

4. **雅可比矩阵**
```python
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
J = robot.jacobian(q)
print(f"Jacobian shape: {J.shape}")  # (6, n)
```

### 多链机器人使用

```python
import numpy as np

# 加载完整模型（包含多个运动链）
robot = RobotModel("bimanual_robot.urdf")

# 方式1: 创建子链视图（独立使用）
left_arm = robot.spawn_chain("left_end_effector")
right_arm = robot.spawn_chain("right_end_effector")

# 独立计算各链的前向运动学
q_left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
q_right = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
T_left = left_arm.fk(q_left, return_end=True)
T_right = right_arm.fk(q_right, return_end=True)

# 方式2: 使用统一配置空间（推荐用于多链协调）
# 构建完整配置向量
q_full = np.zeros(robot.num_dof)
left_indices = robot._get_joint_indices('base_link', 'left_end_effector')
right_indices = robot._get_joint_indices('base_link', 'right_end_effector')
q_full[left_indices] = q_left
q_full[right_indices] = q_right

# 同时计算多个链的前向运动学
result = robot.fk(
    q_full,
    base_link='base_link',
    end_link=None  # None表示返回所有链路的位姿
)
T_left = result['left_end_effector']
T_right = result['right_end_effector']

# 多链逆向运动学
ik_result = robot.ik(
    targets={
        'left_end_effector': T_left_target,
        'right_end_effector': T_right_target,
    },
    end_links=['left_end_effector', 'right_end_effector'],
    base_link='base_link'
)
```

### 双手机器人使用

```python
import numpy as np
from robocore.modeling import RobotModel

# 加载双手机器人模型
robot = RobotModel("bimanual_robot.urdf")

# 获取左右臂的关节索引
left_indices = robot._get_joint_indices('base_link', 'left_end_effector')
right_indices = robot._get_joint_indices('base_link', 'right_end_effector')

# 构建完整配置向量
q_left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
q_right = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
q_full = np.zeros(robot.num_dof)
q_full[left_indices] = q_left
q_full[right_indices] = q_right

# 同时计算左右臂前向运动学
result = robot.fk(
    q_full,
    base_link='base_link',
    end_link=None  # 返回所有链路的位姿
)
T_left = result['left_end_effector']
T_right = result['right_end_effector']

# 同时计算左右臂逆向运动学
ik_result = robot.ik(
    targets={
        'left_end_effector': T_left_target,
        'right_end_effector': T_right_target,
    },
    end_links=['left_end_effector', 'right_end_effector'],
    base_link='base_link'
)
```

### 工作空间分析

```python
# 计算工作空间
robot.compute_workspace(num_samples=5000)

# 检查可达性
point = np.array([0.5, 0.3, 0.2])
if robot.is_point_reachable(point):
    print("Point is reachable")

# 获取边界
bounds = robot.get_workspace_bounds()
print(f"X range: {bounds['x']}")
print(f"Y range: {bounds['y']}")
print(f"Z range: {bounds['z']}")
```

### 网格可视化

```python
# 加载网格
robot = RobotModel("robot.urdf", load_mesh_flag=True)

# 获取变换后的网格
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
mesh_list = robot.get_forward_robot_mesh(q)

# 使用trimesh可视化
import trimesh
scene = trimesh.Scene(mesh_list)
scene.show()
```

### 模型信息查看

```python
# 模型摘要
robot.summary()

# 打印树结构
robot.print_tree(show_joints=True)

# 查看可用末端链路
leaf_links = robot.available_leaf_links()
print(f"Available end links: {leaf_links}")
```

---

## 实现状态

### Phase 1: 基础功能（已完成 ✅）

- [x] URDF解析器
- [x] MJCF解析器
- [x] RobotModel基础类
- [x] 前向运动学（FK）
- [x] 逆向运动学（IK）
- [x] 雅可比矩阵计算
- [x] 模型加载与缓存

### Phase 2: 多链支持（已完成 ✅）

- [x] spawn_chain子链创建
- [x] available_leaf_links叶子链接查找
- [x] 统一配置空间多链IK/FK
- [x] 多链索引系统

### Phase 3: 高级功能（已完成 ✅）

- [x] 工作空间分析
- [x] 网格加载与变换
- [x] 简单形状支持（sphere, box, cylinder, capsule）
- [x] 批量计算支持（PyTorch后端）

### Phase 4: 优化与扩展（部分实现 ⏳）

- [x] 解析结果缓存
- [x] 多链索引优化
- [ ] 动力学计算（待动力学模块）
- [ ] 碰撞检测（待实现）
- [ ] 可视化工具（待实现）

---

## 参考资料

### 推荐书籍

- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Introduction to Robotics** (Craig) - 经典教材

### 参考库

- **Pinocchio** - 高性能动力学库
- **Drake** - MIT机器人工具箱
- **PyBullet** - 物理仿真库
- **MuJoCo** - 物理仿真引擎

### 格式规范

- **URDF**: http://wiki.ros.org/urdf
- **MJCF**: https://mujoco.readthedocs.io/en/latest/XMLreference.html

---

**文档版本**: 1.0  
**最后更新**: 2025-01-XX  
**作者**: Synria Robotics Team

