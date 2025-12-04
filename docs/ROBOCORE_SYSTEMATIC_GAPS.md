# RoboCore功能完整性评估 - 基础机器人学体系

## 📚 按机器人学基础知识体系梳理

---

## 第一层：数学基础与变换 (Transformations)

### 1. 坐标变换与表示 ✅ 较完善

**已有功能** (`robocore/transform/transform_core.py`):
```
✅ RPY (Roll-Pitch-Yaw) → 旋转矩阵
✅ 轴角 (Axis-Angle) → 旋转矩阵 (Rodrigues公式)
✅ 齐次变换矩阵构造 (4x4)
✅ 姿态误差计算 (轴角表示)
✅ 沿轴平移
✅ NumPy/PyTorch双后端支持
```

**可能缺失**:
```
⚠️ 四元数 (Quaternion) 完整支持
   - 四元数 ↔ 旋转矩阵
   - 四元数 ↔ 轴角
   - 四元数 ↔ RPY
   - 四元数插值 (SLERP已有但在trajectory模块)
   - 四元数运算 (乘法、共轭、逆)

⚠️ 欧拉角 (Euler Angles) 多种约定
   - 当前只有RPY (ZYX约定)
   - 缺少其他约定 (XYZ, ZXZ等12种)
   - 欧拉角 ↔ 旋转矩阵的通用转换

❌ 旋转矩阵 → 各种表示的逆转换
   - 旋转矩阵 → RPY (已有但可能不完整)
   - 旋转矩阵 → 轴角
   - 旋转矩阵 → 四元数
   - 旋转矩阵 → 欧拉角

❌ 对偶四元数 (Dual Quaternion)
   - 同时表示旋转+平移
   - 用于螺旋运动 (Screw Motion)
   
❌ 旋量 (Twist) 和 Wrench
   - 6D速度/力表示
   - 伴随变换 (Adjoint)
```

**建议**:
```python
# robocore/transform/rotation.py (新建)
from dataclasses import dataclass

@dataclass
class Quaternion:
    """四元数类 (x, y, z, w)"""
    x: float
    y: float
    z: float
    w: float
    
    def to_rotation_matrix(self) -> np.ndarray: ...
    def from_rotation_matrix(R: np.ndarray) -> 'Quaternion': ...
    def to_axis_angle(self) -> Tuple[np.ndarray, float]: ...
    def multiply(self, other: 'Quaternion') -> 'Quaternion': ...
    def conjugate(self) -> 'Quaternion': ...
    def normalize(self) -> 'Quaternion': ...

def rotation_matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]: ...
def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]: ...
def rotation_matrix_to_quaternion(R: np.ndarray) -> Quaternion: ...
def rotation_matrix_to_euler(R: np.ndarray, convention='XYZ') -> Tuple[float, float, float]: ...
```

---

## 第二层：机器人建模 (Robot Modeling)

### 2. 机器人描述格式 ✅ 已支持

**已有功能**:
```
✅ URDF解析 (robocore/modeling/robot_model.py)
✅ MJCF支持 (通过MuJoCo桥接)
✅ 运动链提取 (串联机构)
✅ 关节限位读取
```

**可能缺失**:
```
⚠️ DH参数完整支持
   - 有DH提取功能 (robocore/transform/dh.py)
   - 但可能缺少：直接从DH参数构建机器人

❌ 并联机构 (Parallel Robots)
   - 当前只支持串联机构
   - 缺少闭环约束处理
   - 缺少并联运动学求解

❌ 树状机构 (Tree Structures)
   - 人形机器人、多臂机器人
   - 分支运动学链

❌ 移动底座集成
   - 移动机械臂 (Mobile Manipulator)
   - 浮动基座 (Floating Base)

❌ 软体机器人 / 连续体机器人
   - 连续变形模型
```

**建议**:
```python
# robocore/modeling/dh_robot.py (新建)
class DHRobot(RobotModel):
    """从DH参数直接构建机器人"""
    def __init__(self, dh_params: List[DHRow], convention='standard'):
        ...

# robocore/modeling/parallel_robot.py (新建)
class ParallelRobot(RobotModel):
    """并联机器人"""
    def __init__(self, urdf_path, closed_chains):
        ...
        
# robocore/modeling/tree_robot.py (新建)
class TreeRobot(RobotModel):
    """树状机器人（人形等）"""
    def forward_kinematics_branch(self, branch_name, q):
        ...
```

---

## 第三层：正运动学 (Forward Kinematics)

### 3. 正运动学求解 ✅ 完善

**已有功能**:
```
✅ URDF正运动学 (robocore/modeling/robot_model.py)
✅ DH正运动学 (standard/modified)
✅ 返回末端位姿 (4x4矩阵)
✅ 返回所有连杆位姿 (可选)
✅ NumPy/PyTorch后端
✅ 批量并行计算
```

**可能缺失**:
```
⚠️ 速度正运动学
   - 给定 q, qd → 末端线速度/角速度
   - 通过雅可比: v = J * qd (已有雅可比，但可能没封装)

⚠️ 加速度正运动学
   - 给定 q, qd, qdd → 末端加速度
   - a = J * qdd + dJ/dt * qd

❌ 任意连杆FK查询
   - 当前只能查末端或所有连杆
   - 缺少指定连杆名称的FK
```

**建议**:
```python
# robocore/kinematics/fk.py (增强)
def forward_velocity(
    robot_model: RobotModel,
    q: np.ndarray,
    qd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回末端线速度和角速度
    Returns: (v_linear [3], v_angular [3])
    """
    J = jacobian(robot_model, q)
    twist = J @ qd
    return twist[:3], twist[3:]

def forward_kinematics_link(
    robot_model: RobotModel,
    q: np.ndarray,
    link_name: str
) -> np.ndarray:
    """返回指定连杆的位姿"""
    ...
```

---

## 第四层：微分运动学 (Differential Kinematics)

### 4. 雅可比矩阵 ✅ 完善

**已有功能**:
```
✅ 几何雅可比 (Geometric Jacobian)
✅ 解析雅可比 (Analytic Jacobian)
✅ 数值雅可比 (前向/中心差分)
✅ 批量并行计算
✅ GPU加速
```

**可能缺失**:
```
⚠️ 雅可比导数 (dJ/dt)
   - 用于加速度运动学
   - 用于动力学计算

⚠️ 零空间 (Null Space) 计算
   - 冗余机器人优化
   - NS = I - J_pinv @ J

❌ 任意连杆的雅可比
   - 当前只能计算末端雅可比
   - 缺少中间连杆的雅可比

❌ 任务空间雅可比
   - 部分自由度约束 (只考虑位置/只考虑姿态)
```

**建议**:
```python
# robocore/kinematics/jacobian.py (增强)
def jacobian_derivative(
    robot_model: RobotModel,
    q: np.ndarray,
    qd: np.ndarray
) -> np.ndarray:
    """计算雅可比导数 dJ/dt"""
    ...

def null_space_projector(J: np.ndarray) -> np.ndarray:
    """计算零空间投影矩阵"""
    J_pinv = np.linalg.pinv(J)
    return np.eye(J.shape[1]) - J_pinv @ J
    
def task_jacobian(
    robot_model: RobotModel,
    q: np.ndarray,
    task_type: str  # 'position', 'orientation', 'full'
) -> np.ndarray:
    """任务相关雅可比"""
    J = jacobian(robot_model, q)
    if task_type == 'position':
        return J[:3, :]  # 只要前3行
    elif task_type == 'orientation':
        return J[3:, :]  # 只要后3行
    return J
```

---

## 第五层：逆运动学 (Inverse Kinematics)

### 5. 逆运动学求解 ✅ 很强

**已有功能**:
```
✅ DLS (Damped Least Squares)
✅ Pseudoinverse
✅ Jacobian Transpose
✅ 自适应阻尼
✅ 自适应步长
✅ 关节限位处理
✅ Multi-start
✅ 批量/顺序IK
✅ GPU加速
```

**可能缺失**:
```
⚠️ 解析IK (Closed-form)
   - 6-DOF特殊几何结构 (如Puma, UR等)
   - 比数值IK快100倍+
   - 可以找到所有解

⚠️ 冗余IK (7-DOF+)
   - 零空间优化
   - 次级任务 (manipulability, joint limit avoidance)

❌ 多解管理
   - IK通常有多个解
   - 缺少解空间采样
   - 缺少解选择策略 (最近、最优等)

❌ 约束IK
   - 指定关节约束 (某些关节固定)
   - 部分位姿约束 (只约束位置/只约束姿态)
   - 已有pos_weight/ori_weight，但不够灵活

❌ 多目标IK
   - 双臂协调
   - 多末端同时到达

❌ 碰撞感知IK
   - 避免自碰撞
   - 避免环境碰撞
```

**建议**:
```python
# robocore/kinematics/ik_analytical.py (新建)
class AnalyticalIKSolver:
    """特定机器人的解析IK"""
    @staticmethod
    def solve_6dof_spherical_wrist(T, d, a):
        """求解球形手腕6自由度机器人 (Puma, UR, ABB等)"""
        # 返回所有可能的解 (最多8个)
        ...

# robocore/kinematics/ik.py (增强)
def inverse_kinematics_redundant(
    robot_model: RobotModel,
    target_pose: np.ndarray,
    q0: np.ndarray,
    secondary_objective: str = 'manipulability',  # 或 'joint_limit', 'energy'
    **kwargs
) -> Dict:
    """冗余机器人IK + 零空间优化"""
    ...

def inverse_kinematics_constrained(
    robot_model: RobotModel,
    target_pose: np.ndarray,
    q0: np.ndarray,
    fixed_joints: List[int] = None,  # 固定某些关节
    task_mask: np.ndarray = None,    # [1,1,1,0,0,0] = 只约束位置
    **kwargs
) -> Dict:
    """约束IK"""
    ...
```

---

## 第六层：奇异性分析 (Singularity Analysis)

### 6. 奇异性检测 ✅ 已有

**已有功能**:
```
✅ 条件数 (Condition Number)
✅ 操作度 (Manipulability)
✅ 奇异值分析 (SVD)
```

**可能缺失**:
```
⚠️ 奇异性分类
   - 边界奇异 (Boundary Singularity)
   - 内部奇异 (Interior Singularity)
   - 位姿奇异 vs 配置奇异

❌ 奇异性回避
   - 梯度场避障
   - 最小化条件数

❌ 奇异性附近的特殊处理
   - 降秩运动 (Reduced-rank motion)
   - 可实现任务分析
```

---

## 第七层：工作空间分析 (Workspace Analysis)

### 7. 工作空间计算 ✅ 较完善

**已有功能**:
```
✅ 可达工作空间 (Reachable Workspace)
✅ 灵巧工作空间 (Dexterous Workspace)
✅ 体积估算
✅ 密度分析
✅ 奇异性无关区域
```

**可能缺失**:
```
⚠️ 工作空间切片
   - 2D截面可视化
   - 特定高度的可达范围

❌ 方向工作空间
   - 给定位置，可达的所有姿态
   - 姿态能力图 (Orientation Capability Map)

❌ 速度/加速度工作空间
   - 给定位置/姿态，最大速度
   - 各向同性分析

❌ 力/力矩工作空间
   - 给定位置，最大输出力
   - 力椭球
```

---

## 第八层：轨迹规划 (Trajectory Planning)

### 8.1 关节空间轨迹 ✅ 完善

**已有功能**:
```
✅ 线性插值
✅ 三次多项式
✅ 五次多项式
✅ 多路径点规划
```

### 8.2 Cartesian空间轨迹 ✅ 较完善

**已有功能**:
```
✅ 直线插值
✅ 圆弧插值
✅ SLERP姿态插值
```

**可能缺失**:
```
⚠️ 自动步数估算 (AD-SDK有，需迁移)
⚠️ 夹爪/末端执行器通道 (AD-SDK有，需迁移)

❌ 样条曲线 (Spline)
   - B样条
   - Bezier曲线
   - NURBS

❌ 路径跟随 vs 轨迹跟随
   - Path following: 不关心时间
   - Trajectory following: 严格时间同步

❌ 时间最优轨迹
   - 给定路径，最短时间
   - 考虑速度/加速度限制
```

### 8.3 速度剖面 ✅ 已有

**已有功能**:
```
✅ 梯形速度剖面
✅ S曲线速度剖面 (Jerk限制)
✅ 恒速剖面
```

**可能缺失**:
```
⚠️ 速度剖面缩放
   - 减速/加速整个轨迹
   - 功能有但可能不完整

❌ 最优速度规划
   - Bang-bang控制
   - 时间最优控制
```

### 8.4 高级规划 ⚠️ 部分缺失

**可能缺失**:
```
⚠️ LQT轨迹优化 (在AD-SDK，需迁移)

❌ 避障规划
   - RRT (Rapidly-exploring Random Tree)
   - PRM (Probabilistic Roadmap)
   - A* / Dijkstra

❌ 动态规划
   - 考虑动力学的轨迹优化
   - 最小能量轨迹

❌ 约束优化
   - 工作空间约束
   - 碰撞约束
   - 视线约束 (Line-of-sight)
```

---

## 第九层：轨迹插值与时间同步 ⚠️ 需加强

### 9. 轨迹密集化 ⚠️ 部分缺失

**已有功能**:
```
✅ 基础插值函数 (Slerp等)
```

**缺失功能**:
```
❌ 时间重采样
   - 稀疏轨迹 → 固定频率密集轨迹
   - resample_trajectory(t, q, freq)

❌ 轨迹平滑
   - 噪声轨迹的平滑处理
   - 滤波 (Moving average, Butterworth)

❌ 轨迹时间缩放
   - 加速/减速整个轨迹
   - 保持形状不变

❌ 在线插值器
   - 实时场景：当前点→目标点平滑过渡
   - 梯形剖面生成
   - 留给SDK实现，但RoboCore应提供算法
```

**建议**:
```python
# robocore/planning/trajectory/interpolation.py (新建)
def resample_trajectory(
    t_orig: np.ndarray,
    q_orig: np.ndarray,
    control_freq: float,
    method: str = 'cubic'
) -> Tuple[np.ndarray, np.ndarray]:
    """将轨迹重采样到固定频率"""
    ...

def smooth_trajectory(
    t: np.ndarray,
    q: np.ndarray,
    method: str = 'moving_average',
    window: int = 5
) -> np.ndarray:
    """平滑噪声轨迹"""
    ...

def time_scale_trajectory(
    t: np.ndarray,
    q: np.ndarray,
    scale_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """时间缩放：scale > 1 = 减速，scale < 1 = 加速"""
    ...
```

---

## 第十层：动力学 (Dynamics) ❌ 完全缺失

### 10. 动力学建模与计算 ❌ 需新建

**当前状态**:
```
❌ robocore/dynamics/ 文件夹为空
```

**需要的功能**:
```
❌ 正动力学 (Forward Dynamics)
   - M(q)qdd = tau - C(q,qd) - g(q)
   - 给定力矩 → 计算加速度

❌ 逆动力学 (Inverse Dynamics)
   - tau = M(q)qdd + C(q,qd) + g(q)
   - 给定运动 → 计算所需力矩

❌ 质量矩阵 (Mass Matrix)
   - M(q): 广义质量矩阵

❌ 科氏力和离心力 (Coriolis & Centrifugal)
   - C(q, qd): 速度相关力

❌ 重力项 (Gravity Vector)
   - g(q): 重力补偿

❌ 摩擦力模型
   - 粘性摩擦
   - 库伦摩擦
   - Stribeck效应

❌ 驱动器模型
   - 电机动力学
   - 减速器效应
```

**建议**:
```python
# robocore/dynamics/__init__.py (新建)
from .inverse_dynamics import inverse_dynamics
from .forward_dynamics import forward_dynamics
from .mass_matrix import mass_matrix
from .coriolis import coriolis_centrifugal
from .gravity import gravity_vector

# robocore/dynamics/inverse_dynamics.py
def inverse_dynamics(
    robot_model: RobotModel,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    external_forces: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算所需关节力矩
    Returns: tau [n×1]
    """
    ...

# robocore/dynamics/forward_dynamics.py
def forward_dynamics(
    robot_model: RobotModel,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray
) -> np.ndarray:
    """
    计算关节加速度
    Returns: qdd [n×1]
    """
    ...
```

**实现方案**:
1. 递归Newton-Euler算法 (高效)
2. Lagrange方程法 (教学)
3. 或集成Pinocchio库 (最优性能)

---

## 第十一层：控制 (Control) ❌ 完全缺失

### 11. 控制算法 ❌ 需新建

**当前状态**:
```
❌ robocore/controller/ 文件夹为空
```

**需要的功能**:
```
❌ 位置控制器
   - PD控制
   - PID控制

❌ 轨迹跟踪控制
   - 前馈 + 反馈
   - Computed Torque Control

❌ 力控制
   - 阻抗控制 (Impedance Control)
   - 导纳控制 (Admittance Control)
   - 混合位置/力控制

❌ 在线插值器
   - 实时场景的平滑插值
   - 梯形速度剖面生成
```

**设计原则**:
- **RoboCore**: 提供控制算法（只计算控制律）
- **SDK**: 负责实时执行（读写硬件）

**建议**:
```python
# robocore/controller/trajectory_controller.py (新建)
class TrajectoryController:
    """轨迹跟踪PD/PID控制"""
    def __init__(self, Kp, Kd, Ki=None):
        ...
    
    def compute_control(
        self, 
        q_ref, qd_ref, qdd_ref,  # 期望状态
        q_actual, qd_actual       # 实际状态
    ) -> np.ndarray:
        """返回控制力矩（或位置修正）"""
        ...

# robocore/controller/computed_torque.py (新建)
class ComputedTorqueController:
    """基于模型的前馈+反馈控制"""
    def __init__(self, robot_model, Kp, Kd):
        ...
    
    def compute_control(self, q_ref, qd_ref, qdd_ref, q, qd):
        # 前馈（动力学补偿）
        tau_ff = inverse_dynamics(self.model, q, qd, qdd_ref)
        # 反馈（误差修正）
        tau_fb = self.Kp @ (q_ref - q) + self.Kd @ (qd_ref - qd)
        return tau_ff + tau_fb
```

---

## 第十二层：传感与估计 (Sensing & Estimation) ❌ 不在RoboCore范围

**说明**: 这些功能通常在SDK层实现（硬件相关）

```
- 状态估计 (State Estimation)
- 卡尔曼滤波 (Kalman Filter)
- 力/力矩传感器处理
- 视觉伺服 (Visual Servoing)
```

---

## 第十三层：碰撞检测 (Collision Detection) ❌ 缺失

### 12. 碰撞检测 ❌ 需新建（可选）

**可能缺失**:
```
❌ 自碰撞检测
   - 连杆之间的碰撞
   - 简化几何体 (球、胶囊、凸包)

❌ 环境碰撞检测
   - 机器人 vs 障碍物
   - AABB / OBB碰撞

❌ 距离计算
   - 最小距离查询
   - 最近点对

❌ 碰撞响应
   - 穿透深度
   - 接触法向
```

**建议**:
- 集成专业库（如FCL, Bullet, PyBullet）
- 或提供简单的几何碰撞检测

---

## 📋 优先级总结

### 🔴 P0 - 基础完整性（必须补）

1. **四元数完整支持** [第一层]
   - 四元数类及转换函数
   - 是机器人学最常用的姿态表示

2. **轨迹插值工具** [第九层]
   - 时间重采样
   - 轨迹平滑
   - 时间缩放

3. **旋转表示转换** [第一层]
   - 旋转矩阵 → RPY/轴角/四元数
   - 是基础工具函数

### 🟡 P1 - 重要功能（提升能力）

4. **动力学模块** [第十层]
   - 逆动力学（支持力矩控制）
   - 质量矩阵、科氏力、重力

5. **控制器模块** [第十一层]
   - PD/PID控制器
   - 计算力矩控制

6. **速度/加速度运动学** [第三、四层]
   - forward_velocity
   - jacobian_derivative

7. **冗余IK** [第五层]
   - 零空间优化
   - 7-DOF支持

8. **LQT轨迹规划** [第八层]
   - 从AD-SDK迁移

### 🟢 P2 - 高级特性（扩展场景）

9. **解析IK** [第五层]
   - 特定机器人的闭式解
   - 性能提升100倍+

10. **并联/树状机器人** [第二层]
    - 双臂、人形机器人

11. **避障规划** [第八层]
    - RRT, PRM

12. **碰撞检测** [第十三层]
    - 自碰撞、环境碰撞

13. **样条轨迹** [第八层]
    - B样条、NURBS

---

## 📐 功能完整度矩阵

| 层级 | 模块 | 完整度 | 缺口 |
|-----|------|--------|------|
| 1 | 坐标变换 | 70% | 四元数、旋转转换 |
| 2 | 机器人建模 | 80% | 并联、树状 |
| 3 | 正运动学 | 95% | 速度FK |
| 4 | 雅可比 | 90% | dJ/dt |
| 5 | 逆运动学 | 85% | 冗余IK、解析IK |
| 6 | 奇异性 | 75% | 奇异性回避 |
| 7 | 工作空间 | 70% | 方向/力工作空间 |
| 8 | 轨迹规划 | 75% | LQT、避障、样条 |
| 9 | 轨迹插值 | 40% | **时间重采样** |
| 10 | 动力学 | **0%** | **全部缺失** |
| 11 | 控制 | **0%** | **全部缺失** |
| 12 | 传感估计 | N/A | 不在范围 |
| 13 | 碰撞检测 | 0% | 全部缺失 |

---

## 🎯 按顺序应该做的事

### 第一阶段：基础补全（1-2周）

1. **四元数支持** [P0]
   ```python
   robocore/transform/quaternion.py
   - Quaternion类
   - 各种转换函数
   ```

2. **旋转表示转换** [P0]
   ```python
   robocore/transform/rotation_convert.py
   - rotation_matrix_to_*系列函数
   ```

3. **轨迹插值工具** [P0]
   ```python
   robocore/planning/trajectory/interpolation.py
   - resample_trajectory
   - smooth_trajectory
   - time_scale_trajectory
   ```

### 第二阶段：核心能力（2-3周）

4. **速度运动学** [P1]
   ```python
   robocore/kinematics/fk.py
   - forward_velocity
   robocore/kinematics/jacobian.py
   - jacobian_derivative
   ```

5. **动力学模块** [P1]
   ```python
   robocore/dynamics/
   - inverse_dynamics.py
   - mass_matrix.py
   - coriolis.py
   - gravity.py
   ```

6. **控制器模块** [P1]
   ```python
   robocore/controller/
   - trajectory_controller.py
   - computed_torque.py
   ```

7. **LQT规划迁移** [P1]
   ```python
   robocore/planning/trajectory/lqt_planner.py
   ```

### 第三阶段：高级特性（按需）

8. **冗余IK** [P2]
9. **解析IK** [P2]
10. **避障规划** [P2]
11. **碰撞检测** [P2]

---

## 💡 关键洞察

### 最缺的不是高级功能，而是基础工具

1. **四元数** - 最常用的姿态表示，竟然不完整
2. **轨迹插值** - 连接规划和执行的桥梁
3. **旋转转换** - 各种表示法之间转换是基本需求

### 动力学和控制虽然缺失，但影响有限

- 对于**位置控制机器人**（舵机）：不需要动力学
- 对于**力矩控制机器人**（伺服）：必须要动力学

### RoboCore应专注算法，不要做硬件相关

- ✅ 提供控制算法（计算控制律）
- ❌ 不做实时IO（留给SDK）
- ✅ 可离线使用（仿真、分析）

---

**建议**: 先从**第一阶段**的基础补全开始，这些是最基础且影响最广的功能。特别是**四元数**和**轨迹插值**，几乎所有机器人应用都会用到。
