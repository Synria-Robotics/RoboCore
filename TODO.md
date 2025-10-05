# RoboCore Development Roadmap

基于机器人学基础知识体系的功能完整性评估和开发计划。

---

## 📊 当前完成度总览

| 模块 | 完整度 | 状态 |
|------|--------|------|
| 坐标变换 | 90% | ✅ 完整实现 (所有转换函数) |
| 机器人建模 | 80% | ✅ 基本完善 (已新增 MJCF 解析基础版) |
| 正运动学 | 95% | ✅ 完善 |
| 雅可比矩阵 | 90% | ✅ 完善 |
| 逆运动学 | 85% | ✅ 很强 |
| 奇异性分析 | 75% | ⚠️ 缺回避策略 |
| 工作空间分析 | 70% | ⚠️ 缺高级特性 |
| 轨迹规划 | 75% | ⚠️ 缺避障 |
| 轨迹插值 | 50% | ⚠️ 需加强重采样 |
| 动力学 | 0% | 🔴 完全缺失 |
| 控制器 | 0% | 🔴 完全缺失 |
| 碰撞检测 | 0% | ⚠️ 可选 |
| 测试覆盖 | 65% | ✅ 核心模块已测试 |

---

## 🔴 阶段一：基础完整性（P0 - 必须补）

### 1. 坐标变换与旋转表示 ✅ (已完成 90%)

状态概述：

- ✅ 核心运动学完成 (FK/IK/Jacobian 批量 & GPU 加速支持结构已预留)
- ✅ 旋转表示转换完成 (全部函数 + 双后端 + 批量)
- ✅ 四元数支持 (核心操作 + SLERP)
- ✅ 测试基础设施 (10 个测试文件, ~65% 覆盖率)
- ⚠️ 轨迹插值相关仍待补充 (重采样 / 平滑 / 缩放)
- 🔴 动力学与控制属于后续 Sprint 目标

#### 1.1 四元数支持 ✅ (95%)

文件: `robocore/transform/conversions.py`, `so3.py`

已实现：normalize / conjugate / inverse / multiply / 全部互转 / SLERP / 双后端 / 批处理

可选：SQUAD 插值

#### 1.2 旋转矩阵转换 ✅ (100%)

全部互转 + 批量 + 双后端已实现。

可选：rotation_distance / rotation_error / 12序列便捷封装（函数已支持 seq）。

#### 1.3 高级变换 ❌ 未实现 (可选 - P2优先级)
文件: `robocore/transform/advanced.py` (待新建)

- [ ] **对偶四元数 (Dual Quaternion)** - 同时表示旋转+平移
  - [ ] `DualQuaternion` 类
  - [ ] 转换函数 (SE(3) ↔ Dual Quaternion)
  - [ ] ScLERP 插值 (螺旋插值)
  - 应用场景: 机器人手眼标定、路径规划

- [ ] **旋量理论 (Screw Theory)** - 6D运动/力表示
  - [ ] `Twist` 类 - 瞬时速度 [v, ω]
  - [ ] `Wrench` 类 - 力/力矩 [f, τ]
  - [ ] `Adjoint(T)` - 伴随变换矩阵
  - [ ] 指数映射: twist → SE(3)
  - 应用场景: 现代机器人学理论基础

**优先级**: P2 (高级特性，按需实现)

---

### 2. 轨迹插值与时间同步

#### 2.1 时间重采样
**文件**: `robocore/planning/trajectory/interpolation.py` (新建)

- [ ] `resample_trajectory(t_orig, q_orig, control_freq, method='cubic')`
  - [ ] 稀疏轨迹 → 固定频率密集轨迹
  - [ ] 支持方法: `'linear'`, `'cubic'`, `'quintic'`
  - [ ] 返回 (t_new, q_new, qd_new, qdd_new)

- [ ] `resample_pose_trajectory(t_orig, poses, control_freq)`
  - [ ] 位姿轨迹重采样（位置+四元数）
  - [ ] 位置用样条，姿态用SLERP

- [ ] 批量/后端支持
  - [ ] NumPy实现
  - [ ] PyTorch实现（GPU加速）

#### 2.2 轨迹平滑
**文件**: `robocore/planning/trajectory/smoothing.py` (新建)

- [ ] `smooth_trajectory(t, q, method='moving_average', window=5)`
  - [ ] Moving average 平滑
  - [ ] Savitzky-Golay 滤波
  - [ ] Butterworth 低通滤波

- [ ] `smooth_trajectory_velocity(t, q)`
  - [ ] 平滑速度突变
  - [ ] 保持位置不变

#### 2.3 轨迹时间缩放
**文件**: `robocore/planning/trajectory/scaling.py` (新建)

- [ ] `time_scale_trajectory(t, q, scale_factor)`
  - [ ] `scale > 1` → 减速
  - [ ] `scale < 1` → 加速
  - [ ] 保持形状不变

- [ ] `match_trajectory_duration(t, q, target_duration)`
  - [ ] 调整轨迹到指定时长

---

## 🟡 阶段二：核心能力提升（P1 - 重要功能）

### 3. 速度与加速度运动学

#### 3.1 速度正运动学
**文件**: `robocore/kinematics/fk.py` (增强)

- [ ] `forward_velocity(robot_model, q, qd)` → (v_linear, v_angular)
  - [ ] 通过雅可比计算: `twist = J @ qd`
  - [ ] 返回 (v[3], ω[3])

- [ ] `forward_kinematics_all_links_velocity(robot_model, q, qd)`
  - [ ] 返回所有连杆的速度

#### 3.2 加速度正运动学
**文件**: `robocore/kinematics/fk.py` (增强)

- [ ] `forward_acceleration(robot_model, q, qd, qdd)` → (a_linear, a_angular)
  - [ ] `a = J @ qdd + dJ/dt @ qd`
  - [ ] 需要先实现 `jacobian_derivative`

#### 3.3 雅可比导数
**文件**: `robocore/kinematics/jacobian.py` (增强)

- [ ] `jacobian_derivative(robot_model, q, qd)` → dJ/dt [6×n]
  - [ ] 数值微分方法（快速）
  - [ ] 解析方法（精确，可选）

- [ ] 批量支持
  - [ ] `jacobian_derivative_batch(robot_model, q_batch, qd_batch)`

#### 3.4 零空间与任务空间
**文件**: `robocore/kinematics/jacobian.py` (增强)

- [ ] `null_space_projector(J)` → N [n×n]
  - [ ] `N = I - J_pinv @ J`
  - [ ] 用于冗余机器人优化

- [ ] `task_jacobian(robot_model, q, task_type)`
  - [ ] `task_type='position'` → J[:3, :]
  - [ ] `task_type='orientation'` → J[3:, :]
  - [ ] `task_type='full'` → J

---

### 4. 动力学模块

#### 4.1 逆动力学
**文件**: `robocore/dynamics/inverse_dynamics.py` (新建)

- [ ] `inverse_dynamics(robot_model, q, qd, qdd, external_forces=None)` → tau [n]
  - [ ] 递归Newton-Euler算法（高效）
  - [ ] 返回所需关节力矩

- [ ] 批量支持
  - [ ] `inverse_dynamics_batch(robot_model, q_batch, qd_batch, qdd_batch)`

- [ ] 后端支持
  - [ ] NumPy实现
  - [ ] PyTorch实现（GPU加速）

#### 4.2 正动力学
**文件**: `robocore/dynamics/forward_dynamics.py` (新建)

- [ ] `forward_dynamics(robot_model, q, qd, tau)` → qdd [n]
  - [ ] 给定力矩 → 计算加速度
  - [ ] ABA算法（Articulated Body Algorithm）

#### 4.3 质量矩阵
**文件**: `robocore/dynamics/mass_matrix.py` (新建)

- [ ] `mass_matrix(robot_model, q)` → M(q) [n×n]
  - [ ] 广义质量矩阵
  - [ ] 复合刚体算法（Composite Rigid Body Algorithm）

#### 4.4 科氏力和离心力
**文件**: `robocore/dynamics/coriolis.py` (新建)

- [ ] `coriolis_centrifugal(robot_model, q, qd)` → C(q,qd) [n]
  - [ ] 速度相关力

#### 4.5 重力项
**文件**: `robocore/dynamics/gravity.py` (新建)

- [ ] `gravity_vector(robot_model, q, gravity=[0, 0, -9.81])` → g(q) [n]
  - [ ] 重力补偿向量

#### 4.6 摩擦力模型（可选）
**文件**: `robocore/dynamics/friction.py` (新建)

- [ ] 粘性摩擦模型
- [ ] 库伦摩擦模型
- [ ] Stribeck效应

#### 4.7 集成方案（备选）
**备选方案**: 集成 Pinocchio 库

- [ ] 评估 Pinocchio 性能
- [ ] 封装 Pinocchio API 到 RoboCore 接口
- [ ] 保持接口一致性

---

### 5. 控制器模块

#### 5.1 位置控制器
**文件**: `robocore/controller/position_controller.py` (新建)

- [ ] `PDController` 类
  - [ ] `__init__(Kp, Kd)`
  - [ ] `compute_control(q_ref, qd_ref, q_actual, qd_actual)` → tau/q_cmd

- [ ] `PIDController` 类
  - [ ] `__init__(Kp, Ki, Kd)`
  - [ ] 积分抗饱和

#### 5.2 轨迹跟踪控制器
**文件**: `robocore/controller/trajectory_controller.py` (新建)

- [ ] `TrajectoryController` 类
  - [ ] 前馈 + PD反馈
  - [ ] `compute_control(q_ref, qd_ref, qdd_ref, q, qd)` → tau/q_cmd

#### 5.3 计算力矩控制
**文件**: `robocore/controller/computed_torque.py` (新建)

- [ ] `ComputedTorqueController` 类
  - [ ] 基于模型的控制
  - [ ] `tau = M(q)qdd_ref + C(q,qd) + g(q) + Kp(q_ref-q) + Kd(qd_ref-qd)`

#### 5.4 阻抗/导纳控制（可选）
**文件**: `robocore/controller/impedance_controller.py` (新建)

- [ ] `ImpedanceController` 类（力控制）
- [ ] `AdmittanceController` 类

---

### 6. 冗余逆运动学

#### 6.1 零空间优化
**文件**: `robocore/kinematics/ik.py` (增强)

- [ ] `inverse_kinematics_redundant(robot_model, target_pose, q0, secondary_objective, **kwargs)`
  - [ ] 主任务: 到达目标位姿
  - [ ] 次级任务: 优化manipulability/关节限位/能量

- [ ] 次级目标选项
  - [ ] `'manipulability'` - 最大化操作度
  - [ ] `'joint_limit'` - 远离关节限位
  - [ ] `'energy'` - 最小化关节运动
  - [ ] `'collision'` - 避碰（需碰撞检测）

#### 6.2 约束逆运动学
**文件**: `robocore/kinematics/ik.py` (增强)

- [ ] `inverse_kinematics_constrained(robot_model, target_pose, q0, constraints, **kwargs)`
  - [ ] `fixed_joints` - 固定某些关节
  - [ ] `task_mask` - 部分位姿约束 (如 [1,1,1,0,0,0] = 仅位置)
  - [ ] `joint_weights` - 关节权重

---

### 7. LQT轨迹规划迁移

#### 7.1 从AD-SDK迁移LQT
**文件**: `robocore/planning/trajectory/lqt_planner.py` (新建)

- [ ] 移植LQT轨迹优化算法
- [ ] 适配RoboCore接口
- [ ] NumPy/PyTorch双后端
- [ ] 批量优化支持

---

## 🟢 阶段三：高级特性（P2 - 扩展场景）

### 8. 解析逆运动学

#### 8.1 特定机器人闭式解
**文件**: `robocore/kinematics/ik_analytical.py` (新建)

- [ ] `solve_6dof_spherical_wrist(T, robot_params)` → List[q]
  - [ ] Puma 560
  - [ ] UR系列
  - [ ] ABB系列

- [ ] 返回所有解（最多8个）
- [ ] 解选择策略
  - [ ] 最近解
  - [ ] 最优manipulability解

---

### 9. 并联与树状机器人

#### 9.1 并联机器人
**文件**: `robocore/modeling/parallel_robot.py` (新建)

- [ ] `ParallelRobot` 类
- [ ] 闭环约束处理
- [ ] 并联FK/IK求解

#### 9.2 树状机器人
**文件**: `robocore/modeling/tree_robot.py` (新建)

- [ ] `TreeRobot` 类（人形、多臂）
- [ ] 分支运动学链
- [ ] `forward_kinematics_branch(branch_name, q)`

#### 9.3 移动机械臂
**文件**: `robocore/modeling/mobile_manipulator.py` (新建)

- [ ] 移动底座 + 机械臂
- [ ] 浮动基座（Floating Base）

---

### 10. 避障与路径规划

#### 10.1 采样规划器
**文件**: `robocore/planning/sampling_planner.py` (新建)

- [ ] RRT (Rapidly-exploring Random Tree)
- [ ] RRT* (优化版本)
- [ ] PRM (Probabilistic Roadmap)

#### 10.2 图搜索
**文件**: `robocore/planning/graph_planner.py` (新建)

- [ ] A* 算法
- [ ] Dijkstra 算法

---

### 11. 样条轨迹

#### 11.1 样条插值
**文件**: `robocore/planning/trajectory/spline.py` (新建)

- [ ] B-spline 插值
- [ ] Bezier 曲线
- [ ] NURBS（可选）

---

### 12. 碰撞检测

#### 12.1 自碰撞检测
**文件**: `robocore/collision/self_collision.py` (新建)

- [ ] 简化几何体（球、胶囊、凸包）
- [ ] AABB碰撞检测

#### 12.2 环境碰撞
**文件**: `robocore/collision/environment_collision.py` (新建)

- [ ] 机器人 vs 障碍物
- [ ] OBB碰撞检测

#### 12.3 距离计算
**文件**: `robocore/collision/distance.py` (新建)

- [ ] 最小距离查询
- [ ] 最近点对

#### 12.4 集成方案（备选）
**备选方案**: 集成专业碰撞库

- [ ] FCL (Flexible Collision Library)
- [ ] PyBullet 碰撞检测
- [ ] 封装统一接口

---

### 13. 奇异性回避

#### 13.1 奇异性分类
**文件**: `robocore/analysis/singularity_analyzer.py` (增强)

- [ ] 边界奇异 vs 内部奇异
- [ ] 位姿奇异 vs 配置奇异

#### 13.2 奇异性回避
**文件**: `robocore/analysis/singularity_avoidance.py` (新建)

- [ ] 梯度场避障
- [ ] 最小化条件数

---

### 14. 工作空间高级分析

#### 14.1 方向工作空间
**文件**: `robocore/analysis/workspace_analyzer.py` (增强)

- [ ] 给定位置，可达的所有姿态
- [ ] 姿态能力图（Orientation Capability Map）

#### 14.2 速度/力工作空间
**文件**: `robocore/analysis/workspace_analyzer.py` (增强)

- [ ] 速度工作空间
- [ ] 力/力矩工作空间
- [ ] 力椭球分析

---

## 📝 测试覆盖

### 需要新增的测试

#### 坐标变换测试
- [x] `test/unit/test_transform.py` - 变换测试 ✅ (已创建)
  - 涵盖四元数、旋转矩阵、RPY等转换
- [ ] `test/unit/test_rotation_conversions.py` - 全面转换测试 (可选)
  - 测试所有转换函数的数值精度
  - 测试批量处理和后端一致性

#### 插值测试
- [ ] `test/unit/test_interpolation.py` - 轨迹重采样/平滑/缩放
- [ ] `test/integration/test_trajectory_dense.py` - 端到端轨迹密集化

#### 动力学测试
- [ ] `test/unit/test_inverse_dynamics.py` - 逆动力学
- [ ] `test/unit/test_forward_dynamics.py` - 正动力学
- [ ] `test/unit/test_mass_matrix.py` - 质量矩阵

#### 控制器测试
- [ ] `test/unit/test_controllers.py` - PD/PID/计算力矩

#### 冗余IK测试
- [ ] `test/integration/test_redundant_ik.py` - 零空间优化

#### 已完成的测试 ✅
- [x] `test/unit/test_fk.py` - 正运动学测试
- [x] `test/unit/test_ik.py` - 逆运动学测试  
- [x] `test/unit/test_jacobian.py` - 雅可比测试
- [x] `test/unit/test_transform.py` - 变换测试
- [x] `test/unit/test_robot_model.py` - 机器人模型测试
- [x] `test/unit/test_backend.py` - 后端管理测试
- [x] `test/unit/test_config.py` - 配置系统测试
- [x] `test/integration/test_ik_accuracy.py` - IK精度集成测试
- [x] `test/integration/test_workspace.py` - 工作空间分析测试
- [x] `test/integration/test_singularity.py` - 奇异性分析测试

---

## 🎯 开发优先级建议

### Sprint 1 (已完成 90%)
1. ✅ **四元数支持** (95% - 所有核心函数已实现)
   - ✅ 所有转换函数 (quaternion ↔ matrix/rpy/axis-angle)
   - ✅ 基础操作 (normalize, conjugate, inverse, multiply)
   - ✅ SLERP插值 (通过scipy)
   - ⚠️ 可选: Quaternion类封装

2. ✅ **旋转表示转换** (100% - 完整实现)
   - ✅ 所有互转函数 (matrix/quaternion/rpy/axis-angle/euler)
   - ✅ 双后端支持 (NumPy/PyTorch)
   - ✅ 批量处理
   - ⚠️ 可选: 辅助函数 (rotation_distance, rotation_error)

3. ⚠️ **轨迹插值工具** (30% - 需补充)
   - ❌ 时间重采样
   - ❌ 轨迹平滑
   - ❌ 时间缩放

### Sprint 2 (Week 3-4): 核心能力 ❌ 未开始
4. ❌ 速度/加速度运动学
5. ❌ 雅可比导数
6. ❌ 动力学模块（核心）

### Sprint 3 (Week 5-6): 控制与优化 ❌ 未开始
7. ❌ 控制器模块
8. ❌ LQT迁移
9. ❌ 冗余IK

### Sprint 4+ (按需): 高级特性 ❌ 未开始
10. 解析IK
11. 并联/树状机器人
12. 避障规划
13. 碰撞检测

---

## 📌 设计原则

### RoboCore 定位
- ✅ **提供算法**：运动学、动力学、轨迹规划
- ✅ **离线使用**：仿真、分析、可视化
- ❌ **不做硬件IO**：实时控制留给SDK层

### 模块化设计
- 每个功能独立文件
- 清晰的接口定义
- NumPy/PyTorch双后端
- 批量处理支持

### 测试驱动
- 每个新功能必须有单元测试
- 保持>80%代码覆盖率
- CI/CD自动化验证

---

## 📚 参考资料

### 推荐书籍
- **Modern Robotics** (Lynch & Park) - 现代机器人学理论
- **Robotics: Modelling, Planning and Control** (Siciliano et al.) - 综合参考
- **Introduction to Robotics** (Craig) - 经典教材

### 参考库
- **Pinocchio** - 高性能动力学库
- **OMPL** - 运动规划库
- **FCL** - 碰撞检测库
- **Drake** - MIT机器人工具箱

---

**Last Updated**: 2025-10-05  
**Status**: 
**已完成模块概览**

✅ 核心运动学 (FK/IK/Jacobian)  
✅ 旋转表示转换 (所有转换函数)  
✅ 测试基础设施 (10 个测试文件)  

⚠️ 轨迹插值待补充 (重采样/平滑/缩放)  
🔴 动力学与控制待开发  
