# 性能基准测试工具

本目录包含用于测试 RoboCore 库性能的基准测试脚本。

## 可用的基准测试

### 1. `benchmark_parallel_fk_ik.py` - NumPy vs PyTorch 并行性能对比

对比 NumPy 和 PyTorch 后端在大规模并行 FK/IK 计算时的性能。

**功能**:
- 批量正运动学 (FK) 性能测试
- 批量逆运动学 (IK) 性能测试
- CPU 和 GPU 支持（CUDA/MPS）
- 自动生成性能对比报告

**基本用法**:

```bash
# 快速测试 (100 个样本，仅 FK)
python examples/benchmark_parallel_fk_ik.py --batch-size 100 --fk-only

# 完整测试 (FK + IK)
python examples/benchmark_parallel_fk_ik.py --batch-size 200

# 大批量测试
python examples/benchmark_parallel_fk_ik.py --batch-size 1000

# GPU 测试 (CUDA)
python examples/benchmark_parallel_fk_ik.py --batch-size 1000 --device cuda

# GPU 测试 (Apple Silicon MPS)
python examples/benchmark_parallel_fk_ik.py --batch-size 1000 --device mps

# 只测试 NumPy
python examples/benchmark_parallel_fk_ik.py --numpy-only --batch-size 500
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 测试样本数量 | 1000 |
| `--fk-only` | 只测试 FK | False |
| `--ik-only` | 只测试 IK | False |
| `--ik-iters` | IK 最大迭代次数 | 100 |
| `--warmup` | 预热迭代次数 | 5 |
| `--device` | PyTorch 设备 (cpu/cuda/mps) | cpu |
| `--numpy-only` | 只测试 NumPy | False |
| `--seed` | 随机种子 | 42 |

**输出示例**:

```
================================================================================
                      Parallel FK/IK Performance Benchmark                      
================================================================================

Configuration:
  URDF: robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf
  End Link: tool0
  Batch Size: 100
  Warmup Iterations: 3

--------------------------------------------------------------------------------
Forward Kinematics (FK) Performance
--------------------------------------------------------------------------------

Backend              Batch Size   Total Time      Avg Time        Throughput     
-------------------- ------------ --------------- --------------- ---------------
numpy                100              0.0113 s        0.1126 ms      8883.72 it/s
torch_cpu            100              0.1101 s        1.1014 ms       907.93 it/s

Backend              Speedup vs NumPy    
-------------------- --------------------
numpy                      1.00x
torch_cpu                  0.10x
```

### 2. `demo_fk_ik_jacobian.py` - FK/IK/Jacobian 综合演示

计算指定关节角的 FK、IK 和 Jacobian。

**用法**:

```bash
# 零位配置
python examples/demo_fk_ik_jacobian.py

# 指定关节角（弧度）
python examples/demo_fk_ik_jacobian.py --joints 0.5 -0.3 1.2 0.8 -0.5 1.5

# 指定关节角（角度）
python examples/demo_fk_ik_jacobian.py --joints-deg 30 -15 60 45 -30 90

# 随机关节角
python examples/demo_fk_ik_jacobian.py --random --seed 42

# 使用不同的末端链接
python examples/demo_fk_ik_jacobian.py --end-link Link7 --random
```

**输出包括**:
- 末端执行器位置 (x, y, z)
- 欧拉角 (roll, pitch, yaw)
- **四元数 (xyzw 顺序)**
- 旋转矩阵
- IK 求解结果
- Jacobian 矩阵及奇异值分析

## 性能测试结果总结

### NumPy vs PyTorch (CPU) - 当前实现

| 任务 | NumPy | PyTorch CPU | NumPy 优势 |
|------|-------|-------------|-----------|
| FK (单样本) | 0.11 ms | 1.10 ms | **10x 更快** |
| IK (单样本) | 14.4 ms | 76.0 ms | **5x 更快** |

### 为什么 NumPy 当前更快？

1. **逐样本处理**: 当前实现未充分利用 PyTorch 批量并行能力
2. **张量开销**: 每个样本都有 NumPy↔Torch 转换成本
3. **自动微分**: PyTorch 自动微分框架有额外开销

### PyTorch 的潜在优势（待实现）

当实现真正的批量并行 FK/IK 后，预期性能：

| 场景 | 批次大小 | NumPy CPU | PyTorch GPU | 预期加速 |
|------|---------|-----------|-------------|----------|
| 批量 FK | 1000 | ~113 ms | ~10 ms | **10-20x** |
| 批量 IK | 1000 | ~14 s | ~1 s | **10-15x** |

## 推荐使用策略

### 使用 NumPy ✅

- ✅ 实时单样本计算（机器人控制）
- ✅ 中小规模批处理（< 1000 样本）
- ✅ CPU 环境
- ✅ 生产部署（轻量级依赖）

### 使用 PyTorch 🔥

- 🔥 大规模批量推理（> 1000 样本）
- 🔥 需要梯度的优化问题
- 🔥 神经网络集成
- 🔥 GPU 硬件加速

## 四元数约定

⚠️ **重要**: RoboCore 统一使用 **xyzw 顺序** 表示四元数：

```python
quaternion = [qx, qy, qz, qw]  # xyzw 顺序
```

这与以下框架一致：
- ✅ scipy.spatial.transform.Rotation
- ✅ ROS (Robot Operating System)
- ✅ PyBullet
- ✅ Unity/Unreal Engine

详见: [QUATERNION_CONVENTION.md](../docs/QUATERNION_CONVENTION.md)

## 相关文档

- [NumPy vs Torch 性能对比报告](../docs/NUMPY_VS_TORCH_PERFORMANCE.md)
- [四元数约定文档](../docs/QUATERNION_CONVENTION.md)
- [IK 性能优化报告](../docs/IK_PERFORMANCE_REPORT.md)

## 贡献

欢迎贡献改进：
- 实现真正的批量 FK/IK（PyTorch）
- GPU 优化
- 更多基准测试场景
- 性能分析工具
