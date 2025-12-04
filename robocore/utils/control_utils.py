"""
Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


def compute_steps_and_delay(
    speed_factor: float,
    T_default: float = 3.6,
    n_steps_ref: int = 120,
    min_steps: int = 50,
    max_steps: int = 500,
    min_delay: float = 0.01,
    max_delay: float = 0.1
):
    """
    根据速度因子同时调整插值步数和每步延迟，使得总时间约为 T_default / speed_factor。

    Returns:
        n_steps (int): 插值步数
        step_delay (float): 每步延迟（秒）
    """
    if speed_factor <= 0:
        raise ValueError("速度因子必须大于 0")

    # 插值步数反比于速度
    raw_steps = int(n_steps_ref / speed_factor)
    n_steps = max(min_steps, min(raw_steps, max_steps))

    # 根据实际步数计算单步时间
    total_time = T_default / speed_factor
    raw_delay = total_time / n_steps
    step_delay = max(min_delay, min(raw_delay, max_delay))

    return n_steps, step_delay



def check_and_clip_joint_limits(
    joints: List[float],
    joint_limits: List[Optional[Sequence[Optional[float]]]],
    joint_names: List[str] = None
) -> Tuple[List[float], List[Tuple[str, float, float]]]:
    """
    归一化关节角度并检查是否超过 joint_limits，超限的角度会被截断

    Args:
        joints: 当前关节角度列表
        joint_limits: 各关节限制范围，格式: [(-2.0, 2.0), (-3.0, 3.0), ...] 来自RobotModel.joint_limits
        joint_names: 对应的关节名顺序，可选

    Returns:
        clipped_joints: 修正后的角度列表
        violations: List of (joint_name, original_val, clipped_val)
    """
    clipped = joints.copy()
    violations = []

    # 生成默认关节名
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(len(joints))]
    
    for i, limit in enumerate(joint_limits):
        if i >= len(joints):
            break
            
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        raw_val = joints[i]
        
        # 归一化至 [-π, π]
        norm_val = (raw_val + np.pi) % (2 * np.pi) - np.pi
        
        if limit is None or (isinstance(limit, (tuple, list)) and (limit[0] is None or limit[1] is None)):
            # 无限制，只进行归一化
            clipped[i] = norm_val
            continue
            
        low, high = limit[0], limit[1]
        
        # 检查joint limit
        if norm_val < low:
            clipped[i] = low
            violations.append((name, raw_val, low))
        elif norm_val > high:
            clipped[i] = high
            violations.append((name, raw_val, high))
        else:
            clipped[i] = norm_val  # 如果未超限则返回归一后的数据

    return clipped, violations



def validate_joint_list(joints: Union[List[float], np.ndarray]):
    """
    验证关节角度输入是否合法（长度为6 + 数值）

    Args:
        joints: 关节角度序列

    Raises:
        TypeError, ValueError
    """
    if not isinstance(joints, (list, np.ndarray)):
        raise TypeError("关节角输入应为 list / ndarray")

    if len(joints) != 6:
        raise ValueError("关节角输入长度必须为 6")

    if not all(isinstance(x, (int, float)) for x in joints):
        raise ValueError("关节角输入应为数值型")