"""Bessica 多末端（双臂）示例：一次加载，派生左右臂链。

演示：
 1. 只解析一次 URDF 构建基础模型（不指定 end_link → 选择最长链）。
 2. 使用 available_leaf_links() 找到所有末端候选。
 3. 通过 spawn_chain(end_link) 快速生成左臂 / 右臂专用链视图（不重复解析）。
 4. 分别计算左右臂末端 FK 与 Jacobian，并构造一个简单的双臂块对角 Jacobian。

运行:
    python examples/kinematics/demo_bessica_multi_end.py
"""
from __future__ import annotations
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print
from robocore.kinematics.jacobian import jacobian

URDF_REL = "assets/robot/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"


def main():
    beauty_print("== Bessica Multi-End Demo (single parse) ==")
    base_model = RobotModel(get_robocore_path(URDF_REL))  # 不指定 end_link
    base_model.print_tree(show_joints=False)
    

    leaves = base_model.available_leaf_links()
    beauty_print(f"Leaf links detected: {leaves}")
    # 选择左右末端（粗匹配名字）
    left_end = next(l for l in leaves if 'left_arm_gripper_left_finger' in l)
    right_end = next(l for l in leaves if 'right_arm_gripper_left_finger' in l)

    left_chain = base_model.spawn_chain(left_end)
    right_chain = base_model.spawn_chain(right_end)

    beauty_print(f"Left chain DOF={left_chain.num_dof()} end={left_chain.end_link}")
    beauty_print(f"Right chain DOF={right_chain.num_dof()} end={right_chain.end_link}")

    q_left = left_chain.random_q()
    q_right = right_chain.random_q()

    # FK
    T_left = left_chain.fk(q_left, backend='numpy', return_end=True)
    T_right = right_chain.fk(q_right, backend='numpy', return_end=True)
    print("Left EE pose:\n", T_left)
    print("Right EE pose:\n", T_right)

    # Jacobian (NumPy)
    J_left = jacobian(left_chain, q_left, backend='numpy')
    J_right = jacobian(right_chain, q_right, backend='numpy')
    beauty_print("Assemble block-diagonal 12x14 dual-arm Jacobian (NumPy)")
    J_block = np.zeros((12, left_chain.num_dof() + right_chain.num_dof()))
    J_block[0:6, 0:left_chain.num_dof()] = J_left
    J_block[6:12, left_chain.num_dof():] = J_right
    print("J_block shape:", J_block.shape)

    # Rank / redundancy
    rank = np.linalg.matrix_rank(J_block)
    null_dim = J_block.shape[1] - rank
    beauty_print(f"Rank={rank}, Nullspace dim={null_dim}")

    # Torch 可选
    try:
        import torch  # noqa
        J_left_t = jacobian(left_chain, q_left, backend='torch')
        J_right_t = jacobian(right_chain, q_right, backend='torch')
        J_block_t = np.zeros_like(J_block)
        J_block_t[0:6, 0:left_chain.num_dof()] = J_left_t.detach().cpu().numpy()
        J_block_t[6:12, left_chain.num_dof():] = J_right_t.detach().cpu().numpy()
        beauty_print("Torch Jacobians assembled (converted to numpy)")
    except Exception:
        beauty_print("Torch unavailable, skipped torch Jacobian section", type="warning")

    beauty_print("Demo complete.")


if __name__ == '__main__':
    main()
