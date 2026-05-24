#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bessica 7DOF dual-arm IK tests.

重点验证：
1. 单臂 7 自由度加载与 DOF 计数。
2. 标准末端 IK 闭环 (FK→IK→FK) 精度 (pinv / dls)。
3. 冗余（位置-only 任务 + nullspace 关节居中）行为：nullspace 应缩小关节偏离中心的度量。
4. 中间链路 (target_link) 局部 IK：对中间 link 的局部姿态/位置求解。

说明：
 - Bessica-D URDF 包含双臂；通过指定 end_link 让 RobotModel 只走一侧臂链。
 - 手爪 finger joints 在 URDF 中为 fixed，不计入 DOF；故每臂 7 个 revolute DOF。
 - 位置-only 任务需要放宽 orientation 容差，否则 IK 解算器仍会因未收敛姿态而判失败。
"""

import math
import numpy as np
import pytest

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
import robocore


LEFT_END = 'left_arm_gripper_left_finger'
RIGHT_END = 'right_arm_gripper_left_finger'


def random_q_in_limits(model, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros(model.num_dof)
    chain_indices = model._get_joint_indices(model.base_link, model.end_link)
    for idx in chain_indices:
        js = model.joint_list[idx]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        # 取中间 50% 区间随机，避免极限导致条件数更差
        mid = 0.5 * (lo + hi)
        span = 0.25 * (hi - lo)
        q[idx] = rng.uniform(mid - span, mid + span)
    return q


@pytest.fixture(scope="module")
def left_arm(bessica_urdf_path):
    return RobotModel(bessica_urdf_path, end_link=LEFT_END)


@pytest.fixture(scope="module")
def right_arm(bessica_urdf_path):
    return RobotModel(bessica_urdf_path, end_link=RIGHT_END)


class TestBessicaSevenDOF:
    def test_dof_counts(self, left_arm, right_arm):
        # Check chain DOF (not unified config space)
        left_indices = left_arm._get_joint_indices(left_arm.base_link, left_arm.end_link)
        right_indices = right_arm._get_joint_indices(right_arm.base_link, right_arm.end_link)
        assert len(left_indices) == 7, "Left arm should have 7 DOF"
        assert len(right_indices) == 7, "Right arm should have 7 DOF"

    @pytest.mark.parametrize("method", ["pinv", "dls"])  # transpose 通常也行，可按需加入
    def test_fk_ik_fk_closure(self, left_arm, method):
        q_target = random_q_in_limits(left_arm, seed=42)
        T_target = forward_kinematics(left_arm, q_target, return_end=True)
        q_init = random_q_in_limits(left_arm, seed=43)

        res = inverse_kinematics(
            left_arm, T_target, q_init,
            method=method,
            max_iters=180, pos_tol=1e-4, ori_tol=1e-4
        )
        assert 'success' in res
        assert len(res['q']) == left_arm.num_dof
        if res['success']:
            T_res = forward_kinematics(left_arm, res['q'], return_end=True)
            # 位置误差
            pos_err = np.linalg.norm(T_target[:3, 3] - T_res[:3, 3])
            # 姿态误差（轴角幅值）
            R_delta = T_target[:3, :3].T @ T_res[:3, :3]
            trace = np.trace(R_delta)
            angle = math.acos(np.clip((trace - 1) / 2, -1, 1))
            assert pos_err < 1e-3, f"Position closure error {pos_err}"
            assert angle < 1e-2, f"Orientation closure error {angle}"

    def test_redundant_position_only_nullspace(self, left_arm):
        """位置-only 任务下测试 nullspace 关节居中效果（7DOF 冗余）。"""
        q_target = random_q_in_limits(left_arm, seed=11)
        T_target = forward_kinematics(left_arm, q_target, return_end=True)
        # 仅取位置部分作为任务目标（行掩码）——用原姿态以便 orientation 不影响（但我们放宽 ori_tol）
        q_init = random_q_in_limits(left_arm, seed=99)

        # 基准：无 nullspace
        base = inverse_kinematics(
            left_arm, T_target, q_init,
            method='pinv',
            row_mask=[1,1,1,0,0,0],
            max_iters=160, pos_tol=1e-4, ori_tol=1e2,  # 放宽姿态容差
            nullspace_gain=0.0,
            joint_centering=True
        )
        # 启用 nullspace 居中
        with_ns = inverse_kinematics(
            left_arm, T_target, q_init,
            method='pinv',
            row_mask=[1,1,1,0,0,0],
            max_iters=160, pos_tol=1e-4, ori_tol=1e2,
            nullspace_gain=0.4,
            joint_centering=True,
            joint_center_gain=0.3
        )

        assert 'q' in base and 'q' in with_ns
        # 计算与关节中心的平均偏差
        def avg_center_offset(model, q):
            acc = 0.0
            chain_indices = model._get_joint_indices(model.base_link, model.end_link)
            for idx in chain_indices:
                js = model.joint_list[idx]
                lo, hi = -1.0, 1.0
                if js.limit:
                    if js.limit[0] is not None: lo = js.limit[0]
                    if js.limit[1] is not None: hi = js.limit[1]
                center = 0.5 * (lo + hi)
                acc += abs(q[idx] - center) / max(1e-9, (hi - lo))
            return acc / len(chain_indices)
        off_base = avg_center_offset(left_arm, np.asarray(base['q']))
        off_ns = avg_center_offset(left_arm, np.asarray(with_ns['q']))
        # nullspace 居中应不劣于基准（允许极小浮动）
        assert off_ns <= off_base + 1e-4, f"Nullspace centering ineffective: {off_ns} > {off_base}"
        # 位置误差都应满足要求
        assert base.get('success') or with_ns.get('success'), "At least one solve should converge position-only"

    def test_partial_link_target(self, left_arm):
        """测试中间链路 (target_link) 局部 IK 闭环。"""
        # 选择中间 link，例如 link5（确保链路存在）
        target_link = 'left_arm_link5'
        # 构造目标：取随机配置的该 link 位姿
        q_seed = random_q_in_limits(left_arm, seed=7)
        # 直接利用整链 FK 拿末端，再用 jacobian 早停逻辑？——简单起见手写局部遍历或借用局部 IK：
        # 这里使用先完整 FK 然后再通过局部 IK 验证（姿态一致→成功更容易）
        # 为稳定，写一个小的局部 forward traversal：
        from robocore.kinematics.jacobian import jacobian  # noqa: F401 (ensure import side effects if any)
        # 复用数值版逻辑（简化：复制 numpy 早停策略）
        def fk_until(model, q, link):
            chain_indices = model._get_joint_indices(model.base_link, model.end_link)
            q_map = {model.joint_list[idx].name: q[idx] for idx in chain_indices}
            import math as _m
            T = np.eye(4)
            for urdf_joint in model._chain_joints:  # type: ignore[attr-defined]
                r,p,y = urdf_joint.origin_rpy
                sr,cr = _m.sin(r), _m.cos(r)
                sp,cp = _m.sin(p), _m.cos(p)
                sy,cy = _m.sin(y), _m.cos(y)
                R_o = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                                [-sp,   cp*sr,             cp*cr           ]])
                T_o = np.eye(4); T_o[:3,:3]=R_o; T_o[:3,3]=np.array(urdf_joint.origin_xyz)
                T_joint_origin = T @ T_o
                R_m = np.eye(3); t_m = np.zeros(3)
                if urdf_joint.joint_type == 'revolute':
                    th = q_map.get(urdf_joint.name, 0.0)
                    ax = np.array(urdf_joint.axis); n = np.linalg.norm(ax) or 1.0; ax = ax/n
                    ct = _m.cos(th); st = _m.sin(th); vt = 1-ct
                    axx, axy, axz = ax
                    R_m = np.array([[ct+axx*axx*vt, axx*axy*vt-axz*st, axx*axz*vt+axy*st],
                                    [axy*axx*vt+axz*st, ct+axy*axy*vt, axy*axz*vt-axx*st],
                                    [axz*axx*vt-axy*st, axz*axy*vt+axx*st, ct+axz*axz*vt]])
                elif urdf_joint.joint_type == 'prismatic':
                    d = q_map.get(urdf_joint.name, 0.0)
                    ax = np.array(urdf_joint.axis); n = np.linalg.norm(ax) or 1.0; ax = ax/n
                    t_m = ax * d
                T_m = np.eye(4); T_m[:3,:3]=R_m; T_m[:3,3]=t_m
                T = T_joint_origin @ T_m
                if urdf_joint.child == link:
                    return T
            raise ValueError('link not found')

        T_link = fk_until(left_arm, q_seed, target_link)
        q_init = random_q_in_limits(left_arm, seed=13)
        res = inverse_kinematics(
            left_arm, T_link, q_init,
            method='pinv',
            target_link=target_link,
            max_iters=150, pos_tol=1e-4, ori_tol=1e-4
        )
        assert 'success' in res
        assert res['success'], f"Partial link IK failed (err={res.get('err_norm')})"


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__, '-v'])
