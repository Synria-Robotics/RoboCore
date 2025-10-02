"""FK/IK 闭环验证示例 (更新版)

步骤:
1. 随机采样一个可达关节配置 q*.
2. 计算其末端位姿 T* (FK).
3. 以零位或给定初值 q0 作为起点，用多种 IK 方法 + 数值 / 解析 Jacobian 恢复 q*.
4. 对比收敛误差与迭代次数，验证 IK 正确性与解析雅克比一致性。

已适配新的 `IKSolverNumPy` 接口：不再在 solve() 里使用 `max_iters / tol / damping` 参数，
而是在构造函数里指定 `max_iters`，收敛阈值由 `pos_tol`, `ori_tol` 控制，阻尼自适应。
"""

from pathlib import Path
import random
import math
from robocore import RobotModel, IKSolver  # IKSolver 是 IKSolverNumPy alias
from time import perf_counter
from robocore.utils.beauty_logger import beauty_print


def rotation_matrix(T):
    """Extract 3x3 rotation from 4x4 pose."""
    return [row[:3] for row in T[:3]]


def orientation_angle_deg(R1, R2):
    """Compute orientation difference in degrees."""
    # R_diff = R1^T @ R2
    R1T = [[R1[j][i] for j in range(3)] for i in range(3)]
    R_diff = [[sum(R1T[i][k] * R2[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    trace = R_diff[0][0] + R_diff[1][1] + R_diff[2][2]
    angle_rad = math.acos(max(-1, min(1, (trace - 1) / 2)))
    return math.degrees(angle_rad)


def sample_q(model: RobotModel, scale: float = 0.6):
    qs = [0.0] * model.dof()
    for js in model._actuated:  # type: ignore[attr-defined]
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * scale
        qs[js.index] = random.uniform(mid - span, mid + span)
    return qs


def pose_to_str(T):
    return "[" + ", ".join("[" + ", ".join(f"{v: .4f}" for v in row) + "]" for row in T) + "]"


def main():
    random.seed(42)
    base = Path(__file__).resolve().parents[1]
    urdf = base / "robocore" / "assets" / "robot" / "urdf" / "Alicia-D_v5_4" / "alicia_duo_with_gripper.urdf"
    model = RobotModel(str(urdf), end_link="tool0")

    # 目标姿态
    q_true = sample_q(model)
    fk_true = model.forward_kinematics(q_true)["end"]
    beauty_print("FK/IK 闭环验证", type="module")
    beauty_print("目标关节 q*: " + str([round(x, 4) for x in q_true]))
    beauty_print("目标末端位姿 (4x4):")
    for row in fk_true:
        beauty_print("  " + str([f"{v: .6f}" for v in row]))

    # 初始姿态（可改成随机 / 多起点）
    q0 = [0.0] * model.dof()

    methods = ["dls", "pinv", "transpose"]
    jacobian_modes = [(False, "numeric"), (True, "analytic")]

    for m in methods:
        for use_ana, tag in jacobian_modes:
            beauty_print(f"Method={m}  Jacobian={tag}", type="module")
            solver = IKSolver(
                model,
                max_iters=120,
                pos_tol=5e-4,   # 可根据需要调紧/放宽
                ori_tol=5e-3,
            )
            t0 = perf_counter()
            res = solver.solve(
                fk_true,
                q0,
                method=m,
                use_analytic_jacobian=use_ana,
                use_central_diff=True,
                pos_weight=1.0,
                ori_weight=1.0,
            )
            dt_ms = (perf_counter() - t0) * 1000.0
            fk_rec = model.forward_kinematics(res["q"])["end"]
            pos_diff = sum((fk_true[i][3] - fk_rec[i][3]) ** 2 for i in range(3)) ** 0.5
            ang_deg = orientation_angle_deg(rotation_matrix(fk_true), rotation_matrix(fk_rec))
            status_type = "success" if res.get("success") else "warning"
            beauty_print(
                f"success={res.get('success')} iters={res.get('iters')} pos_err={pos_diff:.2e} ang_err={ang_deg:.2e}deg total_err={res.get('err_norm',0):.2e} time={dt_ms:.2f}ms",
                type=status_type,
            )
            beauty_print("解 q: " + str([round(x, 4) for x in res["q"]]))

    beauty_print("真实 q*", type="module")
    beauty_print(str([round(x, 4) for x in q_true]))


if __name__ == "__main__":
    main()
