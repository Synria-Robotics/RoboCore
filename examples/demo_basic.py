"""Basic FK and IK demonstration.

Shows the most straightforward usage of RoboCore for forward and inverse kinematics.

Usage:
    python examples/demo_basic.py
"""
from pathlib import Path
import numpy as np
from robocore import RobotModel
from robocore.kinematics import forward_kinematics, inverse_kinematics
from robocore.utils.beauty_logger import beauty_print


def main():
    # Load robot model
    base = Path(__file__).resolve().parents[1]
    urdf = base / "robocore" / "assets" / "robot" / "urdf" / "Alicia-D_v5_4" / "alicia_duo_with_gripper.urdf"
    model = RobotModel(str(urdf), end_link="tool0")
    
    beauty_print(f"Basic FK/IK Demo: {model.name}", type="module")
    beauty_print(f"DOF: {model.dof()}, Joints: {', '.join(model.joint_names())}", type="info")
    
    # === Forward Kinematics ===
    beauty_print("\n[1] Forward Kinematics", type="module")
    q = [0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.0][:model.dof()]
    beauty_print(f"Input config: {[f'{x:+.3f}' for x in q]}")
    
    poses = forward_kinematics(model, q, backend='numpy')
    T_end = poses['end']
    beauty_print("\nEnd-effector pose:")
    for row in T_end:
        print(f"  [{', '.join(f'{x:+.4f}' for x in row)}]")
    
    # === Inverse Kinematics ===
    beauty_print("\n[2] Inverse Kinematics", type="module")
    beauty_print("Target: use the FK result from above")
    
    q0 = np.zeros(model.dof())
    result = inverse_kinematics(
        model,
        T_end,
        q0,
        backend='numpy',
        method='pinv',
        pos_tol=1e-4,
        ori_tol=1e-4,
    )
    
    beauty_print(f"Success: {result['success']}", type="success" if result['success'] else "error")
    beauty_print(f"Iterations: {result['iters']}")
    beauty_print(f"Position error: {result.get('pos_err', 0):.3e} m")
    beauty_print(f"Orientation error: {result.get('ori_err', 0):.3e} rad")
    beauty_print(f"Solved config: {[f'{x:+.3f}' for x in result['q']]}")
    
    # Verify closure
    poses_verify = forward_kinematics(model, result['q'], backend='numpy')
    T_verify = poses_verify['end']
    pos_diff = np.linalg.norm(np.array(T_end[:3, 3]) - np.array(T_verify[:3, 3]))
    beauty_print(f"\nClosure verification - Position difference: {pos_diff:.3e} m", type="info")
    
    beauty_print("\n✓ Basic demo complete", type="success")


if __name__ == "__main__":
    main()
