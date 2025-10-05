"""诊断 NumPy 和 Torch IK 求解器之间的差异。

分析可能的原因：
1. 数值精度差异（float32 vs float64）
2. Jacobian 计算差异
3. 矩阵求逆/伪逆算法差异
4. 初始化和默认参数差异
"""
import numpy as np
import torch
import os
from pathlib import Path
from robocore import RobotModel
from robocore.kinematics.ik_utils.ik_solver_numpy import IKSolverNumPy
from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
from robocore.utils.beauty_logger import beauty_print


def compare_single_case(model, target_pose, q0, method='pinv'):
    """详细对比单个测试用例。"""
    print("\n" + "="*80)
    beauty_print(f"Case Analysis: method={method}", type="module")
    print("="*80)
    
    # NumPy solver
    solver_np = IKSolverNumPy(model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4)
    
    # Torch solvers with different dtypes
    solver_torch_fp64 = IKSolverTorch(
        model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4,
        device=torch.device('cpu'), dtype=torch.float64
    )
    solver_torch_fp32 = IKSolverTorch(
        model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4,
        device=torch.device('cpu'), dtype=torch.float32
    )
    
    # Solve with all three
    result_np = solver_np.solve(target_pose, q0, method=method, use_analytic_jacobian=True)
    result_torch_fp64 = solver_torch_fp64.solve(target_pose, q0, method=method)
    result_torch_fp32 = solver_torch_fp32.solve(target_pose, q0, method=method)
    
    # Print comparison
    print(f"\n{'Backend':<20} | {'Success':<8} | {'Iters':<6} | {'Pos Err':<12} | {'Ori Err':<12}")
    print("-" * 80)
    
    for name, res in [
        ('NumPy (float64)', result_np),
        ('Torch (float64)', result_torch_fp64),
        ('Torch (float32)', result_torch_fp32),
    ]:
        success = "✓" if res.get('success') else "✗"
        iters = res.get('iters', 0)
        pos_err = res.get('pos_err', np.nan)
        ori_err = res.get('ori_err', np.nan)
        print(f"{name:<20} | {success:<8} | {iters:<6} | {pos_err:<12.6e} | {ori_err:<12.6e}")
    
    # Detailed Jacobian comparison
    print("\n" + "-"*80)
    beauty_print("Jacobian Comparison at q0", type="info")
    print("-"*80)
    
    # NumPy Jacobian
    J_np = solver_np.jacobian_solver.solve(q0, method="analytic")
    
    # Torch Jacobian (fp64)
    q0_torch = torch.tensor(q0, dtype=torch.float64, device=torch.device('cpu'))
    J_torch_fp64 = solver_torch_fp64.jacobian_solver.solve(q0_torch, method="analytic")
    
    # Torch Jacobian (fp32)
    q0_torch_fp32 = torch.tensor(q0, dtype=torch.float32, device=torch.device('cpu'))
    J_torch_fp32 = solver_torch_fp32.jacobian_solver.solve(q0_torch_fp32, method="analytic")
    
    J_diff_fp64 = np.abs(J_np - J_torch_fp64.cpu().numpy())
    J_diff_fp32 = np.abs(J_np - J_torch_fp32.cpu().numpy())
    
    print(f"NumPy Jacobian shape: {J_np.shape}")
    print(f"Jacobian diff (Torch fp64 vs NumPy): max={J_diff_fp64.max():.6e}, mean={J_diff_fp64.mean():.6e}")
    print(f"Jacobian diff (Torch fp32 vs NumPy): max={J_diff_fp32.max():.6e}, mean={J_diff_fp32.mean():.6e}")
    
    # Condition number comparison
    print("\n" + "-"*80)
    beauty_print("Condition Number Analysis", type="info")
    print("-"*80)
    
    cond_np = np.linalg.cond(J_np)
    cond_torch_fp64 = torch.linalg.cond(J_torch_fp64).item()
    cond_torch_fp32 = torch.linalg.cond(J_torch_fp32).item()
    
    print(f"Condition number (NumPy):        {cond_np:.6e}")
    print(f"Condition number (Torch fp64):   {cond_torch_fp64:.6e}")
    print(f"Condition number (Torch fp32):   {cond_torch_fp32:.6e}")
    
    # Pseudoinverse comparison
    if method == 'pinv':
        print("\n" + "-"*80)
        beauty_print("Pseudoinverse Comparison", type="info")
        print("-"*80)
        
        # NumPy pinv
        J_pinv_np = np.linalg.pinv(J_np)
        
        # Torch pinv (fp64)
        J_pinv_torch_fp64 = torch.linalg.pinv(J_torch_fp64).cpu().numpy()
        
        # Torch pinv (fp32)
        J_pinv_torch_fp32 = torch.linalg.pinv(J_torch_fp32).cpu().numpy()
        
        pinv_diff_fp64 = np.abs(J_pinv_np - J_pinv_torch_fp64)
        pinv_diff_fp32 = np.abs(J_pinv_np - J_pinv_torch_fp32)
        
        print(f"Pinv diff (Torch fp64 vs NumPy): max={pinv_diff_fp64.max():.6e}, mean={pinv_diff_fp64.mean():.6e}")
        print(f"Pinv diff (Torch fp32 vs NumPy): max={pinv_diff_fp32.max():.6e}, mean={pinv_diff_fp32.mean():.6e}")
    
    return result_np, result_torch_fp64, result_torch_fp32


def batch_analysis(model, n_samples=20):
    """批量分析成功率和精度差异。"""
    print("\n" + "="*80)
    beauty_print(f"Batch Analysis: {n_samples} samples", type="module")
    print("="*80)
    
    rng = np.random.default_rng(77)
    
    # Generate test cases
    test_cases = []
    for _ in range(n_samples):
        q_rand = np.zeros(model.dof())
        for js in model._actuated:
            lo, hi = -1.0, 1.0
            if js.limit and js.limit[0] is not None and js.limit[1] is not None:
                lo, hi = js.limit[0], js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.4 * (hi - lo)
            q_rand[js.index] = float(rng.uniform(mid - span, mid + span))
        
        pose = model.forward_kinematics(q_rand.tolist())['end']
        test_cases.append((pose, np.zeros(model.dof())))
    
    # Test with both methods
    for method in ['pinv', 'dls']:
        print(f"\n{'-'*80}")
        print(f"Method: {method}")
        print(f"{'-'*80}")
        
        stats = {
            'numpy': {'success': [], 'pos_err': [], 'ori_err': []},
            'torch_fp64': {'success': [], 'pos_err': [], 'ori_err': []},
            'torch_fp32': {'success': [], 'pos_err': [], 'ori_err': []},
        }
        
        for pose, q0 in test_cases:
            solver_np = IKSolverNumPy(model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4)
            solver_torch_fp64 = IKSolverTorch(
                model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4,
                device=torch.device('cpu'), dtype=torch.float64
            )
            solver_torch_fp32 = IKSolverTorch(
                model, max_iters=120, pos_tol=1e-4, ori_tol=1e-4,
                device=torch.device('cpu'), dtype=torch.float32
            )
            
            res_np = solver_np.solve(pose, q0, method=method, use_analytic_jacobian=True)
            res_torch_fp64 = solver_torch_fp64.solve(pose, q0, method=method)
            res_torch_fp32 = solver_torch_fp32.solve(pose, q0, method=method)
            
            stats['numpy']['success'].append(res_np.get('success', False))
            stats['numpy']['pos_err'].append(res_np.get('pos_err', np.nan))
            stats['numpy']['ori_err'].append(res_np.get('ori_err', np.nan))
            
            stats['torch_fp64']['success'].append(res_torch_fp64.get('success', False))
            stats['torch_fp64']['pos_err'].append(res_torch_fp64.get('pos_err', np.nan))
            stats['torch_fp64']['ori_err'].append(res_torch_fp64.get('ori_err', np.nan))
            
            stats['torch_fp32']['success'].append(res_torch_fp32.get('success', False))
            stats['torch_fp32']['pos_err'].append(res_torch_fp32.get('pos_err', np.nan))
            stats['torch_fp32']['ori_err'].append(res_torch_fp32.get('ori_err', np.nan))
        
        # Print statistics
        print(f"\n{'Backend':<20} | {'Success Rate':<15} | {'Mean Pos Err':<15} | {'Mean Ori Err':<15}")
        print("-" * 80)
        
        for name in ['numpy', 'torch_fp64', 'torch_fp32']:
            s = stats[name]
            success_rate = np.mean(s['success']) * 100
            mean_pos = np.nanmean(s['pos_err'])
            mean_ori = np.nanmean(s['ori_err'])
            
            backend_label = {
                'numpy': 'NumPy (float64)',
                'torch_fp64': 'Torch (float64)',
                'torch_fp32': 'Torch (float32)',
            }[name]
            
            print(f"{backend_label:<20} | {success_rate:>6.1f}%        | {mean_pos:>13.6e} | {mean_ori:>13.6e}")


def main():
    # Load model
    base = Path(__file__).resolve().parents[1]
    urdf = os.path.join(base, "../robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf")
    model = RobotModel(str(urdf), end_link='tool0')
    
    beauty_print(f"IK Diagnostic Tool: {model.name} ({model.dof()} DOF)", type="module")
    
    # Single case analysis
    rng = np.random.default_rng(42)
    q_test = np.zeros(model.dof())
    for js in model._actuated:
        lo, hi = -1.0, 1.0
        if js.limit and js.limit[0] is not None and js.limit[1] is not None:
            lo, hi = js.limit[0], js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.4 * (hi - lo)
        q_test[js.index] = float(rng.uniform(mid - span, mid + span))
    
    pose_test = model.forward_kinematics(q_test.tolist())['end']
    q0_test = np.zeros(model.dof())
    
    # Compare both methods
    for method in ['pinv', 'dls']:
        compare_single_case(model, pose_test, q0_test, method=method)
    
    # Batch analysis
    batch_analysis(model, n_samples=50)
    
    beauty_print("✓ Diagnostic complete", type="success")


if __name__ == '__main__':
    main()
