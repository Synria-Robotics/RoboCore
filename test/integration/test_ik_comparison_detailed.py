#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细对比 NumPy vs PyTorch 批量 IK 求解过程

目的：找出 Torch 批量 IK 成功率低的根本原因
"""

import numpy as np
import torch
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
from robocore.kinematics.fk import forward_kinematics


def random_q_in_limits(model, seed=42):
    """生成一个在关节限制内的随机配置"""
    rng = np.random.default_rng(seed)
    n = model.num_dof()
    q = np.zeros(n)
    for js in model._actuated:
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * 0.5
        q[js.index] = rng.uniform(mid - span, mid + span)
    return q


def compare_single_ik(model, target_pose, q_init, device='cpu'):
    """对比单个 IK 问题的 NumPy 和 Torch 求解过程"""
    
    print("\n" + "="*80)
    print("IK 对比诊断")
    print("="*80)
    
    # NumPy 求解
    print("\n[NumPy IK]")
    result_np = inverse_kinematics(
        model, target_pose, q_init,
        backend='numpy',
        method='dls',
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
    )
    print(f"  成功: {result_np['success']}")
    print(f"  迭代: {result_np['iters']}")
    print(f"  位置误差: {result_np.get('pos_err', 'N/A'):.6f}")
    print(f"  姿态误差: {result_np.get('ori_err', 'N/A'):.6f}")
    print(f"  总误差: {result_np['err_norm']:.6f}")
    
    # Torch 单样本求解（对照组）
    print("\n[PyTorch IK - 单样本模式]")
    solver_torch = IKSolverTorch(
        model,
        max_iters=100,
        pos_tol=1e-4,
        ori_tol=1e-4,
        device=torch.device(device),
        dtype=torch.float64
    )
    result_torch_single = solver_torch.solve(
        target_pose, q_init, method='dls'
    )
    print(f"  成功: {result_torch_single['success']}")
    print(f"  迭代: {result_torch_single['iters']}")
    print(f"  位置误差: {result_torch_single.get('pos_err', 'N/A'):.6f}")
    print(f"  姿态误差: {result_torch_single.get('ori_err', 'N/A'):.6f}")
    print(f"  总误差: {result_torch_single['err_norm']:.6f}")
    
    # Torch 批量求解（1个样本）
    print("\n[PyTorch IK - 批量模式 (batch=1)]")
    target_batch = torch.from_numpy(np.array([target_pose])).to(dtype=torch.float64, device=device)
    q_init_batch = torch.from_numpy(np.array([q_init])).to(dtype=torch.float64, device=device)
    
    result_torch_batch = solver_torch.solve(
        target_batch, q_init_batch, method='dls'
    )
    print(f"  成功: {result_torch_batch['success'][0].item()}")
    print(f"  迭代: {result_torch_batch['iterations'][0].item()}")
    print(f"  位置误差: {result_torch_batch['pos_err'][0].item():.6f}")
    print(f"  姿态误差: {result_torch_batch['ori_err'][0].item():.6f}")
    
    # 对比最终解
    print("\n" + "-"*80)
    print("最终解对比（关节角度）")
    print("-"*80)
    q_np = np.array(result_np['q'])
    q_torch_single = np.array(result_torch_single['q'])
    q_torch_batch = result_torch_batch['q'][0].cpu().numpy()
    
    print(f"{'Joint':<8} {'NumPy':<12} {'Torch-Single':<12} {'Torch-Batch':<12} {'Diff(NP-TS)':<12} {'Diff(NP-TB)':<12}")
    for i in range(len(q_np)):
        diff_single = q_np[i] - q_torch_single[i]
        diff_batch = q_np[i] - q_torch_batch[i]
        print(f"{i:<8} {q_np[i]:>11.6f} {q_torch_single[i]:>11.6f} {q_torch_batch[i]:>11.6f} {diff_single:>11.6f} {diff_batch:>11.6f}")
    
    return {
        'numpy': result_np,
        'torch_single': result_torch_single,
        'torch_batch': result_torch_batch,
    }


def batch_comparison(model, n_samples=32, device='cpu', seed=42):
    """批量对比多个样本"""
    print("\n" + "="*80)
    print(f"批量对比 ({n_samples} 样本)")
    print("="*80)
    
    rng = np.random.default_rng(seed)
    
    # 生成测试用例
    q_batch = []
    target_poses = []
    for i in range(n_samples):
        q = random_q_in_limits(model, seed=seed+i)
        T = forward_kinematics(model, q, backend='numpy', return_end=True)
        # 随机初始解
        q_init = random_q_in_limits(model, seed=seed+n_samples+i)
        
        q_batch.append(q_init)
        target_poses.append(T)
    
    q_batch = np.array(q_batch)
    
    # NumPy 逐个求解
    print("\n[NumPy - 逐样本求解]")
    np_results = []
    np_success_count = 0
    for i in range(n_samples):
        res = inverse_kinematics(
            model, target_poses[i], q_batch[i],
            backend='numpy', method='dls',
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4
        )
        np_results.append(res)
        if res['success']:
            np_success_count += 1
    
    np_success_rate = np_success_count / n_samples
    print(f"  成功率: {np_success_rate*100:.1f}% ({np_success_count}/{n_samples})")
    
    # 统计 NumPy 误差分布
    np_pos_errs = [r.get('pos_err', float('inf')) for r in np_results]
    np_ori_errs = [r.get('ori_err', float('inf')) for r in np_results]
    print(f"  位置误差: mean={np.mean(np_pos_errs):.6f}, max={np.max(np_pos_errs):.6f}, p90={np.percentile(np_pos_errs, 90):.6f}")
    print(f"  姿态误差: mean={np.mean(np_ori_errs):.6f}, max={np.max(np_ori_errs):.6f}, p90={np.percentile(np_ori_errs, 90):.6f}")
    
    # Torch 批量求解
    print("\n[PyTorch - 批量求解]")
    solver_torch = IKSolverTorch(
        model, max_iters=100, pos_tol=1e-4, ori_tol=1e-4,
        device=torch.device(device), dtype=torch.float64
    )
    
    target_batch = torch.stack([torch.from_numpy(p) for p in target_poses]).to(dtype=torch.float64, device=device)
    q_init_batch = torch.from_numpy(q_batch).to(dtype=torch.float64, device=device)
    
    torch_result = solver_torch.solve(target_batch, q_init_batch, method='dls')
    
    torch_success = torch_result['success'].cpu().numpy()
    torch_success_count = torch_success.sum()
    torch_success_rate = torch_success_count / n_samples
    
    print(f"  成功率: {torch_success_rate*100:.1f}% ({torch_success_count}/{n_samples})")
    
    torch_pos_errs = torch_result['pos_err'].cpu().numpy()
    torch_ori_errs = torch_result['ori_err'].cpu().numpy()
    print(f"  位置误差: mean={np.mean(torch_pos_errs):.6f}, max={np.max(torch_pos_errs):.6f}, p90={np.percentile(torch_pos_errs, 90):.6f}")
    print(f"  姿态误差: mean={np.mean(torch_ori_errs):.6f}, max={np.max(torch_ori_errs):.6f}, p90={np.percentile(torch_ori_errs, 90):.6f}")
    
    # 对比失败样本
    print("\n" + "-"*80)
    print("失败样本分析")
    print("-"*80)
    
    np_failed = [i for i, r in enumerate(np_results) if not r['success']]
    torch_failed = [i for i in range(n_samples) if not torch_success[i]]
    
    only_numpy_failed = set(np_failed) - set(torch_failed)
    only_torch_failed = set(torch_failed) - set(np_failed)
    both_failed = set(np_failed) & set(torch_failed)
    
    print(f"  仅 NumPy 失败: {len(only_numpy_failed)} 个")
    print(f"  仅 Torch 失败: {len(only_torch_failed)} 个 ⚠️")
    print(f"  两者都失败: {len(both_failed)} 个")
    
    # 详细分析仅 Torch 失败的样本
    if only_torch_failed:
        print("\n  仅 Torch 失败的样本详情（前5个）:")
        for idx in list(only_torch_failed)[:5]:
            print(f"\n    样本 #{idx}:")
            print(f"      NumPy: 成功, pos_err={np_results[idx]['pos_err']:.6f}, ori_err={np_results[idx]['ori_err']:.6f}, iters={np_results[idx]['iters']}")
            print(f"      Torch: 失败, pos_err={torch_pos_errs[idx]:.6f}, ori_err={torch_ori_errs[idx]:.6f}, iters={torch_result['iterations'][idx].item()}")
            
            # 计算两者最终解的差异
            q_np_final = np.array(np_results[idx]['q'])
            q_torch_final = torch_result['q'][idx].cpu().numpy()
            q_diff = np.linalg.norm(q_np_final - q_torch_final)
            print(f"      解的差异 (||q_np - q_torch||): {q_diff:.6f}")
    
    return {
        'numpy_success_rate': np_success_rate,
        'torch_success_rate': torch_success_rate,
        'only_torch_failed': only_torch_failed,
        'numpy_results': np_results,
        'torch_result': torch_result,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default='robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf')
    parser.add_argument('--end-link', default='tool0')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--single', action='store_true', help='只测试单个样本')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 加载模型
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"❌ URDF 不存在: {urdf_path}")
        return
    
    model = RobotModel(str(urdf_path), end_link=args.end_link)
    print(f"✓ 模型加载: DOF={model.num_dof()}, end_link={model.end_link}")
    
    if args.single:
        # 单样本详细对比
        q_target = random_q_in_limits(model, seed=args.seed)
        target_pose = forward_kinematics(model, q_target, backend='numpy', return_end=True)
        q_init = random_q_in_limits(model, seed=args.seed + 1)
        
        compare_single_ik(model, target_pose, q_init, device=args.device)
    else:
        # 批量对比
        batch_comparison(model, n_samples=args.batch_size, device=args.device, seed=args.seed)


if __name__ == '__main__':
    main()
