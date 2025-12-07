"""Optimize IK solver default parameters for better success rate.

This script systematically tests different parameter combinations
to find optimal defaults for both NumPy and Torch backends.
"""

import numpy as np
import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print


def test_ik_params(
    robot_model,
    backend,
    num_tests=100,
    seed=42,
    max_iters=100,
    pos_tol=1e-4,
    ori_tol=1e-4,
    min_damping=1e-4,
    max_damping=5e-2,
    base_step=1.0,
    init_strategy='random',
):
    """Test IK with given parameters and return success rate."""
    rc.set_backend(backend)
    
    # Generate random test cases
    np.random.seed(seed)
    joint_configs = robot_model.random_q_batch(num_tests, seed=seed, scale=0.8)
    target_poses = forward_kinematics(robot_model, joint_configs, return_end=True)
    if target_poses.ndim == 2:
        target_poses = target_poses[np.newaxis, ...]
    
    # Test batch IK
    results = inverse_kinematics(
        robot_model,
        target_poses,
        method='dls',
        num_initial_guesses=1,
        initial_guess_strategy=init_strategy,
        random_seed=seed,
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        min_damping=min_damping,
        max_damping=max_damping,
        base_step=base_step,
    )
    
    # Count successes
    if isinstance(results, list):
        successes = sum(1 for r in results if r.get('success', False))
    else:
        successes = 1 if results.get('success', False) else 0
    
    return successes / num_tests


def grid_search(robot_model, backend='numpy'):
    """Perform grid search over parameter space."""
    beauty_print(f"Grid Search for {backend.upper()} Backend", type="module", centered=True)
    
    # Parameter ranges to test
    param_grid = {
        'max_iters': [50, 100, 150, 200],
        'pos_tol': [1e-5, 1e-4, 1e-3],
        'ori_tol': [1e-5, 1e-4, 1e-3],
        'min_damping': [1e-5, 1e-4, 1e-3],
        'max_damping': [1e-3, 5e-3, 1e-2, 5e-2],
        'base_step': [0.5, 0.8, 1.0, 1.2],
    }
    
    # Start with current defaults
    best_params = {
        'max_iters': 100,
        'pos_tol': 1e-4,
        'ori_tol': 1e-4,
        'min_damping': 1e-4,
        'max_damping': 5e-2,
        'base_step': 1.0,
    }
    best_rate = 0.0
    
    # Test current defaults first
    current_rate = test_ik_params(robot_model, backend, num_tests=50, **best_params)
    beauty_print(f"Current defaults: {current_rate*100:.1f}% success")
    if current_rate > best_rate:
        best_rate = current_rate
    
    # Test variations around current defaults
    beauty_print("Testing parameter variations...")
    
    # Test max_iters
    for max_iters in param_grid['max_iters']:
        params = best_params.copy()
        params['max_iters'] = max_iters
        rate = test_ik_params(robot_model, backend, num_tests=50, **params)
        if rate > best_rate:
            best_rate = rate
            best_params['max_iters'] = max_iters
            beauty_print(f"  max_iters={max_iters}: {rate*100:.1f}% (NEW BEST)")
        else:
            beauty_print(f"  max_iters={max_iters}: {rate*100:.1f}%")
    
    # Test tolerances
    for pos_tol in param_grid['pos_tol']:
        for ori_tol in param_grid['ori_tol']:
            params = best_params.copy()
            params['pos_tol'] = pos_tol
            params['ori_tol'] = ori_tol
            rate = test_ik_params(robot_model, backend, num_tests=50, **params)
            if rate > best_rate:
                best_rate = rate
                best_params['pos_tol'] = pos_tol
                best_params['ori_tol'] = ori_tol
                beauty_print(f"  pos_tol={pos_tol:.0e}, ori_tol={ori_tol:.0e}: {rate*100:.1f}% (NEW BEST)")
    
    # Test damping
    for min_damping in param_grid['min_damping']:
        for max_damping in param_grid['max_damping']:
            params = best_params.copy()
            params['min_damping'] = min_damping
            params['max_damping'] = max_damping
            rate = test_ik_params(robot_model, backend, num_tests=50, **params)
            if rate > best_rate:
                best_rate = rate
                best_params['min_damping'] = min_damping
                best_params['max_damping'] = max_damping
                beauty_print(f"  min_damping={min_damping:.0e}, max_damping={max_damping:.0e}: {rate*100:.1f}% (NEW BEST)")
    
    # Test base_step
    for base_step in param_grid['base_step']:
        params = best_params.copy()
        params['base_step'] = base_step
        rate = test_ik_params(robot_model, backend, num_tests=50, **params)
        if rate > best_rate:
            best_rate = rate
            best_params['base_step'] = base_step
            beauty_print(f"  base_step={base_step}: {rate*100:.1f}% (NEW BEST)")
    
    # Final test with best params
    beauty_print("\nFinal validation with best parameters...")
    final_rate = test_ik_params(robot_model, backend, num_tests=200, **best_params)
    beauty_print(f"Best parameters: {best_params}")
    beauty_print(f"Final success rate: {final_rate*100:.1f}% (n=200)")
    
    return best_params, final_rate


def main():
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    robot_model = RobotModel(str(model_path), base_link='base_link', end_link='Link6')
    
    beauty_print("IK Parameter Optimization", type="title", centered=True)
    
    # Test NumPy
    print("\n" + "="*80)
    best_numpy, rate_numpy = grid_search(robot_model, backend='numpy')
    
    # Test Torch
    print("\n" + "="*80)
    best_torch, rate_torch = grid_search(robot_model, backend='torch')
    
    # Summary
    print("\n" + "="*80)
    beauty_print("Optimization Summary", type="module", centered=True)
    print(f"\nNumPy Backend:")
    print(f"  Best params: {best_numpy}")
    print(f"  Success rate: {rate_numpy*100:.1f}%")
    print(f"\nTorch Backend:")
    print(f"  Best params: {best_torch}")
    print(f"  Success rate: {rate_torch*100:.1f}%")


if __name__ == "__main__":
    main()
