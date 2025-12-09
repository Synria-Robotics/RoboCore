"""Test FK correctness with different base_link and end_link combinations.

This test verifies that FK computation is correct when using different
base_link and end_link settings by checking transform composition.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import argparse

import robocore as rc
from robocore.modeling.robot_model import RobotModel
from robocore.utils.beauty_logger import beauty_print, beauty_print_array


def test_transform_composition(model_path, arm_side='left', verbose=False):
    """Test FK correctness by verifying transform composition.
    
    Principle: T_base_to_end = T_base_to_intermediate @ T_intermediate_to_end
    
    :param model_path: Path to robot model file
    :param arm_side: 'left' or 'right'
    :param verbose: Show detailed information
    :return: Dictionary with test results
    """
    if arm_side == 'left':
        base_link = 'base_link'
        intermediate_link = 'left_arm_link1'  # Shoulder
        end_link = 'left_arm_link7'
        joint_prefix = 'left_arm_joint'
    else:
        base_link = 'base_link'
        intermediate_link = 'right_arm_link1'  # Shoulder
        end_link = 'right_arm_link7'
        joint_prefix = 'right_arm_joint'
    
    beauty_print(f"Testing {arm_side.upper()} Arm FK with Transform Composition", type="module", centered=True)
    print(f"  Base: {base_link} -> Intermediate: {intermediate_link} -> End: {end_link}")
    
    # Create models with different base/end combinations
    model_base_to_end = RobotModel(model_path, base_link=base_link, end_link=end_link)
    model_base_to_intermediate = RobotModel(model_path, base_link=base_link, end_link=intermediate_link)
    model_intermediate_to_end = RobotModel(model_path, base_link=intermediate_link, end_link=end_link)
    
    if verbose:
        beauty_print(f"{arm_side.upper()} Arm Model (base->end):", type="info")
        model_base_to_end.summary(show_chain=True)
        beauty_print(f"{arm_side.upper()} Arm Model (base->intermediate):", type="info")
        model_base_to_intermediate.summary(show_chain=True)
        beauty_print(f"{arm_side.upper()} Arm Model (intermediate->end):", type="info")
        model_intermediate_to_end.summary(show_chain=True)
    
    # Generate random joint configurations
    num_tests = 10
    np.random.seed(42)
    q_base_to_end = model_base_to_end.random_q_batch(num_tests, seed=42, scale=0.8)
    
    # Extract joint values for intermediate model
    # The intermediate model has fewer DOF (only from base to intermediate)
    num_dof_base_to_intermediate = model_base_to_intermediate.num_chain_dof
    num_dof_intermediate_to_end = model_intermediate_to_end.num_chain_dof
    
    errors = []
    max_pos_error = 0.0
    max_ori_error = 0.0
    
    for i in range(num_tests):
        q_full = q_base_to_end[i]
        
        # Split joint configuration
        q_base_to_intermediate = q_full[:num_dof_base_to_intermediate]
        q_intermediate_to_end = q_full[num_dof_base_to_intermediate:]
        
        # Compute FK for each segment
        T_base_to_end = model_base_to_end.fk(q_full, return_end=True)
        T_base_to_intermediate = model_base_to_intermediate.fk(q_base_to_intermediate, return_end=True)
        T_intermediate_to_end = model_intermediate_to_end.fk(q_intermediate_to_end, return_end=True)
        
        # Verify: T_base_to_end should equal T_base_to_intermediate @ T_intermediate_to_end
        T_composed = T_base_to_intermediate @ T_intermediate_to_end
        
        # Compute errors
        pos_error = np.linalg.norm(T_base_to_end[:3, 3] - T_composed[:3, 3])
        R_error = T_base_to_end[:3, :3] @ T_composed[:3, :3].T
        # Orientation error as angle
        trace = np.trace(R_error)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        errors.append({
            'pos_error': pos_error,
            'ori_error': angle_error,
            'sample': i
        })
        
        max_pos_error = max(max_pos_error, pos_error)
        max_ori_error = max(max_ori_error, angle_error)
        
        if verbose and i < 3:  # Show first 3 samples
            print(f"\n  Sample {i+1}:")
            print(f"    Position error: {pos_error:.2e} m")
            print(f"    Orientation error: {angle_error:.2e} rad ({np.degrees(angle_error):.4f} deg)")
            print(f"    T_base_to_end position: {beauty_print_array(T_base_to_end[:3, 3])}")
            print(f"    T_composed position:    {beauty_print_array(T_composed[:3, 3])}")
    
    # Summary
    avg_pos_error = np.mean([e['pos_error'] for e in errors])
    avg_ori_error = np.mean([e['ori_error'] for e in errors])
    
    beauty_print(f"Test Results for {arm_side.upper()} Arm:", type="module")
    print(f"  Number of tests: {num_tests}")
    print(f"  Average position error: {avg_pos_error:.2e} m")
    print(f"  Maximum position error: {max_pos_error:.2e} m")
    print(f"  Average orientation error: {avg_ori_error:.2e} rad ({np.degrees(avg_ori_error):.4f} deg)")
    print(f"  Maximum orientation error: {max_ori_error:.2e} rad ({np.degrees(max_ori_error):.4f} deg)")
    
    # Tolerance check
    pos_tol = 1e-6  # 1 micrometer
    ori_tol = 1e-6  # ~0.00006 degrees
    
    passed = (max_pos_error < pos_tol) and (max_ori_error < ori_tol)
    
    if passed:
        beauty_print("✓ Test PASSED", type="success")
    else:
        beauty_print("✗ Test FAILED", type="error")
        print(f"  Position tolerance: {pos_tol:.2e} m")
        print(f"  Orientation tolerance: {ori_tol:.2e} rad")
    
    return {
        'passed': passed,
        'avg_pos_error': avg_pos_error,
        'max_pos_error': max_pos_error,
        'avg_ori_error': avg_ori_error,
        'max_ori_error': max_ori_error,
        'num_tests': num_tests
    }


def test_direct_comparison(model_path, arm_side='left', verbose=False):
    """Test FK by directly comparing results with different base links.
    
    This test compares FK results when using:
    1. base_link='base_link', end_link='arm_link7'
    2. base_link='arm_link1', end_link='arm_link7' (with base transform applied)
    
    :param model_path: Path to robot model file
    :param arm_side: 'left' or 'right'
    :param verbose: Show detailed information
    :return: Dictionary with test results
    """
    if arm_side == 'left':
        base_link_full = 'base_link'
        base_link_shoulder = 'left_arm_link1'
        end_link = 'left_arm_link7'
    else:
        base_link_full = 'base_link'
        base_link_shoulder = 'right_arm_link1'
        end_link = 'right_arm_link7'
    
    beauty_print(f"Testing {arm_side.upper()} Arm FK with Direct Comparison", type="module", centered=True)
    print(f"  Method 1: base={base_link_full}, end={end_link}")
    print(f"  Method 2: base={base_link_shoulder}, end={end_link} (with base transform)")
    
    # Create models
    model_full = RobotModel(model_path, base_link=base_link_full, end_link=end_link)
    model_shoulder = RobotModel(model_path, base_link=base_link_shoulder, end_link=end_link)
    model_base_to_shoulder = RobotModel(model_path, base_link=base_link_full, end_link=base_link_shoulder)
    
    # Generate random joint configurations
    num_tests = 10
    np.random.seed(42)
    q_full = model_full.random_q_batch(num_tests, seed=42, scale=0.8)
    
    # Get DOF counts
    num_dof_full = model_full.num_chain_dof
    num_dof_shoulder = model_shoulder.num_chain_dof
    num_dof_base_to_shoulder = model_base_to_shoulder.num_chain_dof
    
    errors = []
    max_pos_error = 0.0
    max_ori_error = 0.0
    
    for i in range(num_tests):
        q_full_config = q_full[i]
        
        # Split: q_full = [q_base_to_shoulder, q_shoulder_to_end]
        q_base_to_shoulder = q_full_config[:num_dof_base_to_shoulder]
        q_shoulder_to_end = q_full_config[num_dof_base_to_shoulder:]
        
        # Method 1: Direct FK from base to end
        T_full = model_full.fk(q_full_config, return_end=True)
        
        # Method 2: FK from shoulder to end, then compose with base to shoulder
        T_base_to_shoulder = model_base_to_shoulder.fk(q_base_to_shoulder, return_end=True)
        T_shoulder_to_end = model_shoulder.fk(q_shoulder_to_end, return_end=True)
        T_composed = T_base_to_shoulder @ T_shoulder_to_end
        
        # Compute errors
        pos_error = np.linalg.norm(T_full[:3, 3] - T_composed[:3, 3])
        R_error = T_full[:3, :3] @ T_composed[:3, :3].T
        trace = np.trace(R_error)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        errors.append({
            'pos_error': pos_error,
            'ori_error': angle_error,
            'sample': i
        })
        
        max_pos_error = max(max_pos_error, pos_error)
        max_ori_error = max(max_ori_error, angle_error)
        
        if verbose and i < 3:
            print(f"\n  Sample {i+1}:")
            print(f"    Position error: {pos_error:.2e} m")
            print(f"    Orientation error: {angle_error:.2e} rad ({np.degrees(angle_error):.4f} deg)")
            print(f"    T_full position:     {beauty_print_array(T_full[:3, 3])}")
            print(f"    T_composed position: {beauty_print_array(T_composed[:3, 3])}")
    
    # Summary
    avg_pos_error = np.mean([e['pos_error'] for e in errors])
    avg_ori_error = np.mean([e['ori_error'] for e in errors])
    
    beauty_print(f"Test Results for {arm_side.upper()} Arm:", type="module")
    print(f"  Number of tests: {num_tests}")
    print(f"  Average position error: {avg_pos_error:.2e} m")
    print(f"  Maximum position error: {max_pos_error:.2e} m")
    print(f"  Average orientation error: {avg_ori_error:.2e} rad ({np.degrees(avg_ori_error):.4f} deg)")
    print(f"  Maximum orientation error: {max_ori_error:.2e} rad ({np.degrees(max_ori_error):.4f} deg)")
    
    # Tolerance check
    pos_tol = 1e-6
    ori_tol = 1e-6
    
    passed = (max_pos_error < pos_tol) and (max_ori_error < ori_tol)
    
    if passed:
        beauty_print("✓ Test PASSED", type="success")
    else:
        beauty_print("✗ Test FAILED", type="error")
        print(f"  Position tolerance: {pos_tol:.2e} m")
        print(f"  Orientation tolerance: {ori_tol:.2e} rad")
    
    return {
        'passed': passed,
        'avg_pos_error': avg_pos_error,
        'max_pos_error': max_pos_error,
        'avg_ori_error': avg_ori_error,
        'max_ori_error': max_ori_error,
        'num_tests': num_tests
    }


def main(args):
    """Run FK verification tests."""
    beauty_print("FK Base/End Link Verification Tests", type="module", centered=True)
    print(f"Model: {args.model_path}\n")
    
    # Test both arms
    results = {}
    
    # Test 1: Transform composition
    beauty_print("=" * 60, type="info")
    results['left_composition'] = test_transform_composition(
        args.model_path, arm_side='left', verbose=args.verbose
    )
    
    beauty_print("=" * 60, type="info")
    results['right_composition'] = test_transform_composition(
        args.model_path, arm_side='right', verbose=args.verbose
    )
    
    # Test 2: Direct comparison
    beauty_print("=" * 60, type="info")
    results['left_direct'] = test_direct_comparison(
        args.model_path, arm_side='left', verbose=args.verbose
    )
    
    beauty_print("=" * 60, type="info")
    results['right_direct'] = test_direct_comparison(
        args.model_path, arm_side='right', verbose=args.verbose
    )
    
    # Final summary
    beauty_print("=" * 60, type="info")
    beauty_print("Final Summary", type="module", centered=True)
    
    all_passed = all(r['passed'] for r in results.values())
    
    print("\nTest Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {test_name:25s} {status:8s} "
              f"(max pos: {result['max_pos_error']:.2e} m, "
              f"max ori: {result['max_ori_error']:.2e} rad)")
    
    if all_passed:
        beauty_print("\n✓ All tests PASSED!", type="success", centered=True)
        return 0
    else:
        beauty_print("\n✗ Some tests FAILED!", type="error", centered=True)
        return 1


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Bessica_D", version="v1_0", variant="covered_interactive", model_format="mjcf")
    
    parser = argparse.ArgumentParser(description="Test FK correctness with different base/end links")
    parser.add_argument('--model-path', type=str, default=model_path,
                        help='Path to robot model file (default: Bessica-D)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information for each test sample')
    args = parser.parse_args()
    
    exit(main(args))
