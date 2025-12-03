"""
Test IK success rate and error statistics for robocore vs pytorch_kinematics

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import subprocess
import sys
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk import forward_kinematics
from robocore.utils.beauty_logger import beauty_print


def parse_output(output):
    """Parse demo output to extract IK results."""
    result = {}
    lines = output.split('\n')
    
    for i, line in enumerate(lines):
        if 'Success:' in line:
            result['success'] = 'True' in line or 'true' in line.lower()
        elif 'Iterations:' in line:
            try:
                result['iters'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Position Error:' in line:
            try:
                result['pos_err'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'Orientation Error:' in line:
            try:
                result['ori_err'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'Total Error:' in line:
            try:
                result['err_norm'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'q_ik =' in line:
            # Only parse if this is the radians line (not degrees)
            if i > 0 and 'radians' in lines[i-1]:
                try:
                    # Extract joint angles from the line
                    q_str = line.split('=')[1].strip()
                    # Remove brackets and split
                    q_str = q_str.replace('[', '').replace(']', '').replace('+', '').replace('-', ' -').strip()
                    # Split by comma and convert to float
                    q_values = []
                    for x in q_str.split(','):
                        x = x.strip()
                        if x:
                            q_values.append(float(x))
                    if q_values:
                        result['q'] = np.array(q_values)
                except Exception as e:
                    pass
    
    return result


def run_demo(demo_path, end_pose):
    """Run a demo script and parse its output."""
    cmd = [sys.executable, str(demo_path)]
    if end_pose is not None:
        cmd.extend(['--end-pose'] + [str(x) for x in end_pose])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return parse_output(result.stdout), result.stderr
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def generate_reachable_targets(robot_model, n_targets=20, seed=42):
    """Generate random reachable target poses by sampling joint space."""
    rng = np.random.default_rng(seed)
    targets = []
    
    for _ in range(n_targets):
        # Sample random joint angles
        q = robot_model.random_q(rng)
        # Compute FK to get reachable pose
        T = forward_kinematics(robot_model, q, return_end=True)
        
        # Extract position and quaternion
        pos = T[:3, 3]
        from robocore.transform.conversions import matrix_to_quaternion
        quat = matrix_to_quaternion(T[:3, :3])
        
        targets.append({
            'q_true': q,
            'pose': np.concatenate([pos, quat]),
            'T': T
        })
    
    return targets


def main():
    from synriard import get_model_path
    
    # Setup
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    robot_model = RobotModel(str(model_path), end_link='Link6')
    
    # Fixed initial guess
    q_init = np.array([+0.86066, -0.19202, +1.12657, +0.62005, -1.27493, +1.49421])
    
    # Paths to demos
    demo_ik_path = Path(__file__).parent / 'demo_ik.py'
    demo_ik_pk_path = Path(__file__).parent / 'demo_ik_pk.py'
    
    # Generate test targets
    beauty_print("Generating test targets...")
    targets = generate_reachable_targets(robot_model, n_targets=20, seed=42)
    beauty_print(f"Generated {len(targets)} test targets")
    
    # Test both methods
    results_rc = []
    results_pk = []
    
    beauty_print("\nTesting robocore IK...")
    for i, target in enumerate(targets):
        end_pose = target['pose'].tolist()
        result, stderr = run_demo(demo_ik_path, end_pose)
        if result is not None:
            result['target_idx'] = i
            result['q_true'] = target['q_true']
            results_rc.append(result)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{len(targets)}")
    
    beauty_print("\nTesting pytorch_kinematics IK...")
    for i, target in enumerate(targets):
        end_pose = target['pose'].tolist()
        result, stderr = run_demo(demo_ik_pk_path, end_pose)
        if result is not None:
            result['target_idx'] = i
            result['q_true'] = target['q_true']
            results_pk.append(result)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{len(targets)}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Success Rate Statistics")
    print("=" * 80)
    
    success_rc = [r for r in results_rc if r.get('success', False)]
    success_pk = [r for r in results_pk if r.get('success', False)]
    
    print(f"\nrobocore:")
    print(f"  Total tests: {len(results_rc)}")
    print(f"  Successful: {len(success_rc)}")
    print(f"  Success rate: {len(success_rc)/len(results_rc)*100:.1f}%")
    
    print(f"\npytorch_kinematics:")
    print(f"  Total tests: {len(results_pk)}")
    print(f"  Successful: {len(success_pk)}")
    print(f"  Success rate: {len(success_pk)/len(results_pk)*100:.1f}%")
    
    # Error statistics (only for successful cases)
    if success_rc:
        print("\n" + "=" * 80)
        print("Error Statistics (Successful Cases Only)")
        print("=" * 80)
        
        pos_errs_rc = [r['pos_err'] for r in success_rc if 'pos_err' in r]
        ori_errs_rc = [r['ori_err'] for r in success_rc if 'ori_err' in r]
        iters_rc = [r['iters'] for r in success_rc if 'iters' in r]
        
        pos_errs_pk = [r['pos_err'] for r in success_pk if 'pos_err' in r]
        ori_errs_pk = [r['ori_err'] for r in success_pk if 'ori_err' in r]
        iters_pk = [r['iters'] for r in success_pk if 'iters' in r]
        
        print(f"\nrobocore:")
        if pos_errs_rc:
            print(f"  Position Error: mean={np.mean(pos_errs_rc):.6e} m, std={np.std(pos_errs_rc):.6e} m, max={np.max(pos_errs_rc):.6e} m")
        if ori_errs_rc:
            print(f"  Orientation Error: mean={np.mean(ori_errs_rc):.6e} rad, std={np.std(ori_errs_rc):.6e} rad, max={np.max(ori_errs_rc):.6e} rad")
        if iters_rc:
            print(f"  Iterations: mean={np.mean(iters_rc):.1f}, std={np.std(iters_rc):.1f}, max={np.max(iters_rc)}")
        
        print(f"\npytorch_kinematics:")
        if pos_errs_pk:
            print(f"  Position Error: mean={np.mean(pos_errs_pk):.6e} m, std={np.std(pos_errs_pk):.6e} m, max={np.max(pos_errs_pk):.6e} m")
        if ori_errs_pk:
            print(f"  Orientation Error: mean={np.mean(ori_errs_pk):.6e} rad, std={np.std(ori_errs_pk):.6e} rad, max={np.max(ori_errs_pk):.6e} rad")
        if iters_pk:
            print(f"  Iterations: mean={np.mean(iters_pk):.1f}, std={np.std(iters_pk):.1f}, max={np.max(iters_pk)}")
        
        # Comparison
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        
        if pos_errs_rc and pos_errs_pk:
            print(f"\nPosition Error:")
            print(f"  robocore better: {np.mean(pos_errs_rc) < np.mean(pos_errs_pk)}")
            print(f"  Mean difference: {abs(np.mean(pos_errs_rc) - np.mean(pos_errs_pk)):.6e} m")
        
        if ori_errs_rc and ori_errs_pk:
            print(f"\nOrientation Error:")
            print(f"  robocore better: {np.mean(ori_errs_rc) < np.mean(ori_errs_pk)}")
            print(f"  Mean difference: {abs(np.mean(ori_errs_rc) - np.mean(ori_errs_pk)):.6e} rad")
        
        if iters_rc and iters_pk:
            print(f"\nIterations:")
            print(f"  robocore faster: {np.mean(iters_rc) < np.mean(iters_pk)}")
            print(f"  Mean difference: {abs(np.mean(iters_rc) - np.mean(iters_pk)):.1f} iterations")


if __name__ == "__main__":
    main()


