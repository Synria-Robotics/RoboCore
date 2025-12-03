"""
Compare IK results between robocore and pytorch_kinematics

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import numpy as np
import subprocess
import sys
from pathlib import Path

def run_demo(demo_path, end_pose, q_init=None):
    """Run a demo script and parse its output."""
    cmd = [sys.executable, str(demo_path)]
    if q_init is not None:
        # Note: q_init is hardcoded in the demos, so we don't pass it
        pass
    
    # Set end_pose if provided
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


def main():
    from synriard import get_model_path
    
    # Test configuration - single test case
    q_init = np.array([+0.86066, -0.19202, +1.12657, +0.62005, -1.27493, +1.49421])
    end_pose = [0.17006, 0.01704, 0.20533, 0.042114, 0.828366, 0.083037, 0.552396]
    
    # Paths to demos
    demo_ik_path = Path(__file__).parent / 'demo_ik.py'
    demo_ik_pk_path = Path(__file__).parent / 'demo_ik_pk.py'
    
    print("=" * 80)
    print("IK Comparison Test: robocore vs pytorch_kinematics")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Initial Guess: {q_init}")
    print(f"  Target Pose: {end_pose[:3]} (position), {end_pose[3:]} (quaternion)")
    print()
    
    # Run robocore demo
    print("Running robocore IK demo...")
    result_rc, stderr_rc = run_demo(demo_ik_path, end_pose)
    
    # Run pytorch_kinematics demo
    print("Running pytorch_kinematics IK demo...")
    result_pk, stderr_pk = run_demo(demo_ik_pk_path, end_pose)
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    if result_rc is None:
        print("❌ robocore demo failed:")
        print(stderr_rc)
    else:
        print("\n📊 robocore Results:")
        print(f"  Success: {result_rc.get('success', 'N/A')}")
        print(f"  Iterations: {result_rc.get('iters', 'N/A')}")
        print(f"  Position Error: {result_rc.get('pos_err', 'N/A'):.6e} m")
        print(f"  Orientation Error: {result_rc.get('ori_err', 'N/A'):.6e} rad")
        if 'err_norm' in result_rc:
            print(f"  Total Error: {result_rc.get('err_norm', 'N/A'):.6e}")
        if 'q' in result_rc:
            print(f"  Solution: {result_rc['q']}")
    
    if result_pk is None:
        print("\n❌ pytorch_kinematics demo failed:")
        print(stderr_pk)
    else:
        print("\n📊 pytorch_kinematics Results:")
        print(f"  Success: {result_pk.get('success', 'N/A')}")
        print(f"  Iterations: {result_pk.get('iters', 'N/A')}")
        print(f"  Position Error: {result_pk.get('pos_err', 'N/A'):.6e} m")
        print(f"  Orientation Error: {result_pk.get('ori_err', 'N/A'):.6e} rad")
        if 'err_norm' in result_pk:
            print(f"  Total Error: {result_pk.get('err_norm', 'N/A'):.6e}")
        if 'q' in result_pk:
            print(f"  Solution: {result_pk['q']}")
    
    # Direct comparison
    if result_rc is not None and result_pk is not None:
        print("\n" + "=" * 80)
        print("Direct Comparison")
        print("=" * 80)
        
        # Compare success
        success_match = result_rc.get('success') == result_pk.get('success')
        print(f"Success Match: {'✅' if success_match else '❌'}")
        print(f"  robocore: {result_rc.get('success')}")
        print(f"  pytorch_kinematics: {result_pk.get('success')}")
        
        # Compare errors
        if 'pos_err' in result_rc and 'pos_err' in result_pk:
            pos_err_diff = abs(result_rc['pos_err'] - result_pk['pos_err'])
            print(f"\nPosition Error Difference: {pos_err_diff:.6e} m")
            print(f"  robocore: {result_rc['pos_err']:.6e} m")
            print(f"  pytorch_kinematics: {result_pk['pos_err']:.6e} m")
        
        if 'ori_err' in result_rc and 'ori_err' in result_pk:
            ori_err_diff = abs(result_rc['ori_err'] - result_pk['ori_err'])
            print(f"\nOrientation Error Difference: {ori_err_diff:.6e} rad")
            print(f"  robocore: {result_rc['ori_err']:.6e} rad")
            print(f"  pytorch_kinematics: {result_pk['ori_err']:.6e} rad")
        
        # Compare solutions
        if 'q' in result_rc and 'q' in result_pk:
            q_diff = np.linalg.norm(result_rc['q'] - result_pk['q'])
            print(f"\nJoint Angle Solution Difference (L2 norm): {q_diff:.6e} rad")
            print(f"  robocore: {result_rc['q']}")
            print(f"  pytorch_kinematics: {result_pk['q']}")
            
            # Per-joint difference
            print(f"\nPer-Joint Difference:")
            for i, (q_rc, q_pk) in enumerate(zip(result_rc['q'], result_pk['q'])):
                diff = abs(q_rc - q_pk)
                print(f"  Joint {i+1}: {diff:.6e} rad ({np.rad2deg(diff):.6f} deg)")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Both methods succeeded: {'✅' if success_match and result_rc.get('success') else '❌'}")
        if 'pos_err' in result_rc and 'pos_err' in result_pk:
            print(f"Position error: robocore {'better' if result_rc['pos_err'] < result_pk['pos_err'] else 'worse'} by {abs(result_rc['pos_err'] - result_pk['pos_err']):.6e} m")
        if 'ori_err' in result_rc and 'ori_err' in result_pk:
            print(f"Orientation error: robocore {'better' if result_rc['ori_err'] < result_pk['ori_err'] else 'worse'} by {abs(result_rc['ori_err'] - result_pk['ori_err']):.6e} rad")
        if 'q' in result_rc and 'q' in result_pk:
            print(f"Solution difference: {np.linalg.norm(result_rc['q'] - result_pk['q']):.6e} rad (L2 norm)")


if __name__ == "__main__":
    main()

