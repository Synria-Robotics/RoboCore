#!/usr/bin/env python3
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import time

from robocore.utils.beauty_logger import beauty_print, beauty_print_matrix
from robocore.utils.backend import set_backend, get_backend
from robocore.transform import *


def print_matrix(name, matrix, precision=4):
    """Backward-compatible wrapper for new beauty_print_matrix util."""
    beauty_print_matrix(name, matrix, precision=precision)
    print()


def demo_backend_switching():
    """Demo 1: Backend switching."""
    beauty_print("Demo 1: Backend Switching", type="module")
    
    # Start with numpy
    set_backend('numpy')
    beauty_print(f"Current backend: {get_backend()}", type="info")
    
    # Create rotation using numpy
    R_numpy = rpy_to_matrix(0.1, 0.2, 0.3)
    print_matrix("R (numpy)", R_numpy)
    print(f"  Type: {type(R_numpy)}\n")
    
    # Switch to torch
    try:
        set_backend('torch', device='cpu')
        beauty_print(f"Switched to: {get_backend()}", type="success")
        
        R_torch = rpy_to_matrix(0.1, 0.2, 0.3)
        print_matrix("R (torch)", R_torch)
        print(f"  Type: {type(R_torch)}")
        print(f"  Device: {R_torch.device}\n")
        
        # Try GPU if available
        import torch
        if torch.cuda.is_available():
            set_backend('torch', device='cuda:0')
            beauty_print(f"Switched to: {get_backend()} on CUDA", type="success")
            
            R_gpu = rpy_to_matrix(0.1, 0.2, 0.3)
            print_matrix("R (GPU)", R_gpu)
            print(f"  Device: {R_gpu.device}\n")
        else:
            beauty_print("CUDA not available, skipping GPU demo", type="warning")
    except Exception as e:
        beauty_print(f"Torch not available: {e}", type="warning")
    
    # Switch back to numpy
    set_backend('numpy')


def demo_so3_operations():
    """Demo 2: SO(3) rotation operations."""
    beauty_print("Demo 2: SO(3) Rotation Operations", type="module")
    
    # Basic rotations
    beauty_print("Basic Rotations:", type="info")
    Rx = rotation_x(np.pi/4)
    Ry = rotation_y(np.pi/6)
    Rz = rotation_z(np.pi/3)
    
    print_matrix("Rx(45°)", Rx, precision=3)
    print_matrix("Ry(30°)", Ry, precision=3)
    print_matrix("Rz(60°)", Rz, precision=3)
    
    # RPY to matrix
    beauty_print("RPY to Rotation Matrix:", type="info")
    roll, pitch, yaw = 0.1, 0.2, 0.3
    R_rpy = rpy_to_matrix(roll, pitch, yaw)
    print_matrix(f"R(r={roll}, p={pitch}, y={yaw})", R_rpy)
    
    # Rotation composition
    beauty_print("Rotation Composition:", type="info")
    R_composed = rotation_multiply(rotation_multiply(Rz, Ry), Rx)
    print_matrix("R = Rz @ Ry @ Rx", R_composed, precision=3)
    
    # Apply rotation to vector
    beauty_print("Apply Rotation:", type="info")
    v = np.array([1.0, 0.0, 0.0])
    v_rotated = rotation_apply(R_rpy, v)
    print_matrix("v", v)
    print_matrix("R @ v", v_rotated)
    
    # Validation
    beauty_print("Validation:", type="info")
    is_valid = is_rotation_matrix(R_rpy)
    print(f"  is_rotation_matrix(R) = {is_valid}\n")


def demo_rotation_conversions():
    """Demo 3: Rotation representation conversions."""
    beauty_print("Demo 3: Rotation Conversions", type="module")
    
    # Start with RPY
    roll, pitch, yaw = 0.1, 0.2, 0.3
    beauty_print(f"Starting with RPY: [{roll}, {pitch}, {yaw}]", type="info")
    
    # To rotation matrix
    R = rpy_to_matrix(roll, pitch, yaw)
    print(f"  Rotation Matrix: 3×3 (det={np.linalg.det(R):.4f})")
    
    # To quaternion
    q = rpy_to_quaternion(roll, pitch, yaw)
    print(f"  Quaternion [x,y,z,w]: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    
    # To axis-angle
    axis, angle = rpy_to_axis_angle(roll, pitch, yaw)
    print(f"  Axis-Angle: axis=[{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}], angle={angle:.4f}")
    
    # Back to RPY
    rpy_back = matrix_to_rpy(R)
    print(f"  RPY (recovered): [{rpy_back[0]:.4f}, {rpy_back[1]:.4f}, {rpy_back[2]:.4f}]")
    
    # Quaternion operations
    beauty_print("\nQuaternion Operations:", type="info")
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q2 = rpy_to_quaternion(0.0, 0.0, np.pi/2)
    q_mult = quaternion_multiply(q1, q2)
    
    print_matrix("q1 (identity)", q1)
    print_matrix("q2 (90° yaw)", q2)
    print_matrix("q1 * q2", q_mult)


def demo_se3_operations():
    """Demo 4: SE(3) transformation operations."""
    beauty_print("Demo 4: SE(3) Transformation Operations", type="module")
    
    # Create transformation
    beauty_print("Creating Transformation:", type="info")
    R = rpy_to_matrix(0.0, 0.0, np.pi/4)
    t = np.array([1.0, 2.0, 3.0])
    T = make_transform(R, t)
    
    print_matrix("Rotation R", R, precision=3)
    print_matrix("Translation t", t)
    print_matrix("Transform T (4×4)", T, precision=3)
    
    # Transform composition
    beauty_print("Transform Composition:", type="info")
    T1 = make_transform(rotation_z(np.pi/4), np.array([1.0, 0.0, 0.0]))
    T2 = make_transform(rotation_x(np.pi/6), np.array([0.0, 1.0, 0.0]))
    T_composed = transform_multiply(T1, T2)
    
    print_matrix("T1 @ T2", T_composed, precision=3)
    
    # Transform inverse
    beauty_print("Transform Inverse:", type="info")
    T_inv = transform_inverse(T)
    T_check = transform_multiply(T, T_inv)
    
    print_matrix("T^-1 @ T (should be I)", T_check, precision=3)
    
    # Apply to points
    beauty_print("Transform Points:", type="info")
    points = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    points_transformed = transform_apply(T, points)
    
    print_matrix("Points (3×3)", points, precision=2)
    print_matrix("Transformed", points_transformed, precision=2)


def demo_batch_processing():
    """Demo 5: Batch processing."""
    beauty_print("Demo 5: Batch Processing", type="module")

    
    beauty_print("Batch Rotation Generation:", type="info")
    
    # Generate batch of RPY values
    n_samples = 5
    rolls = np.linspace(0, 0.3, n_samples)
    pitches = np.linspace(0, 0.2, n_samples)
    yaws = np.linspace(0, 0.5, n_samples)
    
    print(f"  Generating {n_samples} rotation matrices...")
    
    # Batch conversion
    R_batch = rpy_to_matrix(rolls, pitches, yaws)
    print(f"  Batch shape: {R_batch.shape} = (N={n_samples}, 3, 3)\n")
    
    print_matrix("R_batch", R_batch, precision=3)
    
    # Batch quaternion conversion
    beauty_print("Batch Quaternion Conversion:", type="info")
    q_batch = matrix_to_quaternion(R_batch)
    print(f"  Quaternion batch shape: {q_batch.shape} = (N={n_samples}, 4)\n")
    
    for i in range(min(3, n_samples)):
        q_str = f"[{q_batch[i,0]:.3f}, {q_batch[i,1]:.3f}, {q_batch[i,2]:.3f}, {q_batch[i,3]:.3f}]"
        print(f"    q[{i}] = {q_str}")
    print()


def demo_interpolation():
    """Demo 6: Interpolation."""
    beauty_print("Demo 6: Rotation and Transform Interpolation", type="module")

    
    # SLERP
    beauty_print("Quaternion SLERP:", type="info")
    q_start = np.array([0.0, 0.0, 0.0, 1.0])
    q_end = rpy_to_quaternion(0.0, 0.0, np.pi/2)
    
    print("  Interpolating from identity to 90° yaw:")
    print("  " + "-"*60)
    print(f"  {'t':>6} | {'Quaternion [x, y, z, w]':^40} | {'Yaw'}")
    print("  " + "-"*60)
    
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = slerp(q_start, q_end, t)
        R_interp = quaternion_to_matrix(q_interp)
        rpy_interp = matrix_to_rpy(R_interp)
        yaw_deg = np.degrees(rpy_interp[2])
        
        q_str = f"[{q_interp[0]:6.3f}, {q_interp[1]:6.3f}, {q_interp[2]:6.3f}, {q_interp[3]:6.3f}]"
        print(f"  {t:>5.2f}  | {q_str:^40} | {yaw_deg:>5.1f}°")
    
    print("  " + "-"*60 + "\n")
    
    # Transform interpolation
    beauty_print("Transform Interpolation:", type="info")
    T_start = make_transform(rotation_z(0.0), np.array([0.0, 0.0, 0.0]))
    T_end = make_transform(rotation_z(np.pi/2), np.array([1.0, 1.0, 0.0]))
    
    T_mid = transform_interpolate(T_start, T_end, 0.5)
    print_matrix("T(t=0.5)", T_mid, precision=3)


def demo_utilities():
    """Demo 7: Utility functions."""
    beauty_print("Demo 7: Utility Functions", type="module")
    
    # Rotation distance
    beauty_print("Rotation Distance:", type="info")
    R1 = rotation_z(0.0)
    R2 = rotation_z(np.pi/4)
    R3 = rotation_z(np.pi/2)
    
    dist_12 = rotation_distance(R1, R2)
    dist_13 = rotation_distance(R1, R3)
    
    print(f"  distance(0°, 45°) = {np.degrees(dist_12):.2f}° = {dist_12:.4f} rad")
    print(f"  distance(0°, 90°) = {np.degrees(dist_13):.2f}° = {dist_13:.4f} rad\n")
    
    # Rotation error
    beauty_print("Rotation Error:", type="info")
    R_current = rotation_z(0.1)
    R_target = rotation_z(0.5)
    error = rotation_error(R_current, R_target)
    
    print_matrix("Error vector (axis * angle)", error)
    print(f"  Error magnitude: {np.linalg.norm(error):.4f} rad = {np.degrees(np.linalg.norm(error)):.2f}°\n")
    
    # Look-at matrix
    beauty_print("Look-At Matrix:", type="info")
    eye = np.array([3.0, 4.0, 5.0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    
    T_lookat = look_at(eye, target, up)
    print_matrix("Look-at transform", T_lookat, precision=3)
    
    # Validation
    beauty_print("Validation:", type="info")
    is_valid_R = is_rotation_matrix(R1)
    is_valid_T = is_transform_matrix(T_lookat)
    
    print(f"  is_rotation_matrix(R1) = {is_valid_R}")
    print(f"  is_transform_matrix(T_lookat) = {is_valid_T}\n")


def demo_performance():
    """Demo 8: Performance comparison."""
    beauty_print("Demo 8: Performance Comparison (Batch Operations)", type="module")

    
    n_samples = 1000
    beauty_print(f"Benchmarking with {n_samples} samples...", type="info")
    
    # Generate test data
    rolls = np.random.uniform(-np.pi, np.pi, n_samples)
    pitches = np.random.uniform(-np.pi/2, np.pi/2, n_samples)
    yaws = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Numpy benchmark
    set_backend('numpy')
    start = time.time()
    R_numpy = rpy_to_matrix(rolls, pitches, yaws)
    q_numpy = matrix_to_quaternion(R_numpy)
    time_numpy = time.time() - start
    
    print(f"\n  {'Backend':<15} | {'Device':<10} | {'Time (ms)':<12} | {'Speedup'}")
    print("  " + "-"*60)
    print(f"  {'NumPy':<15} | {'CPU':<10} | {time_numpy*1000:>10.2f}  | 1.0×")
    
    # Torch benchmarks
    try:
        import torch
        
        # CPU
        set_backend('torch', device='cpu')
        _ = rpy_to_matrix(rolls[:10], pitches[:10], yaws[:10])  # warmup
        
        start = time.time()
        R_torch = rpy_to_matrix(rolls, pitches, yaws)
        q_torch = matrix_to_quaternion(R_torch)
        time_torch_cpu = time.time() - start
        
        speedup = time_numpy / time_torch_cpu
        print(f"  {'PyTorch':<15} | {'CPU':<10} | {time_torch_cpu*1000:>10.2f}  | {speedup:.2f}×")
        
        # GPU
        if torch.cuda.is_available():
            set_backend('torch', device='cuda:0')
            _ = rpy_to_matrix(rolls[:10], pitches[:10], yaws[:10])  # warmup
            torch.cuda.synchronize()
            
            start = time.time()
            R_gpu = rpy_to_matrix(rolls, pitches, yaws)
            q_gpu = matrix_to_quaternion(R_gpu)
            torch.cuda.synchronize()
            time_torch_gpu = time.time() - start
            
            speedup = time_numpy / time_torch_gpu
            print(f"  {'PyTorch':<15} | {'CUDA':<10} | {time_torch_gpu*1000:>10.2f}  | {speedup:.2f}×")
    except Exception as e:
        beauty_print(f"Torch benchmarks skipped: {e}", type="warning")
    
    print("  " + "-"*60 + "\n")
    
    # Reset
    set_backend('numpy')


def main():
    """Run all demos."""
    beauty_print("RoboCore Transform Module Demo", type="module")
    print("  Unified numpy/torch backend for 3D transformations")    
    
    try:
        demo_backend_switching()
        demo_so3_operations()
        demo_rotation_conversions()
        demo_se3_operations()
        demo_batch_processing()
        demo_interpolation()
        demo_utilities()
        demo_performance()
        
        print("\n" + "="*80)
        beauty_print("All demos completed successfully!", type="success")
        print("="*80 + "\n")
        
    except Exception as e:
        beauty_print(f"Error: {e}", type="error")


if __name__ == "__main__":
    main()
