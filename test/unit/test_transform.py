#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import torch
import pytest
from robocore.transform import (
    rpy_to_matrix,
    matrix_to_rpy,
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_error,
    make_transform,
)


class TestSO3Conversions:
    """Test SO(3) rotation representations."""
    
    def test_rpy_roundtrip(self):
        """Test RPY to matrix and back."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        
        R = rpy_to_matrix(roll, pitch, yaw)
        roll2, pitch2, yaw2 = matrix_to_rpy(R)
        
        # Should be close (within numerical precision)
        assert np.abs(roll - roll2) < 1e-10
        assert np.abs(pitch - pitch2) < 1e-10
        assert np.abs(yaw - yaw2) < 1e-10
    
    def test_quaternion_roundtrip(self):
        """Test quaternion to matrix and back."""
        # Unit quaternion (90 deg around Z)
        q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # [w, x, y, z]
        
        R = quaternion_to_matrix(q)
        q2 = matrix_to_quaternion(R)
        
        # Quaternions q and -q represent same rotation
        assert np.allclose(q, q2) or np.allclose(q, -q2)
    
    def test_axis_angle_roundtrip(self):
        """Test axis-angle to matrix and back."""
        axis = np.array([0, 0, 1])  # Z-axis
        angle = np.pi / 4
        
        R = axis_angle_to_matrix(axis, angle)
        axis2, angle2 = matrix_to_axis_angle(R)
        
        # Check angle
        assert np.abs(angle - angle2) < 1e-10
        # Check axis (should be parallel)
        assert np.abs(np.dot(axis, axis2) - 1.0) < 1e-10
    
    def test_rotation_matrix_properties(self):
        """Test that rotation matrices satisfy orthogonality."""
        R = rpy_to_matrix(0.1, 0.2, 0.3)
        
        # R^T @ R should be identity
        I = R.T @ R
        assert np.allclose(I, np.eye(3))
        
        # det(R) should be 1
        det = np.linalg.det(R)
        assert np.abs(det - 1.0) < 1e-10
    
    def test_rotation_error(self):
        """Test rotation error computation."""
        R1 = rpy_to_matrix(0.1, 0.2, 0.3)
        R2 = rpy_to_matrix(0.1, 0.2, 0.3)
        
        # Same rotation should have zero error
        err = rotation_error(R1, R2)
        assert np.linalg.norm(err) < 1e-10
        
        # Different rotations should have non-zero error
        R3 = rpy_to_matrix(0.5, 0.6, 0.7)
        err2 = rotation_error(R1, R3)
        assert np.linalg.norm(err2) > 0.1


class TestSE3Transforms:
    """Test SE(3) transformation matrices."""
    
    def test_make_transform(self):
        """Test 4x4 transformation matrix creation."""
        R = rpy_to_matrix(0.1, 0.2, 0.3)
        t = np.array([1.0, 2.0, 3.0])
        
        T = make_transform(R, t)
        
        # Check shape
        assert T.shape == (4, 4)
        
        # Check rotation block
        assert np.allclose(T[:3, :3], R)
        
        # Check translation
        assert np.allclose(T[:3, 3], t)
        
        # Check bottom row
        assert np.allclose(T[3, :], [0, 0, 0, 1])
    
    def test_transform_composition(self):
        """Test that transform composition works correctly."""
        R1 = rpy_to_matrix(0.1, 0.0, 0.0)
        t1 = np.array([1.0, 0.0, 0.0])
        T1 = make_transform(R1, t1)
        
        R2 = rpy_to_matrix(0.0, 0.2, 0.0)
        t2 = np.array([0.0, 1.0, 0.0])
        T2 = make_transform(R2, t2)
        
        # Compose
        T12 = T1 @ T2
        
        # Result should still be valid transformation
        assert T12.shape == (4, 4)
        assert np.allclose(T12[3, :], [0, 0, 0, 1])
        
        # Rotation part should be orthogonal
        R12 = T12[:3, :3]
        assert np.allclose(R12.T @ R12, np.eye(3))


class TestBackendConsistency:
    """Test that transform functions work consistently across backends."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires PyTorch")
    def test_rpy_numpy_vs_torch(self):
        """Test RPY conversion consistency between NumPy and PyTorch."""
        from robocore.utils.backend import set_backend, get_backend
        
        roll, pitch, yaw = 0.1, 0.2, 0.3
        
        # NumPy
        prev = get_backend()
        set_backend('numpy')
        R_np = rpy_to_matrix(roll, pitch, yaw)
        
        # PyTorch
        set_backend('torch')
        R_torch = rpy_to_matrix(roll, pitch, yaw)
        if torch.is_tensor(R_torch):
            R_torch = R_torch.cpu().numpy()
        
        set_backend(prev)
        
        # Should match
        assert np.allclose(R_np, R_torch, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
