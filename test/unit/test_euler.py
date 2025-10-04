"""
Unit tests for Euler angle conversions.
Tests all 12 Euler sequences (6 intrinsic + 6 extrinsic) against SciPy.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robocore.transform.conversions import matrix_to_euler
from robocore.transform.so3 import euler_to_matrix


class TestEulerConversions:
    """Test Euler angle conversion functions."""
    
    # All 12 sequences: intrinsic (lowercase) and extrinsic (uppercase)
    SEQUENCES = ['xyz', 'XYZ', 'zyx', 'ZYX', 'xzy', 'XZY', 
                 'yxz', 'YXZ', 'yzx', 'YZX', 'zxy', 'ZXY']
    
    # Multiple test angle sets
    TEST_ANGLES = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.5, -0.3, 0.8]),
        np.array([-0.2, 0.4, -0.6]),
        np.array([1.5, -1.2, 0.9]),
        np.array([0.0, 0.0, 0.0]),
        np.array([np.pi/4, np.pi/6, np.pi/3]),
    ]
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_forward_conversion(self, seq):
        """Test angles -> matrix conversion matches SciPy."""
        angles = np.array([0.1, 0.2, 0.3])
        
        # SciPy reference
        R_scipy = Rotation.from_euler(seq, angles).as_matrix()
        
        # Our implementation
        R_ours = euler_to_matrix(angles[0], angles[1], angles[2], seq)
        
        # Check they match
        np.testing.assert_allclose(R_ours, R_scipy, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Forward conversion failed for {seq}")
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_inverse_conversion(self, seq):
        """Test matrix -> angles extraction matches SciPy."""
        # Generate a random rotation
        angles = np.array([0.5, -0.3, 0.8])
        R = Rotation.from_euler(seq, angles).as_matrix()
        
        # Extract using our function
        angles_ours = matrix_to_euler(R, seq)
        
        # Extract using SciPy
        angles_scipy = Rotation.from_matrix(R).as_euler(seq)
        
        # Check they match
        np.testing.assert_allclose(angles_ours, angles_scipy, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Inverse conversion failed for {seq}")
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_roundtrip_conversion(self, seq):
        """Test angles -> matrix -> angles roundtrip."""
        angles = np.array([0.1, 0.2, 0.3])
        
        # Forward
        R = euler_to_matrix(angles[0], angles[1], angles[2], seq)
        
        # Inverse
        angles_recovered = matrix_to_euler(R, seq)
        
        # Check roundtrip
        np.testing.assert_allclose(angles_recovered, angles, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Roundtrip failed for {seq}")
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    @pytest.mark.parametrize("angles", TEST_ANGLES)
    def test_multiple_angle_sets(self, seq, angles):
        """Test multiple angle sets for all sequences."""
        # Forward: angles -> matrix
        R_scipy = Rotation.from_euler(seq, angles).as_matrix()
        R_ours = euler_to_matrix(angles[0], angles[1], angles[2], seq)
        
        # Check matrix matches
        np.testing.assert_allclose(R_ours, R_scipy, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Forward failed for {seq} with angles {angles}")
        
        # Inverse: matrix -> angles
        angles_ours = matrix_to_euler(R_scipy, seq)
        angles_scipy = Rotation.from_matrix(R_scipy).as_euler(seq)
        
        # Check angles match
        np.testing.assert_allclose(angles_ours, angles_scipy, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Inverse failed for {seq} with angles {angles}")
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_batch_processing(self, seq):
        """Test batch processing of multiple rotation matrices."""
        # Create batch of angles
        batch_angles = np.array([
            [0.1, 0.2, 0.3],
            [0.5, -0.3, 0.8],
            [-0.2, 0.4, -0.6],
            [1.0, -0.5, 0.7],
        ])
        
        # Create batch of matrices using SciPy
        R_batch = Rotation.from_euler(seq, batch_angles).as_matrix()
        
        # Extract using our function
        angles_ours = matrix_to_euler(R_batch, seq)
        
        # Extract using SciPy
        angles_scipy = Rotation.from_matrix(R_batch).as_euler(seq)
        
        # Check they match
        np.testing.assert_allclose(angles_ours, angles_scipy, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Batch processing failed for {seq}")
    
    def test_identity_rotation(self):
        """Test identity rotation (zero angles)."""
        angles = np.array([0.0, 0.0, 0.0])
        
        for seq in self.SEQUENCES:
            R = euler_to_matrix(angles[0], angles[1], angles[2], seq)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-10, rtol=1e-10,
                                       err_msg=f"Identity rotation failed for {seq}")
            
            angles_recovered = matrix_to_euler(R, seq)
            np.testing.assert_allclose(angles_recovered, angles, atol=1e-10, rtol=1e-10,
                                       err_msg=f"Identity extraction failed for {seq}")
    
    def test_intrinsic_vs_extrinsic(self):
        """Test that intrinsic and extrinsic conventions are different."""
        angles = np.array([0.1, 0.2, 0.3])
        
        # Intrinsic XYZ and extrinsic XYZ should produce different matrices
        R_intrinsic = euler_to_matrix(angles[0], angles[1], angles[2], 'xyz')
        R_extrinsic = euler_to_matrix(angles[0], angles[1], angles[2], 'XYZ')
        
        # They should NOT be equal (for non-zero angles)
        assert not np.allclose(R_intrinsic, R_extrinsic, atol=1e-10), \
            "Intrinsic and extrinsic should produce different matrices"
        
        # But extrinsic ABC = intrinsic CBA (with reversed angles)
        R_extrinsic_xyz = euler_to_matrix(angles[0], angles[1], angles[2], 'XYZ')
        R_intrinsic_zyx = euler_to_matrix(angles[2], angles[1], angles[0], 'zyx')
        
        np.testing.assert_allclose(R_extrinsic_xyz, R_intrinsic_zyx, atol=1e-10, rtol=1e-10,
                                   err_msg="Extrinsic XYZ should equal intrinsic ZYX with reversed angles")
    
    def test_large_angles(self):
        """Test with larger angles (outside [-π, π])."""
        angles = np.array([2.5, -2.0, 1.8])
        
        for seq in self.SEQUENCES:
            R_scipy = Rotation.from_euler(seq, angles).as_matrix()
            R_ours = euler_to_matrix(angles[0], angles[1], angles[2], seq)
            
            np.testing.assert_allclose(R_ours, R_scipy, atol=1e-10, rtol=1e-10,
                                       err_msg=f"Large angles failed for {seq}")
    
    def test_invalid_sequence(self):
        """Test that invalid sequences raise appropriate errors."""
        angles = np.array([0.1, 0.2, 0.3])
        
        # euler_to_matrix raises KeyError for invalid sequences
        with pytest.raises((NotImplementedError, KeyError)):
            euler_to_matrix(angles[0], angles[1], angles[2], 'invalid')
        
        # matrix_to_euler raises NotImplementedError for invalid sequences
        R = np.eye(3)
        with pytest.raises(NotImplementedError):
            matrix_to_euler(R, 'invalid')
    
    @pytest.mark.parametrize("seq", SEQUENCES)
    def test_matrix_orthogonality(self, seq):
        """Test that generated matrices are orthogonal."""
        angles = np.array([0.5, -0.3, 0.8])
        R = euler_to_matrix(angles[0], angles[1], angles[2], seq)
        
        # Check R^T @ R = I
        RTR = R.T @ R
        np.testing.assert_allclose(RTR, np.eye(3), atol=1e-10, rtol=1e-10,
                                   err_msg=f"Matrix not orthogonal for {seq}")
        
        # Check det(R) = 1
        det = np.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-10, rtol=1e-10,
                                   err_msg=f"Determinant not 1 for {seq}")
    
    def test_extraction_from_scipy_matrix(self):
        """Test extraction from SciPy-generated matrices."""
        angles_list = [
            [0.1, 0.2, 0.3],
            [0.5, -0.3, 0.8],
            [-0.2, 0.4, -0.6],
        ]
        
        for seq in self.SEQUENCES:
            for angles in angles_list:
                # Create matrix with SciPy
                R_scipy = Rotation.from_euler(seq, angles).as_matrix()
                
                # Extract with our function
                angles_ours = matrix_to_euler(R_scipy, seq)
                
                # Reconstruct matrix
                R_reconstructed = euler_to_matrix(angles_ours[0], angles_ours[1], angles_ours[2], seq)
                
                # Check matrix matches
                np.testing.assert_allclose(R_reconstructed, R_scipy, atol=1e-10, rtol=1e-10,
                                           err_msg=f"Extraction/reconstruction failed for {seq}")


class TestEulerSpecialCases:
    """Test special cases and edge conditions."""
    
    def test_small_angles(self):
        """Test with very small angles."""
        angles = np.array([1e-8, 1e-9, 1e-10])
        
        for seq in ['xyz', 'zyx']:
            R = euler_to_matrix(angles[0], angles[1], angles[2], seq)
            angles_recovered = matrix_to_euler(R, seq)
            
            # Should be close to identity (tolerance adjusted for numerical precision)
            np.testing.assert_allclose(R, np.eye(3), atol=2e-8, rtol=1e-7)
    
    def test_single_axis_rotations(self):
        """Test rotations about single axes."""
        angle = 0.5
        
        # X-axis only
        R_x = euler_to_matrix(angle, 0, 0, 'xyz')
        angles_x = matrix_to_euler(R_x, 'xyz')
        np.testing.assert_allclose(angles_x, [angle, 0, 0], atol=1e-10, rtol=1e-10)
        
        # Y-axis only
        R_y = euler_to_matrix(0, angle, 0, 'xyz')
        angles_y = matrix_to_euler(R_y, 'xyz')
        np.testing.assert_allclose(angles_y, [0, angle, 0], atol=1e-10, rtol=1e-10)
        
        # Z-axis only
        R_z = euler_to_matrix(0, 0, angle, 'xyz')
        angles_z = matrix_to_euler(R_z, 'xyz')
        np.testing.assert_allclose(angles_z, [0, 0, angle], atol=1e-10, rtol=1e-10)
    
    def test_consistency_with_rpy(self):
        """Test that RPY is equivalent to specific Euler conventions."""
        from robocore.transform.conversions import matrix_to_rpy, rpy_to_matrix
        
        rpy = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
        
        # RPY(roll, pitch, yaw) = Extrinsic ZYX(yaw, pitch, roll)
        # This is because R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R_rpy = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
        R_zyx = euler_to_matrix(rpy[2], rpy[1], rpy[0], 'ZYX')  # reversed order
        
        np.testing.assert_allclose(R_rpy, R_zyx, atol=1e-10, rtol=1e-10,
                                   err_msg="RPY should match extrinsic ZYX with reversed angles")
        
        # Also equivalent to Intrinsic xyz(roll, pitch, yaw)
        R_xyz = euler_to_matrix(rpy[0], rpy[1], rpy[2], 'xyz')
        
        np.testing.assert_allclose(R_rpy, R_xyz, atol=1e-10, rtol=1e-10,
                                   err_msg="RPY should match intrinsic xyz with same angles")


class TestEulerBatchDimensions:
    """Test batch processing with different array shapes."""
    
    def test_single_matrix_2d(self):
        """Test single 3x3 matrix input."""
        R = Rotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
        
        angles = matrix_to_euler(R, 'xyz')
        assert angles.shape == (3,), f"Expected shape (3,), got {angles.shape}"
    
    def test_batch_matrices_3d(self):
        """Test batch of matrices (N, 3, 3)."""
        batch_angles = np.array([
            [0.1, 0.2, 0.3],
            [0.5, -0.3, 0.8],
        ])
        R_batch = Rotation.from_euler('xyz', batch_angles).as_matrix()
        
        angles = matrix_to_euler(R_batch, 'xyz')
        assert angles.shape == (2, 3), f"Expected shape (2, 3), got {angles.shape}"
    
    def test_large_batch(self):
        """Test large batch processing."""
        n = 100
        batch_angles = np.random.uniform(-np.pi, np.pi, (n, 3))
        
        for seq in ['xyz', 'zyx']:
            R_batch = Rotation.from_euler(seq, batch_angles).as_matrix()
            angles_ours = matrix_to_euler(R_batch, seq)
            angles_scipy = Rotation.from_matrix(R_batch).as_euler(seq)
            
            np.testing.assert_allclose(angles_ours, angles_scipy, atol=1e-10, rtol=1e-10,
                                       err_msg=f"Large batch failed for {seq}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
