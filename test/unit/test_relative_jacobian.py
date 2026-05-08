"""Unit tests for Phase 2.1: Relative pose Jacobian and error computation.

Copyright (c) 2025 Synria Robotics Co., Ltd.
License: MIT

Tests:
1. Relative pose error correctness
2. Adjoint matrix properties
3. Analytical vs numerical Jacobian consistency
4. Task descriptor validation
"""

import pytest
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import (
    relative_pose_error,
    adjoint_matrix,
    relative_jacobian,
    Task,
)
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import rpy_to_matrix


def transform_matrix(position, rpy):
    """Helper to create transformation matrix from position and RPY angles."""
    R = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    if hasattr(R, 'detach'):
        R = R.detach().cpu().numpy()
    return make_transform(R, np.array(position))


class TestRelativePoseError:
    """Test relative pose error computation."""
    
    def test_zero_error_identity(self):
        """Error should be zero when current = desired."""
        T_left = np.eye(4)
        T_right = transform_matrix([0.5, 0, 0], [0, 0, 0])  # 50cm to right
        
        # Desired = current relative pose
        T_rel_desired = np.linalg.inv(T_left) @ T_right
        
        error = relative_pose_error(T_left, T_right, T_rel_desired)
        assert error.shape == (6,)
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-10)
    
    def test_position_error_only(self):
        """Pure translation error should only affect position components."""
        T_left = np.eye(4)
        T_right = transform_matrix([0.5, 0, 0], [0, 0, 0])
        T_rel_desired = transform_matrix([0.6, 0, 0], [0, 0, 0])  # 10cm difference
        
        error = relative_pose_error(T_left, T_right, T_rel_desired)
        
        # Position error should be 0.1m in x
        np.testing.assert_allclose(error[:3], [0.1, 0, 0], atol=1e-6)
        # Orientation error should be ~zero
        np.testing.assert_allclose(error[3:], [0, 0, 0], atol=1e-6)
    
    def test_orientation_error_only(self):
        """Pure rotation error should only affect orientation components."""
        T_left = np.eye(4)
        T_right = transform_matrix([0.5, 0, 0], [0, 0, 0])
        
        # Desired: same position, but 90deg rotation around Z
        T_rel_desired = transform_matrix([0.5, 0, 0], [0, 0, np.pi/2])
        
        error = relative_pose_error(T_left, T_right, T_rel_desired)
        
        # Position error should be zero
        np.testing.assert_allclose(error[:3], [0, 0, 0], atol=1e-6)
        # Orientation error should be pi/2 around Z
        np.testing.assert_allclose(error[3:], [0, 0, np.pi/2], atol=1e-5)
    
    def test_left_frame_consistency(self):
        """Error should be in left arm's coordinate frame."""
        # Left arm rotated 90deg around Z
        T_left = transform_matrix([0, 0, 0], [0, 0, np.pi/2])
        T_right = transform_matrix([0, 0.5, 0], [0, 0, np.pi/2])  # 50cm in world +Y
        
        # In left's frame (rotated 90deg), world +Y = left's -X
        # So right is at [-0.5, 0, 0] in left's frame
        T_rel_current = np.linalg.inv(T_left) @ T_right
        expected_pos = T_rel_current[0:3, 3]
        
        T_rel_desired = T_rel_current.copy()
        error = relative_pose_error(T_left, T_right, T_rel_desired)
        
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-10)


class TestAdjointMatrix:
    """Test adjoint matrix computation."""
    
    def test_identity_transform(self):
        """Adjoint of identity should be 6x6 identity."""
        Ad = adjoint_matrix(np.eye(4))
        np.testing.assert_allclose(Ad, np.eye(6), atol=1e-10)
    
    def test_pure_translation(self):
        """Adjoint structure for pure translation."""
        T = transform_matrix([1, 2, 3], [0, 0, 0])
        Ad = adjoint_matrix(T)
        
        # R blocks should be identity
        np.testing.assert_allclose(Ad[0:3, 0:3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(Ad[3:6, 3:6], np.eye(3), atol=1e-10)
        
        # Upper-right should be zero
        np.testing.assert_allclose(Ad[0:3, 3:6], np.zeros((3, 3)), atol=1e-10)
        
        # Lower-left should be skew(p)
        p = np.array([1, 2, 3])
        expected_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        np.testing.assert_allclose(Ad[3:6, 0:3], expected_skew, atol=1e-10)
    
    def test_pure_rotation(self):
        """Adjoint structure for pure rotation."""
        T = transform_matrix([0, 0, 0], [0, 0, np.pi/4])
        Ad = adjoint_matrix(T)
        
        R = T[0:3, 0:3]
        # All rotation blocks should match
        np.testing.assert_allclose(Ad[0:3, 0:3], R, atol=1e-10)
        np.testing.assert_allclose(Ad[3:6, 3:6], R, atol=1e-10)
        
        # Lower-left should be zero (no translation)
        np.testing.assert_allclose(Ad[3:6, 0:3], np.zeros((3, 3)), atol=1e-10)
    
    def test_composition_property(self):
        """Ad(T1 @ T2) = Ad(T1) @ Ad(T2)."""
        T1 = transform_matrix([1, 0, 0], [0, 0, np.pi/6])
        T2 = transform_matrix([0, 1, 0], [0, np.pi/4, 0])
        
        Ad_T1 = adjoint_matrix(T1)
        Ad_T2 = adjoint_matrix(T2)
        Ad_T12 = adjoint_matrix(T1 @ T2)
        
        np.testing.assert_allclose(Ad_T12, Ad_T1 @ Ad_T2, atol=1e-10)


class TestRelativeJacobian:
    """Test analytical relative Jacobian vs numerical differentiation."""
    
    @pytest.fixture
    def dual_arms(self):
        """Load dual-arm models."""
        import os
        from pathlib import Path
        
        # Use Bessica-D dual-arm URDF
        base_path = Path(__file__).parent.parent.parent
        urdf_path = base_path / "robocore/assets/robot_descriptions/urdf/Bessica-D_v1_0/Bessica-D_Covered.urdf"
        
        if not urdf_path.exists():
            pytest.skip(f"Bessica URDF not found: {urdf_path}")
        
        # Load left and right arms with their respective end links
        left = RobotModel(str(urdf_path), end_link='left_arm_gripper_left_finger')
        right = RobotModel(str(urdf_path), end_link='right_arm_gripper_left_finger')
        return left, right
    
    def test_jacobian_shape(self, dual_arms):
        """Jacobian should be 6 x (nL + nR)."""
        left, right = dual_arms
        q_left = np.array([0.1, 0.2, 0.3, -0.5, 0.1, 0.2, 0.0])
        q_right = np.array([0.1, -0.2, 0.3, -0.5, 0.1, -0.2, 0.0])
        
        J_rel = relative_jacobian(left, right, q_left, q_right)
        
        assert J_rel.shape == (6, 14)
    
    def test_numerical_consistency(self, dual_arms):
        """Analytical Jacobian should match numerical differentiation."""
        left, right = dual_arms
        q_left = np.array([0.2, 0.3, -0.1, -0.8, 0.2, 0.3, 0.0])
        q_right = np.array([0.2, -0.3, -0.1, -0.8, 0.2, -0.3, 0.0])
        
        # Analytical Jacobian
        J_analytical = relative_jacobian(left, right, q_left, q_right)
        
        # Numerical Jacobian via finite differences
        eps = 1e-6
        
        def rel_pose_vector(qL, qR):
            """Convert relative pose to 6D vector."""
            TL = left.fk(qL, return_end=True)
            TR = right.fk(qR, return_end=True)
            if hasattr(TL, 'detach'):
                TL = TL.detach().cpu().numpy()
                TR = TR.detach().cpu().numpy()
            T_rel = np.linalg.inv(TL) @ TR
            
            p = T_rel[0:3, 3]
            from robocore.transform.conversions import matrix_to_axis_angle
            axis, angle = matrix_to_axis_angle(T_rel[0:3, 0:3])
            return np.concatenate([p, axis * angle])
        
        q_combined = np.concatenate([q_left, q_right])
        J_numerical = np.zeros((6, 14))
        
        p0 = rel_pose_vector(q_left, q_right)
        
        for i in range(14):
            q_plus = q_combined.copy()
            q_plus[i] += eps
            
            qL_p = q_plus[:7]
            qR_p = q_plus[7:]
            
            p_plus = rel_pose_vector(qL_p, qR_p)
            J_numerical[:, i] = (p_plus - p0) / eps
        
        # Compare (allow some numerical error)
        np.testing.assert_allclose(J_analytical, J_numerical, atol=1e-4, rtol=1e-3)


class TestTaskDescriptor:
    """Test Task dataclass validation."""
    
    def test_valid_task_creation(self):
        """Valid task should initialize without errors."""
        T = np.eye(4)
        task = Task(type='absolute_left', target=T, weight=2.0, priority=1)
        
        assert task.type == 'absolute_left'
        assert task.weight == 2.0
        assert task.priority == 1
    
    def test_invalid_task_type(self):
        """Invalid type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid task type"):
            Task(type='invalid_type', target=np.eye(4))
    
    def test_invalid_target_shape(self):
        """Non-4x4 target should raise ValueError."""
        with pytest.raises(ValueError, match="Target must be 4x4"):
            Task(type='relative', target=np.eye(3))
    
    def test_negative_weight(self):
        """Negative weight should raise ValueError."""
        with pytest.raises(ValueError, match="Weight must be positive"):
            Task(type='absolute_left', target=np.eye(4), weight=-1.0)
    
    def test_invalid_row_mask(self):
        """Invalid row_mask shape should raise ValueError."""
        with pytest.raises(ValueError, match="row_mask must be"):
            Task(type='relative', target=np.eye(4), row_mask=[1, 1, 1])
    
    def test_partial_constraint_mask(self):
        """Valid partial constraint mask."""
        task = Task(
            type='relative',
            target=np.eye(4),
            row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only
        )
        assert task.row_mask.sum() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
