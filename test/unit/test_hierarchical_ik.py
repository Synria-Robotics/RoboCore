"""Unit tests for Phase 2.2: Hierarchical task priority IK.

Copyright (c) 2025 Synria Robotics Co., Ltd.
License: MIT

Tests:
1. Nullspace projector properties
2. Hierarchical IK convergence
3. Primary task strict satisfaction
4. Secondary task optimization in nullspace
"""

import pytest
import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.bimanual import (
    nullspace_projector,
    dual_ik_hierarchical,
    relative_pose_error,
    Task,
)
from robocore.transform.se3 import make_transform
from robocore.transform.so3 import rpy_to_matrix


class TestNullspaceProjector:
    """Test nullspace projection matrix properties."""
    
    def test_identity_nullspace(self):
        """Nullspace of zero matrix should be identity."""
        J = np.zeros((2, 5))
        N = nullspace_projector(J)
        
        np.testing.assert_allclose(N, np.eye(5), atol=1e-10)
    
    def test_full_rank_nullspace(self):
        """Nullspace of full-rank square matrix should be zero."""
        J = np.eye(5)
        N = nullspace_projector(J)
        
        # N should project everything to zero
        np.testing.assert_allclose(N, np.zeros((5, 5)), atol=1e-6)
    
    def test_projection_property(self):
        """N should be idempotent: N @ N = N."""
        J = np.random.randn(3, 7)
        N = nullspace_projector(J)
        
        np.testing.assert_allclose(N @ N, N, atol=1e-6)
    
    def test_orthogonal_to_jacobian(self):
        """N should annihilate J: J @ N ≈ 0."""
        J = np.random.randn(4, 8)
        N = nullspace_projector(J)
        
        np.testing.assert_allclose(J @ N, np.zeros((4, 8)), atol=1e-6)
    
    def test_preserves_nullspace_vectors(self):
        """Vectors in nullspace should be unchanged: N @ v = v."""
        # Create J with known nullspace
        J = np.array([[1, 0, 0, 2],
                      [0, 1, 0, 3]])
        
        N = nullspace_projector(J)
        
        # Vector [0, 0, 1, 0] is in nullspace of J
        v_null = np.array([0, 0, 1, 0])
        np.testing.assert_allclose(N @ v_null, v_null, atol=1e-6)


@pytest.mark.skip(reason="dual_ik_hierarchical not yet implemented (Phase 2.2)")
class TestHierarchicalIK:
    """Test hierarchical task priority IK solver."""
    
    @pytest.fixture
    def dual_arms(self, bessica_urdf_path):
        """Load dual-arm models."""
        left = RobotModel(bessica_urdf_path, end_link='left_arm_gripper_left_finger')
        right = RobotModel(bessica_urdf_path, end_link='right_arm_gripper_right_finger')
        return left, right
    
    def test_single_priority_level(self, dual_arms):
        """Single priority level should behave like weighted IK."""
        left, right = dual_arms
        
        # Simple configuration
        q_init_left = np.array([0.1, 0.2, 0.0, -0.6, 0.0, 0.2, 0.0])
        q_init_right = np.array([0.1, -0.2, 0.0, -0.6, 0.0, -0.2, 0.0])
        
        # Get initial relative pose
        T_L = left.fk(q_init_left, return_end=True)
        T_R = right.fk(q_init_right, return_end=True)
        
        if hasattr(T_L, 'detach'):
            T_L = T_L.detach().cpu().numpy()
            T_R = T_R.detach().cpu().numpy()
        
        T_rel = np.linalg.inv(T_L) @ T_R
        
        # Perturb
        q_pert_left = q_init_left + np.array([0.15, 0.1, -0.08, 0.12, -0.08, 0.08, 0.0])
        q_pert_right = q_init_right + np.array([0.15, -0.1, -0.08, 0.12, 0.08, -0.08, 0.0])
        
        # Single task: restore relative pose
        tasks = [
            Task(
                type='relative',
                target=T_rel,
                row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only
            ),
        ]
        
        result = dual_ik_hierarchical(
            left, right,
            task_groups=[tasks],  # Single priority level
            q0_left=q_pert_left,
            q0_right=q_pert_right,
            max_iters=50,
            tol_primary=5e-3,
            verbose=False
        )
        
        assert result['success'], "Single-level task should converge"
        assert result['primary_residual'] < 5e-3, "Primary residual should be within tolerance"
    
    def test_two_priority_levels(self, dual_arms):
        """Two priority levels: primary strict, secondary optimized."""
        left, right = dual_arms
        
        # Initial configuration
        q_init_left = np.array([0.0, 0.15, 0.0, -0.7, 0.0, 0.15, 0.0])
        q_init_right = np.array([0.0, -0.15, 0.0, -0.7, 0.0, -0.15, 0.0])
        
        # Record initial relative pose (primary constraint)
        T_L_init = left.fk(q_init_left, return_end=True)
        T_R_init = right.fk(q_init_right, return_end=True)
        
        if hasattr(T_L_init, 'detach'):
            T_L_init = T_L_init.detach().cpu().numpy()
            T_R_init = T_R_init.detach().cpu().numpy()
        
        T_rel_target = np.linalg.inv(T_L_init) @ T_R_init
        
        # Secondary target: move left arm slightly
        T_left_target = T_L_init.copy()
        T_left_target[0, 3] += 0.05  # +5cm in X
        
        # Start from perturbed config
        q_pert_left = q_init_left + np.array([0.2, 0.1, -0.1, 0.15, -0.1, 0.1, 0.0])
        q_pert_right = q_init_right + np.array([0.2, -0.1, -0.1, 0.15, 0.1, -0.1, 0.0])
        
        # Define hierarchical tasks
        primary_tasks = [
            Task(
                type='relative',
                target=T_rel_target,
                row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only
            ),
        ]
        
        secondary_tasks = [
            Task(
                type='absolute_left',
                target=T_left_target,
                row_mask=np.array([1, 1, 1, 0, 0, 0])  # Position only
            ),
        ]
        
        result = dual_ik_hierarchical(
            left, right,
            task_groups=[primary_tasks, secondary_tasks],
            q0_left=q_pert_left,
            q0_right=q_pert_right,
            max_iters=100,
            tol_primary=5e-3,
            tol_secondary=1e-2,
            verbose=False
        )
        
        assert result['success'], "Primary task should converge"
        assert result['primary_residual'] < 5e-3, "Primary should be strictly satisfied"
        
        # Validate primary constraint
        q_L_final = np.array(result['q_left'])
        q_R_final = np.array(result['q_right'])
        
        T_L_final = left.fk(q_L_final, return_end=True)
        T_R_final = right.fk(q_R_final, return_end=True)
        
        if hasattr(T_L_final, 'detach'):
            T_L_final = T_L_final.detach().cpu().numpy()
            T_R_final = T_R_final.detach().cpu().numpy()
        
        e_rel = relative_pose_error(T_L_final, T_R_final, T_rel_target)
        e_rel_pos = np.linalg.norm(e_rel[:3])
        
        assert e_rel_pos < 0.01, f"Relative position error {e_rel_pos*1000:.1f}mm should be < 10mm"
    
    def test_primary_not_affected_by_secondary(self, dual_arms):
        """Adding secondary task should not degrade primary solution."""
        left, right = dual_arms
        
        q_init_left = np.array([0.1, 0.2, 0.0, -0.6, 0.0, 0.2, 0.0])
        q_init_right = np.array([0.1, -0.2, 0.0, -0.6, 0.0, -0.2, 0.0])
        
        T_L = left.fk(q_init_left, return_end=True)
        T_R = right.fk(q_init_right, return_end=True)
        
        if hasattr(T_L, 'detach'):
            T_L = T_L.detach().cpu().numpy()
            T_R = T_R.detach().cpu().numpy()
        
        T_rel = np.linalg.inv(T_L) @ T_R
        
        q_pert_left = q_init_left + 0.2 * np.random.randn(7)
        q_pert_right = q_init_right + 0.2 * np.random.randn(7)
        
        # Solve with primary only
        primary_task = Task(type='relative', target=T_rel, row_mask=np.array([1,1,1,0,0,0]))
        
        result_primary_only = dual_ik_hierarchical(
            left, right,
            task_groups=[[primary_task]],
            q0_left=q_pert_left,
            q0_right=q_pert_right,
            max_iters=50,
            verbose=False
        )
        
        # Solve with primary + secondary
        secondary_task = Task(
            type='absolute_left',
            target=T_L,
            row_mask=np.array([1,1,1,0,0,0])
        )
        
        result_with_secondary = dual_ik_hierarchical(
            left, right,
            task_groups=[[primary_task], [secondary_task]],
            q0_left=q_pert_left,
            q0_right=q_pert_right,
            max_iters=50,
            verbose=False
        )
        
        # Primary residual should be similar or better
        assert result_with_secondary['primary_residual'] <= result_primary_only['primary_residual'] * 1.5, \
            "Adding secondary task should not significantly degrade primary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
