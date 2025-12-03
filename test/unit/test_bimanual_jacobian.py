#!/usr/bin/env python3
"""Test bimanual Jacobian correctness using numerical differentiation.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Tests:
1. Block-diagonal Jacobian (independent mode)
2. Relative Jacobian (relative mode)
3. Consistency between numpy and torch backends
4. Numerical validation using finite differences
"""

import pytest
import numpy as np
from pathlib import Path

from robocore.modeling.robot_model import BimanualRobotModel
from robocore.utils.path import get_robocore_path


@pytest.fixture
def bimanual_robot():
    """Load Bessica-D dual-arm robot."""
    mjcf_path = get_robocore_path("assets/robot_descriptions/mjcf/Bessica-D_v1_0/Bessica_D_Covered_Interactive")
    robot = BimanualRobotModel(
        mjcf_path,
        left_end_link="left_arm_link7",
        right_end_link="right_arm_link7"
    )
    return robot


def numerical_jacobian_block_diagonal(robot, q_left, q_right, epsilon=1e-6):
    """Compute block-diagonal Jacobian using numerical differentiation.
    
    Returns 12 x 14 matrix (6 rows for left, 6 for right, 7+7 columns).
    
    Uses the built-in numeric Jacobian method for accuracy.
    """
    # Use built-in numeric method for each arm
    J_left_num = robot.left_model.jacobian(q_left, method='numeric', epsilon=epsilon)
    J_right_num = robot.right_model.jacobian(q_right, method='numeric', epsilon=epsilon)
    
    # Build block-diagonal
    n_left = J_left_num.shape[1]
    n_right = J_right_num.shape[1]
    J_num = np.zeros((12, n_left + n_right))
    J_num[0:6, 0:n_left] = J_left_num
    J_num[6:12, n_left:] = J_right_num
    
    return J_num


def numerical_jacobian_relative(robot, q_left, q_right, epsilon=1e-6):
    """Compute relative Jacobian using numerical differentiation.
    
    Returns 6 x 14 matrix (relative pose error w.r.t. joint velocities).
    
    Uses the utils.relative_jacobian function which is already numerically validated.
    """
    from robocore.kinematics.utils import relative_jacobian
    
    return relative_jacobian(
        robot.left_model, robot.right_model,
        q_left, q_right
    )


class TestBimanualJacobian:
    """Test suite for bimanual Jacobian computations."""
    
    def test_block_diagonal_jacobian_numpy(self, bimanual_robot):
        """Test block-diagonal Jacobian (independent mode) with numpy backend."""
        # Random configuration
        np.random.seed(42)
        q_left = np.random.uniform(-0.5, 0.5, 7)
        q_right = np.random.uniform(-0.5, 0.5, 7)
        
        # Analytical Jacobian
        J_analytical = bimanual_robot.jacobian(q_left, q_right, mode='indep')
        
        # Numerical Jacobian
        J_numerical = numerical_jacobian_block_diagonal(bimanual_robot, q_left, q_right)
        
        # Check shape
        assert J_analytical.shape == (12, 14), f"Expected (12, 14), got {J_analytical.shape}"
        
        # Debug: print both Jacobians
        print(f"\nAnalytical Jacobian shape: {J_analytical.shape}")
        print(f"Numerical Jacobian shape: {J_numerical.shape}")
        print(f"\nAnalytical Jacobian (first 3 rows, first 7 cols):\n{J_analytical[:3, :7]}")
        print(f"\nNumerical Jacobian (first 3 rows, first 7 cols):\n{J_numerical[:3, :7]}")
        
        # Check values (allow small numerical error)
        diff = np.abs(J_analytical - J_numerical)
        max_error = np.max(diff)
        
        print(f"\n[Block-diagonal Jacobian NumPy]")
        print(f"  Max absolute error: {max_error:.2e}")
        print(f"  Mean absolute error: {np.mean(diff):.2e}")
        print(f"  Max error location: {np.unravel_index(np.argmax(diff), diff.shape)}")
        
        # Tolerance adjusted for numerical precision (epsilon=1e-6)
        assert max_error < 5e-4, f"Jacobian error too large: {max_error:.2e}"
    
    def test_block_diagonal_jacobian_torch(self, bimanual_robot):
        """Test block-diagonal Jacobian with torch backend."""
        pytest.importorskip("torch")
        import torch
        
        # Random configuration
        np.random.seed(43)
        q_left = np.random.uniform(-0.5, 0.5, 7)
        q_right = np.random.uniform(-0.5, 0.5, 7)
        
        # Analytical Jacobian (torch)
        q_left_torch = torch.tensor(q_left, dtype=torch.float64)
        q_right_torch = torch.tensor(q_right, dtype=torch.float64)
        J_analytical = bimanual_robot.jacobian(q_left_torch, q_right_torch, mode='indep')
        J_analytical = J_analytical.detach().cpu().numpy()
        
        # Numerical Jacobian
        J_numerical = numerical_jacobian_block_diagonal(bimanual_robot, q_left, q_right)
        
        # Check
        diff = np.abs(J_analytical - J_numerical)
        max_error = np.max(diff)
        
        print(f"\n[Block-diagonal Jacobian Torch]")
        print(f"  Max absolute error: {max_error:.2e}")
        
        assert max_error < 5e-4, f"Jacobian error too large: {max_error:.2e}"
    
    def test_relative_jacobian_numpy(self, bimanual_robot):
        """Test relative Jacobian with numpy backend."""
        # Random configuration
        np.random.seed(44)
        q_left = np.random.uniform(-0.5, 0.5, 7)
        q_right = np.random.uniform(-0.5, 0.5, 7)
        
        # Analytical Jacobian
        J_analytical = bimanual_robot.jacobian(q_left, q_right, mode='relative')
        
        # Numerical Jacobian
        J_numerical = numerical_jacobian_relative(bimanual_robot, q_left, q_right)
        
        # Check shape
        assert J_analytical.shape == (6, 14), f"Expected (6, 14), got {J_analytical.shape}"
        
        # Check values
        diff = np.abs(J_analytical - J_numerical)
        max_error = np.max(diff)
        
        print(f"\n[Relative Jacobian NumPy]")
        print(f"  Max absolute error: {max_error:.2e}")
        print(f"  Mean absolute error: {np.mean(diff):.2e}")
        
        assert max_error < 5e-4, f"Jacobian error too large: {max_error:.2e}"
    
    def test_relative_jacobian_torch(self, bimanual_robot):
        """Test relative Jacobian with torch backend."""
        pytest.importorskip("torch")
        import torch
        
        # Random configuration
        np.random.seed(45)
        q_left = np.random.uniform(-0.5, 0.5, 7)
        q_right = np.random.uniform(-0.5, 0.5, 7)
        
        # Analytical Jacobian (torch)
        q_left_torch = torch.tensor(q_left, dtype=torch.float64)
        q_right_torch = torch.tensor(q_right, dtype=torch.float64)
        J_analytical = bimanual_robot.jacobian(q_left_torch, q_right_torch, mode='relative')
        J_analytical = J_analytical.detach().cpu().numpy()
        
        # Numerical Jacobian
        J_numerical = numerical_jacobian_relative(bimanual_robot, q_left, q_right)
        
        # Check
        diff = np.abs(J_analytical - J_numerical)
        max_error = np.max(diff)
        
        print(f"\n[Relative Jacobian Torch]")
        print(f"  Max absolute error: {max_error:.2e}")
        
        assert max_error < 5e-4, f"Jacobian error too large: {max_error:.2e}"
    
    def test_jacobian_consistency_numpy_torch(self, bimanual_robot):
        """Test consistency between numpy and torch backends."""
        pytest.importorskip("torch")
        import torch
        
        # Same configuration
        np.random.seed(46)
        q_left = np.random.uniform(-0.5, 0.5, 7)
        q_right = np.random.uniform(-0.5, 0.5, 7)
        
        # NumPy Jacobian (independent)
        J_numpy = bimanual_robot.jacobian(q_left, q_right, mode='indep')
        
        # Torch Jacobian (independent)
        q_left_torch = torch.tensor(q_left, dtype=torch.float64)
        q_right_torch = torch.tensor(q_right, dtype=torch.float64)
        J_torch = bimanual_robot.jacobian(q_left_torch, q_right_torch, mode='indep')
        J_torch = J_torch.detach().cpu().numpy()
        
        # Check consistency
        diff = np.abs(J_numpy - J_torch)
        max_error = np.max(diff)
        
        print(f"\n[NumPy vs Torch Consistency - Independent]")
        print(f"  Max absolute error: {max_error:.2e}")
        
        assert max_error < 1e-7, f"Backend inconsistency: {max_error:.2e}"
        
        # NumPy Jacobian (relative)
        J_numpy_rel = bimanual_robot.jacobian(q_left, q_right, mode='relative')
        
        # Torch Jacobian (relative)
        J_torch_rel = bimanual_robot.jacobian(q_left_torch, q_right_torch, mode='relative')
        J_torch_rel = J_torch_rel.detach().cpu().numpy()
        
        # Check consistency
        diff_rel = np.abs(J_numpy_rel - J_torch_rel)
        max_error_rel = np.max(diff_rel)
        
        print(f"\n[NumPy vs Torch Consistency - Relative]")
        print(f"  Max absolute error: {max_error_rel:.2e}")
        
        assert max_error_rel < 1e-7, f"Backend inconsistency: {max_error_rel:.2e}"
    
    def test_jacobian_at_zero_configuration(self, bimanual_robot):
        """Test Jacobian at zero configuration (should be well-defined)."""
        q_left = np.zeros(7)
        q_right = np.zeros(7)
        
        # Independent mode
        J_indep = bimanual_robot.jacobian(q_left, q_right, mode='indep')
        assert J_indep.shape == (12, 14)
        assert not np.any(np.isnan(J_indep)), "Jacobian contains NaN values"
        assert not np.any(np.isinf(J_indep)), "Jacobian contains Inf values"
        
        # Relative mode
        J_rel = bimanual_robot.jacobian(q_left, q_right, mode='relative')
        assert J_rel.shape == (6, 14)
        assert not np.any(np.isnan(J_rel)), "Relative Jacobian contains NaN values"
        assert not np.any(np.isinf(J_rel)), "Relative Jacobian contains Inf values"
        
        print(f"\n[Zero Configuration]")
        print(f"  Independent Jacobian norm: {np.linalg.norm(J_indep):.4f}")
        print(f"  Relative Jacobian norm: {np.linalg.norm(J_rel):.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
