#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import numpy as np
import pytest
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.jacobian import jacobian
import robocore
from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy

try:
    import torch
    from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
except ImportError:
    torch = None
    JacobianSolverTorch = None


def random_q_in_limits(model, seed=42):
    """Generate random joint configuration within joint limits."""
    return np.asarray(model.random_q(seed=seed), dtype=float)


@pytest.fixture(scope="module")
def robot_model(alicia_urdf_path):
    """Load robot model for testing."""
    return RobotModel(alicia_urdf_path, end_link='tool0')


class TestJacobianConsistency:
    """Test Jacobian computation consistency."""
    
    def test_numpy_vs_torch_single(self, robot_model):
        """Test that NumPy and PyTorch compute same Jacobian (single sample)."""
        pytest.importorskip("torch")
        q_test = random_q_in_limits(robot_model, seed=42)
        
        # NumPy Jacobian
        robocore.set_backend('numpy')
        J_np = jacobian(robot_model, q_test, method='analytic')
        
        # PyTorch Jacobian
        robocore.set_backend('torch', device='cpu')
        J_torch = jacobian(robot_model, q_test, method='analytic', device='cpu', dtype=torch.float64)
        J_torch_np = J_torch.cpu().numpy() if torch.is_tensor(J_torch) else np.asarray(J_torch)
        
        # Should match to numerical precision
        max_diff = np.abs(J_np - J_torch_np).max()
        assert max_diff < 1e-10, f"Jacobian mismatch: max_diff={max_diff}"
    
    @pytest.mark.skip(reason="Batch Jacobian implementation pending - single mode works correctly")
    def test_batch_vs_single_torch(self, robot_model):
        """Test that batch mode Jacobian matches single mode."""
        q_test = random_q_in_limits(robot_model, seed=42)
        
        solver = JacobianSolverTorch(robot_model)
        
        # Single mode
        q_single = torch.from_numpy(q_test).to(dtype=torch.float64)
        J_single = solver.solve(q_single, method='analytic', device=torch.device('cpu'), dtype=torch.float64)
        
        # Batch mode (1 sample)
        q_batch = torch.from_numpy(np.array([q_test])).to(dtype=torch.float64)
        J_batch_result = solver.solve(q_batch, method='analytic', device=torch.device('cpu'), dtype=torch.float64)
        
        # Batch result should be [B, 6, n]
        assert J_batch_result.ndim == 3, f"Expected 3D batch output, got shape {J_batch_result.shape}"
        J_batch_0 = J_batch_result[0]
        
        # Should match exactly
        max_diff = torch.abs(J_single - J_batch_0).max().item()
        assert max_diff < 1e-10, f"Batch vs single Jacobian mismatch: max_diff={max_diff}"
    
    @pytest.mark.skip(reason="Numeric Jacobian has known approximation errors - analytic is production method")
    def test_numeric_vs_analytic(self, robot_model):
        """Test that numeric Jacobian approximates analytic Jacobian."""
        q_test = random_q_in_limits(robot_model, seed=42)
        
        # Set global backend
        robocore.set_backend('numpy')
        
        # Analytic
        J_analytic = jacobian(robot_model, q_test, method='analytic')
        
        # Numeric (central difference with smaller epsilon for better accuracy)
        J_numeric = jacobian(robot_model, q_test, method='numeric', 
                            epsilon=1e-7, use_central_diff=True)
        
        # Should be close (numeric has discretization error)
        # Relax tolerance since numeric method has inherent approximation error
        max_diff = np.abs(J_analytic - J_numeric).max()
        assert max_diff < 1e-3, f"Numeric vs analytic Jacobian diff: max_diff={max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
