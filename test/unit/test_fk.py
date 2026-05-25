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
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy
import robocore

try:
    import torch
    from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch
except ImportError:
    torch = None
    FKSolverTorch = None


def random_q_in_limits(model, seed=42):
    """Generate random joint configuration within joint limits."""
    return np.asarray(model.random_q(seed=seed), dtype=float)


@pytest.fixture(scope="module")
def robot_model(alicia_urdf_path):
    """Load robot model for testing."""
    return RobotModel(alicia_urdf_path, end_link='tool0')


class TestFKConsistency:
    """Test forward kinematics consistency."""
    
    def test_numpy_vs_torch_single(self, robot_model):
        """Test that NumPy and PyTorch compute same FK (single sample)."""
        pytest.importorskip("torch")
        q_test = random_q_in_limits(robot_model, seed=42)
        
        # NumPy FK
        robocore.set_backend('numpy')
        T_np = forward_kinematics(robot_model, q_test, return_end=True)
        
        # PyTorch FK
        robocore.set_backend('torch', device='cpu')
        T_torch = forward_kinematics(robot_model, q_test, return_end=True, 
                                     device='cpu', dtype=torch.float64)
        T_torch_np = T_torch.cpu().numpy() if torch.is_tensor(T_torch) else np.asarray(T_torch)
        
        # Should match to numerical precision
        max_diff = np.abs(T_np - T_torch_np).max()
        assert max_diff < 1e-9, f"FK mismatch: max_diff={max_diff}"
    
    def test_batch_mode_torch(self, robot_model):
        """Test PyTorch batch FK computation."""
        pytest.importorskip("torch")
        n_samples = 10
        q_batch_list = [random_q_in_limits(robot_model, seed=i) for i in range(n_samples)]
        
        # Individual FK
        T_individual = []
        for q in q_batch_list:
            robocore.set_backend('torch', device='cpu')
            T = forward_kinematics(robot_model, q, return_end=True,
                                  device='cpu', dtype=torch.float64)
            T_individual.append(T)
        
        # Batch FK
        solver = FKSolverTorch(robot_model)
        q_batch_tensor = torch.from_numpy(np.array(q_batch_list)).to(dtype=torch.float64)
        T_batch = solver.solve(q_batch_tensor, return_end_only=True, device='cpu', dtype=torch.float64)
        
        # Compare each sample
        for i in range(n_samples):
            T_ind = T_individual[i]
            T_bat = T_batch['end'][i] if isinstance(T_batch, dict) else T_batch[i]

            T_ind_np = T_ind.detach().cpu().numpy() if torch.is_tensor(T_ind) else np.asarray(T_ind)
            T_bat_np = T_bat.detach().cpu().numpy() if torch.is_tensor(T_bat) else np.asarray(T_bat)
            max_diff = np.abs(T_ind_np - T_bat_np).max()
            # Relax tolerance for batch FK - numerical differences are acceptable
            assert max_diff < 1e-6, f"Batch FK sample {i} mismatch: max_diff={max_diff}"
    
    def test_zero_configuration(self, robot_model):
        """Test FK at zero configuration."""
        q_zero = np.zeros(robot_model.num_chain_dof)
        
        robocore.set_backend('numpy')
        T_np = forward_kinematics(robot_model, q_zero, return_end=True)
        
        # Should be valid transformation matrices
        assert T_np.shape == (4, 4), "NumPy FK output shape incorrect"
        
        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(T_np[3, :], [0, 0, 0, 1]), "NumPy FK homogeneous row incorrect"

        if torch is not None:
            robocore.set_backend('torch', device='cpu')
            T_torch = forward_kinematics(robot_model, q_zero, return_end=True,
                                         device='cpu', dtype=torch.float64)
            T_torch_np = T_torch.cpu().numpy() if torch.is_tensor(T_torch) else np.asarray(T_torch)
            assert T_torch_np.shape == (4, 4), "PyTorch FK output shape incorrect"
            assert np.allclose(T_torch_np[3, :], [0., 0., 0., 1.]), \
                "PyTorch FK homogeneous row incorrect"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
