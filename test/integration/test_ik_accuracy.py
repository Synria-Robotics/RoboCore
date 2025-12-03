#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import torch
import pytest
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.ik_utils.ik_solver_torch import IKSolverTorch
import robocore
from robocore.kinematics.fk import forward_kinematics


def random_q_in_limits(model, seed=42):
    """Generate random joint configuration within joint limits."""
    rng = np.random.default_rng(seed)
    n = model.num_dof
    q = np.zeros(n)
    for js in model._chain_actuated:
        lo, hi = -1.0, 1.0
        if js.limit:
            if js.limit[0] is not None:
                lo = js.limit[0]
            if js.limit[1] is not None:
                hi = js.limit[1]
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo) * 0.5
        q[js.index] = rng.uniform(mid - span, mid + span)
    return q


@pytest.fixture(scope="module")
def robot_model():
    """Load robot model for testing."""
    urdf_path = Path('robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf')
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return RobotModel(str(urdf_path), end_link='tool0')


class TestIKAccuracy:
    """Test IK solver accuracy across backends."""
    
    def test_single_sample_consistency(self, robot_model):
        """Test that single-sample IK is consistent across backends."""
        seed = 42
        q_target = random_q_in_limits(robot_model, seed=seed)
        target_pose = forward_kinematics(robot_model, q_target, return_end=True)
        q_init = random_q_in_limits(robot_model, seed=seed + 1)
        
        # NumPy solution
        robocore.set_backend('numpy')
        result_np = inverse_kinematics(
            robot_model, target_pose, q_init,
            method='dls',
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4,
        )
        
        # PyTorch single-sample solution
        robocore.set_backend('torch', device='cpu')
        result_torch = inverse_kinematics(
            robot_model, target_pose, q_init,
            method='dls',
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4,
            torch_device='cpu',
        )
        
        # Both should succeed
        assert result_np['success'], "NumPy IK failed"
        assert result_torch['success'], "PyTorch IK failed"
        
        # Errors should be similar (within tolerance)
        assert result_np['pos_err'] < 1e-3, f"NumPy position error too large: {result_np['pos_err']}"
        assert result_torch['pos_err'] < 1e-3, f"PyTorch position error too large: {result_torch['pos_err']}"
    
    def test_batch_mode_accuracy(self, robot_model):
        """Test that batch mode achieves good success rate."""
        n_samples = 64
        seed = 42
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Generate test cases
        q_batch = []
        target_poses = []
        for i in range(n_samples):
            q = random_q_in_limits(robot_model, seed=seed+i)
            T = forward_kinematics(robot_model, q, return_end=True)
            q_init = random_q_in_limits(robot_model, seed=seed+n_samples+i)
            
            q_batch.append(q_init)
            target_poses.append(T)
        
        q_batch = np.array(q_batch)
        
        # NumPy baseline
        np_success_count = 0
        for i in range(n_samples):
            res = inverse_kinematics(
                robot_model, target_poses[i], q_batch[i],
                , method='dls',
                max_iters=100, pos_tol=1e-4, ori_tol=1e-4
            )
            if res['success']:
                np_success_count += 1
        
        np_success_rate = np_success_count / n_samples
        
        # PyTorch batch
        solver_torch = IKSolverTorch(
            robot_model, max_iters=100, pos_tol=1e-4, ori_tol=1e-4,
            device=torch.device(device), dtype=torch.float64
        )
        
        target_batch = torch.stack([torch.from_numpy(p) for p in target_poses]).to(dtype=torch.float64, device=device)
        q_init_batch = torch.from_numpy(q_batch).to(dtype=torch.float64, device=device)
        
        torch_result = solver_torch.solve(target_batch, q_init_batch, method='dls')
        torch_success = torch_result['success'].cpu().numpy()
        torch_success_rate = torch_success.sum() / n_samples
        
        # Assert that success rates are comparable (within 5%)
        success_rate_diff = abs(torch_success_rate - np_success_rate)
        assert success_rate_diff < 0.05, (
            f"Success rate mismatch: NumPy={np_success_rate:.1%}, "
            f"PyTorch={torch_success_rate:.1%}, diff={success_rate_diff:.1%}"
        )
        
        # Both should achieve >70% success rate on this test set
        assert np_success_rate > 0.70, f"NumPy success rate too low: {np_success_rate:.1%}"
        assert torch_success_rate > 0.70, f"PyTorch success rate too low: {torch_success_rate:.1%}"


if __name__ == '__main__':
    # Allow running as standalone script
    pytest.main([__file__, '-v'])
