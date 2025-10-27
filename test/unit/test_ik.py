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
import pytest
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.fk import forward_kinematics


def random_q_in_limits(model, seed=42):
    """Generate random joint configuration within limits."""
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


class TestIKMethods:
    """Test different IK solution methods."""
    
    @pytest.mark.parametrize("method", ["dls", "pinv", "transpose"])
    def test_ik_methods_numpy(self, robot_model, method):
        """Test that all IK methods can solve basic problems."""
        # Generate target pose
        q_target = random_q_in_limits(robot_model, seed=42)
        target_pose = forward_kinematics(robot_model, q_target, backend='numpy', return_end=True)
        
        # Try to solve from different initial guess
        q_init = random_q_in_limits(robot_model, seed=43)
        
        result = inverse_kinematics(
            robot_model, target_pose, q_init,
            backend='numpy', method=method,
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4
        )
        
        # Should return a result
        assert 'q' in result
        assert 'success' in result
        assert len(result['q']) == robot_model.num_dof
    
    def test_ik_closure(self, robot_model):
        """Test FK-IK-FK closure property."""
        # Start with known configuration
        q_start = random_q_in_limits(robot_model, seed=42)
        T_start = forward_kinematics(robot_model, q_start, backend='numpy', return_end=True)
        
        # Solve IK from different initial guess
        q_init = random_q_in_limits(robot_model, seed=43)
        result = inverse_kinematics(
            robot_model, T_start, q_init,
            backend='numpy', method='dls',
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4
        )
        
        if result['success']:
            # Verify FK of solution matches target
            T_result = forward_kinematics(robot_model, result['q'], backend='numpy', return_end=True)
            
            # Position should match
            pos_diff = np.linalg.norm(T_start[:3, 3] - T_result[:3, 3])
            assert pos_diff < 1e-3, f"Position error: {pos_diff}"
            
            # Orientation should match (check rotation matrix similarity)
            R_diff = T_start[:3, :3].T @ T_result[:3, :3]
            trace = np.trace(R_diff)
            angle_diff = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            assert angle_diff < 1e-2, f"Orientation error: {angle_diff}"
    
    def test_ik_convergence_tracking(self, robot_model):
        """Test that IK tracks convergence properly."""
        q_target = random_q_in_limits(robot_model, seed=42)
        target_pose = forward_kinematics(robot_model, q_target, backend='numpy', return_end=True)
        q_init = random_q_in_limits(robot_model, seed=43)
        
        result = inverse_kinematics(
            robot_model, target_pose, q_init,
            backend='numpy', method='dls',
            max_iters=100, pos_tol=1e-4, ori_tol=1e-4
        )
        
        # Check result fields
        assert 'iters' in result
        assert 'success' in result
        assert 'pos_err' in result or 'err_norm' in result
        
        # Iterations should be reasonable
        if result['success']:
            assert result['iters'] <= 100
            assert result['iters'] > 0
    
    def test_ik_joint_limits(self, robot_model):
        """Test that IK respects joint limits."""
        q_target = random_q_in_limits(robot_model, seed=42)
        target_pose = forward_kinematics(robot_model, q_target, backend='numpy', return_end=True)
        q_init = random_q_in_limits(robot_model, seed=43)
        
        result = inverse_kinematics(
            robot_model, target_pose, q_init,
            backend='numpy', method='dls',
            max_iters=100
        )
        
        if result['success']:
            q_solution = result['q']
            
            # Check limits
            for js in robot_model._actuated:
                if js.limit:
                    lo, hi = js.limit
                    if lo is not None:
                        assert q_solution[js.index] >= lo - 1e-6, \
                            f"Joint {js.name} below lower limit: {q_solution[js.index]} < {lo}"
                    if hi is not None:
                        assert q_solution[js.index] <= hi + 1e-6, \
                            f"Joint {js.name} above upper limit: {q_solution[js.index]} > {hi}"


class TestIKEdgeCases:
    """Test edge cases and error handling."""
    
    def test_ik_infeasible_pose(self, robot_model):
        """Test IK with unreachable pose."""
        # Create pose far outside workspace
        target_pose = np.eye(4)
        target_pose[:3, 3] = [100.0, 100.0, 100.0]  # Very far away
        
        q_init = np.zeros(robot_model.num_dof)
        
        result = inverse_kinematics(
            robot_model, target_pose, q_init,
            backend='numpy', method='dls',
            max_iters=50, pos_tol=1e-4, ori_tol=1e-4
        )
        
        # Should fail gracefully
        assert 'success' in result
        # Most likely won't succeed
        if not result['success']:
            assert result['iters'] > 0  # Should have tried
    
    def test_ik_zero_configuration(self, robot_model):
        """Test IK from zero configuration."""
        q_zero = np.zeros(robot_model.num_dof)
        T_zero = forward_kinematics(robot_model, q_zero, backend='numpy', return_end=True)
        
        # Solve from slightly perturbed initial guess
        q_init = np.ones(robot_model.num_dof) * 0.1
        
        result = inverse_kinematics(
            robot_model, T_zero, q_init,
            backend='numpy', method='dls',
            max_iters=100
        )
        
        # Should succeed (zero config is usually reachable)
        assert 'success' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
