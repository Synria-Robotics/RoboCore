#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import pytest
from pathlib import Path
from robocore.modeling.robot_model import RobotModel


@pytest.fixture(scope="module")
def urdf_path():
    """Path to test URDF file."""
    path = Path('robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf')
    if not path.exists():
        pytest.skip(f"URDF not found: {path}")
    return str(path)


class TestRobotModel:
    """Test RobotModel loading and properties."""
    
    def test_model_loading(self, urdf_path):
        """Test that robot model loads successfully."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        assert model is not None
        assert model.urdf_path == urdf_path
        assert model.end_link == 'tool0'
    
    def test_dof(self, urdf_path):
        """Test DOF calculation."""
        model = RobotModel(urdf_path, end_link='tool0')
        # Use unified config space size
        nq = model.num_dof
        
        # Alicia-D should have at least 6 DOF (may have more in unified space)
        assert nq >= 6
        assert nq > 0
    
    def test_actuated_joints(self, urdf_path):
        """Test actuated joint enumeration."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        # Get chain joint indices
        chain_indices = model._get_joint_indices(model.base_link, model.end_link)
        assert len(chain_indices) > 0
        
        # Check joint properties
        for idx in chain_indices:
            js = model.joint_list[idx]
            assert js.name is not None
            assert idx >= 0
            assert js.joint_type in ('revolute', 'prismatic')
    
    def test_joint_chain(self, urdf_path):
        """Test kinematic chain construction."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        chain = model._chain_joints
        chain_indices = model._get_joint_indices(model.base_link, model.end_link)
        assert len(chain) >= len(chain_indices)  # May include fixed joints
        
        # Check chain continuity
        for joint in chain:
            assert joint.parent is not None
            assert joint.child is not None
    
    def test_joint_limits(self, urdf_path):
        """Test joint limit retrieval."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        chain_indices = model._get_joint_indices(model.base_link, model.end_link)
        for idx in chain_indices:
            js = model.joint_list[idx]
            if js.limit:
                lo, hi = js.limit
                # Limits should be reasonable
                assert lo < hi or (lo is None and hi is None)
    
    def test_base_and_end_links(self, urdf_path):
        """Test base and end link identification."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        assert model.base_link is not None
        assert model.end_link == 'tool0'
        assert len(model.base_link) > 0
    
    def test_invalid_end_link(self, urdf_path):
        """Test that invalid end link raises error."""
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            RobotModel(urdf_path, end_link='nonexistent_link')
    
    def test_model_caching(self, urdf_path):
        """Test that multiple loads work correctly."""
        model1 = RobotModel(urdf_path, end_link='tool0')
        model2 = RobotModel(urdf_path, end_link='tool0')
        
        # Should have same properties
        assert model1.num_dof == model2.num_dof
        assert model1.base_link == model2.base_link
        assert model1.end_link == model2.end_link


class TestJointSpec:
    """Test JointSpec data structure."""
    
    def test_joint_spec_attributes(self, urdf_path):
        """Test that JointSpec has required attributes."""
        model = RobotModel(urdf_path, end_link='tool0')
        
        chain_indices = model._get_joint_indices(model.base_link, model.end_link)
        for idx in chain_indices:
            js = model.joint_list[idx]
            # Required attributes
            assert hasattr(js, 'name')
            assert hasattr(js, 'joint_type')
            assert hasattr(js, 'axis')
            assert hasattr(js, 'origin_rpy')
            assert hasattr(js, 'origin_xyz')
            
            # Check types
            assert isinstance(js.name, str)
            assert isinstance(idx, int)
            assert js.joint_type in ('revolute', 'prismatic', 'fixed')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
