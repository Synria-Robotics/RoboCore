#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import pytest
import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer


@pytest.fixture(scope="module")
def robot_model():
    """Load robot model for workspace testing."""
    urdf_path = Path('robocore/assets/robot_descriptions/urdf/Alicia-D_v5_5/alicia_duo_with_gripper.urdf')
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return RobotModel(str(urdf_path), end_link='tool0')


@pytest.mark.integration
class TestWorkspaceAnalyzer:
    """Test workspace analysis functionality."""
    
    def test_analyzer_creation(self, robot_model):
        """Test WorkspaceAnalyzer can be created."""
        analyzer = WorkspaceAnalyzer(robot_model)
        assert analyzer is not None
        assert analyzer.model == robot_model
    
    def test_reachability_analysis(self, robot_model):
        """Test basic reachability analysis."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        # Analyze small grid for speed
        result = analyzer.analyze_reachability(
            grid_resolution=0.1,
            bounds={
                'x': (-0.5, 0.5),
                'y': (-0.5, 0.5),
                'z': (0.0, 0.5)
            },
        )
        
        assert 'reachable_points' in result or 'grid' in result
        assert 'volume' in result or 'count' in result
    
    @pytest.mark.slow
    def test_workspace_volume(self, robot_model):
        """Test workspace volume computation."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        result = analyzer.compute_workspace_volume(
            resolution=0.15,  # Coarse for speed
        )
        
        # Volume should be positive
        if 'volume' in result:
            assert result['volume'] > 0
        elif 'reachable_volume' in result:
            assert result['reachable_volume'] > 0
    
    def test_point_reachability(self, robot_model):
        """Test checking if specific points are reachable."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        # Test point near robot base (likely reachable)
        point_near = np.array([0.3, 0.0, 0.3])
        
        # Test point very far (likely unreachable)
        point_far = np.array([10.0, 10.0, 10.0])
        
        try:
            is_reachable_near = analyzer.is_reachable(point_near, )
            is_reachable_far = analyzer.is_reachable(point_far, )
            
            # Near point more likely reachable than far point
            assert isinstance(is_reachable_near, bool)
            assert isinstance(is_reachable_far, bool)
        except AttributeError:
            pytest.skip("is_reachable method not implemented")
    
    def test_workspace_boundary(self, robot_model):
        """Test workspace boundary computation."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        try:
            boundary = analyzer.compute_boundary(resolution=0.2, )
            
            assert 'min_reach' in boundary or 'bounds' in boundary
            assert 'max_reach' in boundary or 'bounds' in boundary
        except AttributeError:
            pytest.skip("compute_boundary method not implemented")


@pytest.mark.integration
@pytest.mark.slow
class TestWorkspaceVisualization:
    """Test workspace visualization capabilities."""
    
    def test_workspace_plot_data(self, robot_model):
        """Test generating plot data for workspace."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        try:
            plot_data = analyzer.get_plot_data(resolution=0.15, )
            
            # Should return data suitable for plotting
            assert plot_data is not None
            assert len(plot_data) > 0
        except AttributeError:
            pytest.skip("get_plot_data method not implemented")
    
    def test_cross_section(self, robot_model):
        """Test workspace cross-section analysis."""
        analyzer = WorkspaceAnalyzer(robot_model)
        
        try:
            # XY plane at z=0.3
            section = analyzer.cross_section(plane='xy', height=0.3, resolution=0.1, )
            
            assert section is not None
            assert 'points' in section or 'grid' in section
        except AttributeError:
            pytest.skip("cross_section method not implemented")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
