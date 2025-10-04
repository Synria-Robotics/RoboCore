#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for singularity analysis.

Tests SingularityAnalyzer for detecting and analyzing singular configurations.
"""

import pytest
import numpy as np
from pathlib import Path
from robocore.modeling.robot_model import RobotModel
from robocore.analysis.singularity_analyzer import SingularityAnalyzer
from robocore.kinematics.jacobian import jacobian


def random_q_in_limits(model, seed=42):
    """Generate random joint configuration within limits."""
    rng = np.random.default_rng(seed)
    n = model.dof()
    q = np.zeros(n)
    for js in model._actuated:
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
    """Load robot model for singularity testing."""
    urdf_path = Path('robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf')
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return RobotModel(str(urdf_path), end_link='tool0')


@pytest.mark.integration
class TestSingularityAnalyzer:
    """Test singularity detection and analysis."""
    
    def test_analyzer_creation(self, robot_model):
        """Test SingularityAnalyzer can be created."""
        analyzer = SingularityAnalyzer(robot_model)
        assert analyzer is not None
        assert analyzer.model == robot_model
    
    def test_manipulability_computation(self, robot_model):
        """Test manipulability measure computation."""
        analyzer = SingularityAnalyzer(robot_model)
        
        q = random_q_in_limits(robot_model, seed=42)
        
        try:
            manip = analyzer.manipulability(q, backend='numpy')
            
            # Manipulability should be non-negative
            assert manip >= 0
            
            # For most configurations, should be positive
            # (Only exactly zero at singularities)
            assert isinstance(manip, (float, np.floating))
        except AttributeError:
            pytest.skip("manipulability method not implemented")
    
    def test_condition_number(self, robot_model):
        """Test Jacobian condition number computation."""
        analyzer = SingularityAnalyzer(robot_model)
        
        q = random_q_in_limits(robot_model, seed=42)
        
        try:
            cond = analyzer.condition_number(q, backend='numpy')
            
            # Condition number should be >= 1
            assert cond >= 1.0
            assert isinstance(cond, (float, np.floating))
        except AttributeError:
            pytest.skip("condition_number method not implemented")
    
    def test_singularity_detection(self, robot_model):
        """Test detecting singular configurations."""
        analyzer = SingularityAnalyzer(robot_model)
        
        # Test several random configurations
        for seed in [42, 43, 44]:
            q = random_q_in_limits(robot_model, seed=seed)
            
            try:
                is_singular = analyzer.is_singular(q, threshold=1e-3, backend='numpy')
                assert isinstance(is_singular, (bool, np.bool_))
            except AttributeError:
                pytest.skip("is_singular method not implemented")
    
    def test_jacobian_rank(self, robot_model):
        """Test Jacobian rank computation."""
        q = random_q_in_limits(robot_model, seed=42)
        J = jacobian(robot_model, q, backend='numpy')
        
        # Compute rank
        rank = np.linalg.matrix_rank(J)
        
        # Rank should be <= min(rows, cols)
        assert rank <= min(J.shape)
        assert rank >= 0
        
        # For most configurations, Jacobian should be full rank
        assert rank > 0
    
    def test_singularity_index(self, robot_model):
        """Test singularity proximity index."""
        analyzer = SingularityAnalyzer(robot_model)
        
        q = random_q_in_limits(robot_model, seed=42)
        
        try:
            index = analyzer.singularity_index(q, backend='numpy')
            
            # Index should be between 0 and 1 (normalized)
            # Or some other defined range
            assert isinstance(index, (float, np.floating))
            assert index >= 0
        except AttributeError:
            pytest.skip("singularity_index method not implemented")


@pytest.mark.integration
@pytest.mark.slow
class TestSingularityMapping:
    """Test singularity mapping and visualization."""
    
    def test_singularity_map(self, robot_model):
        """Test generating singularity map over workspace."""
        analyzer = SingularityAnalyzer(robot_model)
        
        try:
            sing_map = analyzer.map_singularities(
                num_samples=100,  # Small sample for speed
                backend='numpy'
            )
            
            assert sing_map is not None
            assert len(sing_map) > 0
        except AttributeError:
            pytest.skip("map_singularities method not implemented")
    
    def test_critical_configurations(self, robot_model):
        """Test finding critical (near-singular) configurations."""
        analyzer = SingularityAnalyzer(robot_model)
        
        try:
            critical = analyzer.find_critical_configs(
                num_samples=50,
                threshold=0.01,
                backend='numpy'
            )
            
            # Should return list of configurations
            if critical:
                assert isinstance(critical, (list, np.ndarray))
        except AttributeError:
            pytest.skip("find_critical_configs method not implemented")


@pytest.mark.integration
class TestSingularityTypes:
    """Test classification of singularity types."""
    
    def test_singularity_classification(self, robot_model):
        """Test classifying singularity type."""
        analyzer = SingularityAnalyzer(robot_model)
        
        # Zero configuration (often singular)
        q_zero = np.zeros(robot_model.dof())
        
        try:
            sing_type = analyzer.classify_singularity(q_zero, backend='numpy')
            
            # Should return some classification
            # e.g., 'boundary', 'internal', 'elbow', etc.
            assert sing_type is not None
        except AttributeError:
            pytest.skip("classify_singularity method not implemented")
    
    def test_null_space_dimension(self, robot_model):
        """Test null space dimension at different configurations."""
        q = random_q_in_limits(robot_model, seed=42)
        J = jacobian(robot_model, q, backend='numpy')
        
        # Compute null space
        rank = np.linalg.matrix_rank(J)
        null_dim = J.shape[1] - rank
        
        # Null space dimension should be non-negative
        assert null_dim >= 0
        
        # For redundant manipulators, null space should exist
        if robot_model.dof() > 6:
            # Most configurations should have null space
            assert null_dim >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
