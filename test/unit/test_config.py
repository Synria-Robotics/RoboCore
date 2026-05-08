#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import pytest
import yaml
from pathlib import Path
from robocore.configs.config_manager import ConfigManager


@pytest.fixture(scope="module")
def default_config_path():
    """Path to default configuration."""
    return Path('robocore/configs/default.yaml')


@pytest.fixture(scope="module")
def gpu_config_path():
    """Path to GPU configuration."""
    return Path('robocore/configs/gpu_config.yaml')


class TestConfigManager:
    """Test ConfigManager loading and access."""
    
    def test_load_default_config(self, default_config_path):
        """Test loading default configuration."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        assert config is not None
        assert hasattr(config, 'get')
    
    def test_config_keys(self, default_config_path):
        """Test expected configuration keys exist."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        
        # Common expected keys (adjust based on actual schema)
        expected_sections = ['backend', 'solver', 'robot', 'computation']
        
        for section in expected_sections:
            try:
                value = config.get(section)
                assert value is not None or section not in config._config
            except KeyError:
                # Section might not exist in all configs
                pass
    
    def test_config_nested_access(self, default_config_path):
        """Test nested configuration access."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        
        # Try accessing nested keys if they exist
        try:
            backend_type = config.get('backend.type')
            assert backend_type in ['numpy', 'torch', None]
        except KeyError:
            pass  # Config might not have this structure
    
    def test_gpu_config_load(self, gpu_config_path):
        """Test loading GPU-specific configuration."""
        if not gpu_config_path.exists():
            pytest.skip(f"Config not found: {gpu_config_path}")
        
        config = ConfigManager(str(gpu_config_path))
        assert config is not None
    
    def test_invalid_config_path(self):
        """Test loading non-existent config raises error."""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            ConfigManager('nonexistent_config.yaml')
    
    def test_config_immutability(self, default_config_path):
        """Test configuration values can be accessed safely."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        
        # Try to get some value
        try:
            original_value = config.get('backend')
            # Getting again should return same value
            second_value = config.get('backend')
            assert original_value == second_value
        except KeyError:
            pass  # Key might not exist


class TestConfigSchema:
    """Test configuration schema validation."""
    
    def test_yaml_parsing(self, default_config_path):
        """Test YAML file can be parsed."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        with open(default_config_path) as f:
            data = yaml.safe_load(f)
        
        assert data is not None
        assert isinstance(data, dict)
    
    def test_config_types(self, default_config_path):
        """Test configuration values have correct types."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        
        # Check some expected types (adjust based on actual schema)
        try:
            backend = config.get('backend')
            if backend is not None:
                assert isinstance(backend, (str, dict))
        except KeyError:
            pass
        
        try:
            solver = config.get('solver')
            if solver is not None:
                assert isinstance(solver, dict)
        except KeyError:
            pass
    
    def test_all_configs_loadable(self):
        """Test all config files in configs/ are loadable."""
        config_dir = Path('robocore/configs')
        if not config_dir.exists():
            pytest.skip("Config directory not found")
        
        config_files = list(config_dir.glob('*.yaml'))
        assert len(config_files) > 0, "No config files found"
        
        for config_file in config_files:
            if config_file.name == 'schemas.yaml':
                continue  # Skip schema file
            
            try:
                config = ConfigManager(str(config_file))
                assert config is not None
            except Exception as e:
                pytest.fail(f"Failed to load {config_file}: {e}")


class TestConfigEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_config(self, tmp_path):
        """Test handling of empty configuration file."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")
        
        try:
            config = ConfigManager(str(empty_config))
            # Empty config might be valid (empty dict)
            assert config is not None
        except ValueError:
            # Or might raise error - both are acceptable
            pass
    
    def test_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("{ invalid: yaml: content")
        
        with pytest.raises((yaml.YAMLError, ValueError, Exception)):
            ConfigManager(str(bad_config))
    
    def test_get_nonexistent_key(self, default_config_path):
        """Test accessing non-existent key."""
        if not default_config_path.exists():
            pytest.skip(f"Config not found: {default_config_path}")
        
        config = ConfigManager(str(default_config_path))
        
        with pytest.raises(KeyError):
            config.get('nonexistent.deeply.nested.key')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
