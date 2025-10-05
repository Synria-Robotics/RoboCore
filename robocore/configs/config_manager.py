"""Configuration manager for RoboCore.

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

from pathlib import Path
from typing import Optional, Union, Dict, Any
from omegaconf import OmegaConf, DictConfig
from .schemas import RoboCoreConfig


class ConfigManager:
    """Manager for RoboCore configurations."""
    
    def __init__(self, config: Union[str, Path, Dict, DictConfig, RoboCoreConfig, None] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config : str, Path, Dict, DictConfig, RoboCoreConfig, or None
            Configuration source. Can be:
            - Path to YAML file
            - Dictionary
            - OmegaConf DictConfig
            - RoboCoreConfig dataclass instance
            - None (use default config)
        """
        if config is None:
            # Use default configuration
            self.cfg = OmegaConf.structured(RoboCoreConfig)
        elif isinstance(config, (str, Path)):
            # Load from YAML file
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            file_cfg = OmegaConf.load(config_path)
            # Merge with default structured config for type safety
            default_cfg = OmegaConf.structured(RoboCoreConfig)
            self.cfg = OmegaConf.merge(default_cfg, file_cfg)
        elif isinstance(config, dict):
            # Create from dictionary
            default_cfg = OmegaConf.structured(RoboCoreConfig)
            self.cfg = OmegaConf.merge(default_cfg, config)
        elif isinstance(config, DictConfig):
            # Use OmegaConf DictConfig directly
            self.cfg = config
        elif isinstance(config, RoboCoreConfig):
            # Convert dataclass to OmegaConf
            self.cfg = OmegaConf.structured(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Parameters
        ----------
        path : str or Path
            Path to save config file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, path)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Parameters
        ----------
        updates : Dict
            Dictionary of updates (supports nested keys with dot notation)
        """
        self.cfg = OmegaConf.merge(self.cfg, updates)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation, e.g., 'robot.urdf_path')
        default : Any
            Default value if key not found
        
        Returns
        -------
        Any
            Configuration value
        """
        return OmegaConf.select(self.cfg, key, default=default)
    
    def to_dict(self) -> Dict:
        """Convert configuration to plain dictionary."""
        return OmegaConf.to_container(self.cfg, resolve=True)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return OmegaConf.to_yaml(self.cfg)
    
    def __repr__(self) -> str:
        return f"ConfigManager(\n{OmegaConf.to_yaml(self.cfg)})"


def load_config(path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    path : str or Path
        Path to config file
    
    Returns
    -------
    ConfigManager
        Configuration manager instance
    """
    return ConfigManager(path)


def get_default_config() -> ConfigManager:
    """
    Get default RoboCore configuration.
    
    Returns
    -------
    ConfigManager
        Configuration manager with default values
    """
    return ConfigManager()


def create_config_from_dict(config_dict: Dict) -> ConfigManager:
    """
    Create configuration from dictionary.
    
    Parameters
    ----------
    config_dict : Dict
        Configuration dictionary
    
    Returns
    -------
    ConfigManager
        Configuration manager instance
    """
    return ConfigManager(config_dict)
