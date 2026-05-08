"""Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import pathlib

import robocore


def get_robocore_path(relative_path=""):
    """
    Get the installed robocore package path.

    :return: Absolute path to the installed robocore package directory
    """
    path = os.path.join(Path(robocore.__file__).parent.absolute(), relative_path)
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


def get_robocore_root():
    """
    Get the root directory of the RoboCore project from installed package.

    :return: Absolute path to the RoboCore project root directory
    """
    robocore_pkg_path = Path(get_robocore_path())
    return str(robocore_pkg_path.parent.absolute())


def create_dir(path: str | Path) -> None:
    """Create directory if it does not exist.

    :param path: Directory path to create
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def list_absl_path(dir_path: str | Path, recursive: bool = False,
                   prefix: Optional[str] = None, suffix: Optional[str] = None) -> List[str]:
    """Get absolute paths of files in directory.

    :param dir_path: Directory path
    :param recursive: If True, search subdirectories
    :param prefix: Filter by filename prefix
    :param suffix: Filter by filename suffix
    :return: List of absolute file paths
    """
    if recursive:
        return [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dir_path)
            for file in files
            if (suffix is None or file.endswith(suffix)) and
            (prefix is None or file.startswith(prefix))
        ]
    else:
        return [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if (suffix is None or file.endswith(suffix)) and
            (prefix is None or file.startswith(prefix))
        ]


def get_resource(path: str | Path, mode: str = 'rb') -> bytes:
    """Load resource file.

    :param path: File path
    :param mode: File open mode
    :return: File contents
    """
    with open(path, mode=mode) as f:
        return f.read()
