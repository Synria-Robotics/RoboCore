import os
from pathlib import Path
import robocore

def get_robocore_path(relative_path = ""):
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