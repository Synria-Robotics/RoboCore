from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "RoboCore"

init_text = (ROOT / "robocore" / "__init__.py").read_text(encoding="utf-8")
match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text)
version = match.group(1) if match else "0.0.0"

# Read requirements from requirements.txt
def parse_requirements(requirements_file):
    """Parse requirements.txt file and return list of dependencies."""
    requirements_path = ROOT / requirements_file
    if not requirements_path.exists():
        return []
    
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
    return requirements

install_requires = parse_requirements("requirements.txt")

setup(
    name="robocore",
    version=version,
    description="Unified High-Throughput Robotics Library",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Synria Robotics Team",  # 需要填写
    author_email="support@synriarobotics.ai",  # 需要添加
    url="https://github.com/Synria-Robotics/RoboCore",  # 需要添加
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    package_data={
        "robocore": [
            "modeling/parser/mjcf_parser/*.xml",
            "configs/*.yaml",
        ],
    },
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "ruff", "pytest", "mypy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # 需要修正为 GPL-3.0
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence & Robotics",
    ],
)
