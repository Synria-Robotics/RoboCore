from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "RoboCore"

init_text = (ROOT / "robocore" / "__init__.py").read_text(encoding="utf-8")
match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text)
version = match.group(1) if match else "0.0.0"

setup(
    name="synria-robocore",
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
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "torch": ["torch"],
        "mujoco": ["mujoco"],
        "dev": ["black", "ruff", "pytest", "mypy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # 需要修正为 GPL-3.0
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
