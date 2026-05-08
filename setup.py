#!/usr/bin/env python3
"""Setuptools entry: Eigen + pybind11 FK, Jacobian, and IK extensions (required if sources present)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parent

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    Pybind11Extension = None  # type: ignore[misc, assignment]
    build_ext = None  # type: ignore[misc, assignment]


def _eigen_include() -> str | None:
    env = os.environ.get("EIGEN3_INCLUDE_DIR", "").strip()
    if env:
        p = Path(env)
        if (p / "Eigen" / "Dense").is_file():
            return str(p)
    candidates = [
        Path("/opt/homebrew/include/eigen3"),
        Path("/usr/local/include/eigen3"),
        Path("/usr/include/eigen3"),
    ]
    for p in candidates:
        if (p / "Eigen" / "Dense").is_file():
            return str(p)
    return None


_NATIVE_CPPS = (
    "robocore/kinematics/fk_utils/_fk_chain_core.cpp",
    "robocore/kinematics/jacobian_utils/_jacobian_chain_core.cpp",
    "robocore/kinematics/ik_utils/_ik_chain_core.cpp",
    "robocore/kinematics/ik_utils/_multichain_ik_core.cpp",
)

_NATIVE_EXTENSION_SPECS: tuple[tuple[str, str], ...] = (
    ("robocore.kinematics.fk_utils._fk_chain_core", "robocore/kinematics/fk_utils/_fk_chain_core.cpp"),
    (
        "robocore.kinematics.jacobian_utils._jacobian_chain_core",
        "robocore/kinematics/jacobian_utils/_jacobian_chain_core.cpp",
    ),
    ("robocore.kinematics.ik_utils._ik_chain_core", "robocore/kinematics/ik_utils/_ik_chain_core.cpp"),
    (
        "robocore.kinematics.ik_utils._multichain_ik_core",
        "robocore/kinematics/ik_utils/_multichain_ik_core.cpp",
    ),
)


def _ext_modules():
    if not any((ROOT / p).is_file() for p in _NATIVE_CPPS):
        return []
    if Pybind11Extension is None:
        raise RuntimeError(
            "pybind11 is required to build native kinematics extensions. "
            "Use `pip install -e .` (PEP 517 installs pybind11 from pyproject.toml) "
            "or `pip install pybind11` before `python setup.py build_ext`."
        )
    ei = _eigen_include()
    if ei is None:
        raise RuntimeError(
            "Eigen3 headers not found (need Eigen/Dense). Install Eigen, e.g. "
            "`brew install eigen` (macOS), `apt install libeigen3-dev` (Debian/Ubuntu), "
            "or set EIGEN3_INCLUDE_DIR to the directory that contains the `Eigen/` folder."
        )
    extra = ["-O3", "-DEIGEN_NO_DEBUG"]
    if sys.platform == "darwin":
        extra.append("-stdlib=libc++")
    exts = []
    for mod_name, rel_path in _NATIVE_EXTENSION_SPECS:
        if (ROOT / rel_path).is_file():
            exts.append(
                Pybind11Extension(
                    mod_name,
                    [rel_path],
                    include_dirs=[ei],
                    cxx_std=17,
                    extra_compile_args=extra,
                )
            )
    return exts


# Force setuptools to build extensions when present (PEP 517 "wheel only" guard)
class _BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:  # noqa: D102
        return bool(_ext_modules())


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):  # noqa: D102
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for module in modules:
            module_file = Path(module[2])
            if module_file.name.endswith("_test.py") and "mjcf_parser" in module_file.parts:
                continue
            filtered_modules.append(module)
        return filtered_modules


ext_list = _ext_modules()
cmdclass = {"build_py": build_py}
if build_ext and ext_list:
    cmdclass["build_ext"] = build_ext

setup(
    distclass=_BinaryDistribution,
    ext_modules=ext_list,
    cmdclass=cmdclass,
    zip_safe=False,
)
