"""Beauty logger original implementation (restored).

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

import os
import sys
from typing import Any, Optional


class BeautyLogger:
    """
    Lightweight logger for RoboCore package.
    """

    def __init__(self, log_dir: str, log_name: str = 'RoboCore.log', verbose: bool = True):
        """
        Lightweight logger for RoboCore package.

        Example::

            >>> from RoboCore.utils.logger import BeautyLogger
            >>> logger = BeautyLogger(log_dir=".", log_name="RoboCore.log", verbose=True)

        :param log_dir: the path for saving the log file
        :param log_name: the name of the log file
        :param verbose: whether to print the log to the console
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.verbose = verbose

    def _write_log(self, content, type):
        with open(self.log_path, "a") as f:
            f.write("[RoboCore:{}] {}\n".format(type.upper(), content))

    def warning(self, content, local_verbose=True):
        """
        Print the warning message.

        Example::

            >>> logger.warning("This is a warning message.")

        :param content: the content of the warning message
        :param local_verbose: whether to print the warning message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="warning")
        self._write_log(content, type="warning")

    def module(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.module("This is a module message.")

        :param content: the content of the module message
        :param local_verbose: whether to print the module message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.info("This is a info message.")

        :param content: the content of the info message
        :param local_verbose: whether to print the info message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")


def beauty_print(content, type: str = None, width: int = 80, centered: bool = True):
    """
    Print the content with different colors.

    Example::

        >>> import RoboCore as rc
        >>> rc.logger.beauty_print("This is a warning message.", type="warning")
        >>> rc.logger.beauty_print("Section Title", type="module", centered=True)

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error", "success", "separator"
    :param width: width of the separator line (default: 80)
    :param centered: for "module" type, whether to center the title in separator (default: False)
    :return:
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m[RoboCore:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
        if centered:
            # Centered title with separator
            print("\n" + "=" * width)
            print("\033[1;33m{}\033[0m".format(content.center(width)))
            print("=" * width)
        else:
            # Original module format
            print("\033[1;33m[RoboCore:MODULE] {}\033[0m".format(content))  # For a new module (light yellow)
    elif type == "info":
        print("\033[1;35m[RoboCore:INFO] {}\033[0m".format(content))  # For info (light purple)
    elif type == "error":
        print("\033[1;31m[RoboCore:ERROR] {}\033[0m".format(content))  # For error (red)
        raise ValueError(content)
    elif type == "success":
        print("\033[1;32m[RoboCore:SUCCESS] {}\033[0m".format(content))  # For success (green)
    else:
        raise ValueError("Invalid level")


def beauty_print_matrix(name: str, data: Any, precision: int = 4, max_batch_items: int = 1, indent: int = 2):
    """
    Pretty print a scalar / vector / matrix / batch of matrices with RoboCore style.

    Automatically handles:
    - torch tensors (moved to cpu and converted to numpy)
    - numpy arrays / Python lists / scalars
    - Batch data (N, m, n) where m,n <= 6 treated as matrices batch

    :param name: label of the value
    :param data: value (scalar / 1D / 2D / 3D)
    :param precision: number of decimal places
    :param max_batch_items: number of batch entries to preview
    :param indent: left indentation (spaces)
    """
    # Lazy imports to avoid hard dependency if user does not need them
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None  # type: ignore
    try:
        import torch as _torch  # type: ignore
    except Exception:  # pragma: no cover
        _torch = None  # type: ignore

    # Normalize input
    arr = data
    if _torch is not None and isinstance(arr, _torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif _np is not None and not isinstance(arr, (int, float)) and not isinstance(arr, str):
        if not isinstance(arr, _np.ndarray):
            try:
                arr = _np.array(arr)
            except Exception:
                pass

    # Simple scalar
    if isinstance(arr, (int, float)) or (hasattr(arr, "ndim") and getattr(arr, "ndim") == 0):
        print(" " * indent + f"{name} = {float(arr):.{precision}f}")

    # If still something unexpected, just print raw
    if not hasattr(arr, "ndim"):
        print(" " * indent + f"{name} = {arr}")

    ndim = arr.ndim  # type: ignore
    fmt = f"{{:>{precision + 6}.{precision}f}}"
    pad = " " * indent

    if ndim == 1:
        # Vector
        try:
            line = "  ".join(fmt.format(float(v)) for v in arr)
            print(pad + f"{name} = [{line}]")
        except Exception:
            print(pad + f"{name} = {arr}")
    elif ndim == 2:
        # Single matrix
        print(pad + f"{name} =")
        for row in arr:
            try:
                row_str = "  ".join(fmt.format(float(v)) for v in row)
            except Exception:
                row_str = "  ".join(str(v) for v in row)
            print(pad + "  [" + row_str + "]")
    elif ndim == 3:
        n = arr.shape[0]
        print(pad + f"{name} (batch size={n})")
        preview = min(max_batch_items, n)
        for bi in range(preview):
            if preview > 1:
                print(pad + f"  [item {bi}]")
            for row in arr[bi]:
                try:
                    row_str = "  ".join(fmt.format(float(v)) for v in row)
                except Exception:
                    row_str = "  ".join(str(v) for v in row)
                print(pad + "    [" + row_str + "]")
        if preview < n:
            print(pad + f"  ... ({n - preview} more)")
    else:
        # Higher dimension – fallback summary
        print(pad + f"{name} shape={getattr(arr, 'shape', '?')}")


def beauty_print_array(arr: Any, precision: int = 5, sign: bool = True) -> str:
    """Return a formatted string for 1D / 2D numeric arrays (numpy / torch / list).

    Behavior:
    - Scalars -> formatted with specified precision
    - 1D -> [ +0.12345, -0.12345, ... ]
    - 2D -> multi-line matrix style
    - Other shapes -> falls back to str(arr)

    :param arr: input data
    :param precision: decimal places
    :param sign: whether to always show sign
    :return: string
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None  # type: ignore
    try:
        import torch as _torch  # type: ignore
    except Exception:  # pragma: no cover
        _torch = None  # type: ignore

    # Normalize to numpy array when possible
    if _torch is not None and isinstance(arr, _torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif _np is not None:
        if not isinstance(arr, (int, float)) and not isinstance(arr, str):
            if not isinstance(arr, _np.ndarray):
                try:
                    arr = _np.array(arr)
                except Exception:
                    pass

    # Scalars
    if isinstance(arr, (int, float)):
        fmt = f"%{'+' if sign else ''}.{precision}f"
        return fmt % float(arr)

    if _np is None or not hasattr(arr, 'ndim'):
        return str(arr)

    if arr.ndim == 0:
        fmt = f"%{'+' if sign else ''}.{precision}f"
        return fmt % float(arr)

    number_fmt = f"{{:{'+' if sign else ''}.{precision}f}}"

    if arr.ndim == 1:
        values = ', '.join(number_fmt.format(float(x)) for x in arr)
        return f"[{values}]"
    elif arr.ndim == 2:
        lines = []
        for row in arr:
            row_str = '  '.join(number_fmt.format(float(x)) for x in row)
            lines.append(f"  [{row_str}]")
        return "[\n" + "\n".join(lines) + "\n]"
    else:
        return str(arr)


__all__ = ["BeautyLogger", "beauty_print", "beauty_print_matrix", "beauty_print_array"]
