"""Beauty logger original implementation (restored)."""

import os
import sys


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


def beauty_print(content, type: str = None):
    """
    Print the content with different colors.

    Example::

        >>> import RoboCore as rc
        >>> rc.logger.beauty_print("This is a warning message.", type="warning")

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error"
    :return:
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m[RoboCore:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
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

__all__ = ["BeautyLogger", "beauty_print"]
