"""Simulation bridge implementations."""

from importlib import import_module

_LAZY_SUBMODULES = {
    'mujoco',
}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['mujoco']
