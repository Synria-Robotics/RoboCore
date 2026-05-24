"""Analysis tools for robotics."""

from importlib import import_module

_LAZY_EXPORTS = {
    "SingularityAnalyzer": ("robocore.analysis.singularity_analyzer", "SingularityAnalyzer"),
    "WorkspaceAnalyzer": ("robocore.analysis.workspace_analyzer", "WorkspaceAnalyzer"),
    "analyze_workspace_comparison": (
        "robocore.analysis.workspace_analyzer",
        "analyze_workspace_comparison",
    ),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
