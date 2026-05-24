"""Helpers for resolving robot-description package URIs."""

from __future__ import annotations

from pathlib import Path
from typing import Union


def resolve_description_path(path: Union[str, Path]) -> str:
    """Resolve robot-description URIs to local filesystem paths.

    Supported URI forms:
    - ``synriard://<name>/<version>/<variant>/<format>``
    - ``openrd://<name>/<version>/<variant>/<format>``

    Plain filesystem paths are returned unchanged.
    """
    path_str = str(path)
    if "://" not in path_str:
        return path_str

    provider, rest = path_str.split("://", 1)
    parts = [p for p in rest.split("/") if p]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid robot-description URI {path_str!r}; expected "
            f"{provider}://<name>/<version>/<variant>/<format>"
        )

    name, version, variant, model_format = parts
    if provider == "synriard":
        module_name = "synriard"
        install_hint = "pip install synria-robocore[descriptions]"
    elif provider == "openrd":
        module_name = "openrd"
        install_hint = "install the OpenRD robot-description package"
    else:
        raise ValueError(f"Unsupported robot-description provider: {provider!r}")

    try:
        module = __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Robot description package {module_name!r} is required to resolve "
            f"{path_str!r}. Install it with `{install_hint}`."
        ) from exc

    return module.get_model_path(
        name,
        version=version,
        variant=variant,
        model_format=model_format,
    )


__all__ = ["resolve_description_path"]
