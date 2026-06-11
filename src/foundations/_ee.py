"""Lazy Earth Engine import helpers."""

from __future__ import annotations

from typing import Any


def require_ee() -> Any:
    """Import and return the Earth Engine module.

    Raises
    ------
    ImportError
        If the Earth Engine Python API is not installed.
    """
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "earthengine-api is required for GeoSupportFM foundation loaders. "
            "Install it with: pip install earthengine-api"
        ) from exc
    return ee
