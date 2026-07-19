"""GeoSupportFM public package namespace.

The research repository keeps domain modules directly below ``src`` and its
benchmark helpers at the repository root.  Extending ``__path__`` preserves
that layout while exposing the documented ``geosupportfm.*`` imports.
"""
from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
__path__ = [str(_SRC), str(_ROOT)]

__version__ = "0.4.0"

