"""Raster aggregation tools implemented with NumPy."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def _validate_factor(factor: int) -> int:
    if not isinstance(factor, int):
        raise TypeError("factor must be an integer")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return factor


def _reshape_for_blocks(array: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError("array must be 2D")
    rows = (arr.shape[0] // factor) * factor
    cols = (arr.shape[1] // factor) * factor
    if rows == 0 or cols == 0:
        raise ValueError("factor is larger than the array dimensions")
    return arr[:rows, :cols]


def aggregate_array(array: np.ndarray, factor: int, reducer: str = "mean") -> np.ndarray:
    """Aggregate a 2D raster array by an integer factor."""
    factor = _validate_factor(factor)
    arr = _reshape_for_blocks(array, factor)

    new_rows = arr.shape[0] // factor
    new_cols = arr.shape[1] // factor
    blocks = arr.reshape(new_rows, factor, new_cols, factor)

    if reducer == "mean":
        return np.nanmean(blocks, axis=(1, 3))
    if reducer == "sum":
        return np.nansum(blocks, axis=(1, 3))
    if reducer == "median":
        return np.nanmedian(blocks, axis=(1, 3))
    if reducer == "min":
        return np.nanmin(blocks, axis=(1, 3))
    if reducer == "max":
        return np.nanmax(blocks, axis=(1, 3))
    raise ValueError("Unsupported reducer")


def aggregate_raster_stack(
    stack: Sequence[np.ndarray],
    factor: int,
    reducer: str = "mean",
) -> list[np.ndarray]:
    """Aggregate a sequence of raster bands using the same factor."""
    return [aggregate_array(band, factor=factor, reducer=reducer) for band in stack]
