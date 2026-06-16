"""Core variogram and cross-variogram computations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariogramSummary:
    """Compact summary of an empirical variogram."""

    nugget: float
    sill: float
    practical_range: float
    nugget_ratio: float
    structure_score: float
    monotonicity: float


def _as_2d_array(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("coords must be a 2D array with at least two columns")
    return arr[:, :2]


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return arr


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _triangular_pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def _bin_edges(distances: np.ndarray, n_lags: int, max_distance: Optional[float]) -> np.ndarray:
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    finite = distances[np.isfinite(distances)]
    upper = float(np.max(finite)) if max_distance is None else float(max_distance)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("max_distance must be positive")
    return np.linspace(0.0, upper, n_lags + 1)


def _build_variogram_frame(distances: np.ndarray, semivariances: np.ndarray, edges: np.ndarray, pair_min: int) -> pd.DataFrame:
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for i, center in enumerate(centers):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(centers) - 1:
            mask = (distances >= lo) & (distances <= hi)
        else:
            mask = (distances >= lo) & (distances < hi)
        n_pairs = int(np.sum(mask))
        if n_pairs < pair_min:
            rows.append((center, np.nan, n_pairs))
        else:
            rows.append((center, float(np.mean(semivariances[mask])), n_pairs))
    return pd.DataFrame(rows, columns=["lag_center", "semivariance", "npairs"])


def empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 10,
    max_distance: Optional[float] = None,
    pair_min: int = 5,
) -> pd.DataFrame:
    """Compute an empirical semivariogram by distance bin.

    Returns a DataFrame with columns: lag_center, semivariance, npairs.
    """
    coords_arr = _as_2d_array(coords)
    values_arr = _as_1d_array(values)
    if len(coords_arr) != len(values_arr):
        raise ValueError("coords and values must have the same number of rows")

    i, j = _triangular_pairs(len(values_arr))
    pair_distances = _pairwise_distances(coords_arr)[i, j]
    pair_semivariance = 0.5 * (values_arr[i] - values_arr[j]) ** 2
    edges = _bin_edges(pair_distances, n_lags=n_lags, max_distance=max_distance)
    return _build_variogram_frame(pair_distances, pair_semivariance, edges, pair_min=pair_min)


def empirical_cross_variogram(
    coords: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_lags: int = 10,
    max_distance: Optional[float] = None,
    pair_min: int = 5,
) -> pd.DataFrame:
    """Compute an empirical cross-variogram by distance bin."""
    coords_arr = _as_2d_array(coords)
    a = _as_1d_array(values_a)
    b = _as_1d_array(values_b)
    if not (len(coords_arr) == len(a) == len(b)):
        raise ValueError("coords, values_a, and values_b must have the same length")

    i, j = _triangular_pairs(len(a))
    pair_distances = _pairwise_distances(coords_arr)[i, j]
    pair_cross = 0.5 * (a[i] - a[j]) * (b[i] - b[j])
    edges = _bin_edges(pair_distances, n_lags=n_lags, max_distance=max_distance)
    return _build_variogram_frame(pair_distances, pair_cross, edges, pair_min=pair_min)


def variogram_summary(frame: pd.DataFrame) -> VariogramSummary:
    """Summarize an empirical variogram frame into a few robust diagnostics."""
    if frame.empty:
        raise ValueError("frame cannot be empty")
    if "semivariance" not in frame.columns:
        raise ValueError("frame must contain a semivariance column")

    clean = frame.dropna(subset=["semivariance"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("frame contains no valid semivariance values")

    semivariance = clean["semivariance"].to_numpy(dtype=float)
    lag = clean["lag_center"].to_numpy(dtype=float)

    nugget = float(semivariance[0])
    sill = float(np.nanmax(semivariance))
    if sill <= 0:
        sill = float(np.nanmean(semivariance) + 1e-12)

    target = 0.95 * sill
    practical_range = float(lag[-1])
    for l, g in zip(lag, semivariance):
        if g >= target:
            practical_range = float(l)
            break

    nugget_ratio = float(np.clip(nugget / sill, 0.0, 1.0))
    structure_score = float(np.clip((1.0 - nugget_ratio) * (practical_range / max(lag[-1], 1e-12)), 0.0, 1.0))

    diffs = np.diff(semivariance)
    if len(diffs) == 0:
        monotonicity = 1.0
    else:
        monotonicity = float(np.mean(diffs >= -1e-12))

    return VariogramSummary(
        nugget=nugget,
        sill=sill,
        practical_range=practical_range,
        nugget_ratio=nugget_ratio,
        structure_score=structure_score,
        monotonicity=monotonicity,
    )
