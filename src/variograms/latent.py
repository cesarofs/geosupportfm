"""Latent-dimension variography utilities."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .core import empirical_variogram, variogram_summary


def _validate_latent_inputs(coords: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_arr = np.asarray(coords, dtype=float)
    emb_arr = np.asarray(embeddings, dtype=float)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
        raise ValueError("coords must be a 2D array with at least two columns")
    if emb_arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(coords_arr) != len(emb_arr):
        raise ValueError("coords and embeddings must have the same number of rows")
    return coords_arr[:, :2], emb_arr


def latent_variography_report(
    coords: np.ndarray,
    embeddings: np.ndarray,
    band_names: Optional[Iterable[str]] = None,
    n_lags: int = 10,
    max_distance: Optional[float] = None,
    pair_min: int = 5,
) -> pd.DataFrame:
    """Compute variogram summaries for each latent dimension.

    The output is ranked by a simple spatial-structure score.
    """
    coords_arr, emb_arr = _validate_latent_inputs(coords, embeddings)
    n_dims = emb_arr.shape[1]
    if band_names is None:
        names = [f"dim_{i:02d}" for i in range(n_dims)]
    else:
        names = list(band_names)
        if len(names) != n_dims:
            raise ValueError("band_names must match the number of embedding dimensions")

    rows = []
    for idx, name in enumerate(names):
        frame = empirical_variogram(
            coords_arr,
            emb_arr[:, idx],
            n_lags=n_lags,
            max_distance=max_distance,
            pair_min=pair_min,
        )
        summary = variogram_summary(frame)
        rows.append(
            {
                "dimension": idx,
                "name": name,
                "nugget": summary.nugget,
                "sill": summary.sill,
                "practical_range": summary.practical_range,
                "nugget_ratio": summary.nugget_ratio,
                "structure_score": summary.structure_score,
                "monotonicity": summary.monotonicity,
            }
        )

    report = pd.DataFrame(rows)
    return report.sort_values(["structure_score", "practical_range"], ascending=False).reset_index(drop=True)
