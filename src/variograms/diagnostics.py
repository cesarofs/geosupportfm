"""Spatial structure diagnostics for latent embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .latent import latent_variography_report


@dataclass(frozen=True)
class SpatialStructureDiagnostics:
    """Aggregate diagnostics for a latent embedding set."""

    n_dimensions: int
    structured_fraction: float
    mean_structure_score: float
    median_nugget_ratio: float
    mean_practical_range: float
    top_dimension: str


def spatial_structure_diagnostics(
    coords: np.ndarray,
    embeddings: np.ndarray,
    band_names: Optional[Iterable[str]] = None,
    n_lags: int = 10,
    max_distance: Optional[float] = None,
    pair_min: int = 5,
    structure_threshold: float = 0.35,
) -> tuple[SpatialStructureDiagnostics, pd.DataFrame]:
    """Return aggregate diagnostics plus the per-dimension report."""
    report = latent_variography_report(
        coords=coords,
        embeddings=embeddings,
        band_names=band_names,
        n_lags=n_lags,
        max_distance=max_distance,
        pair_min=pair_min,
    )
    if report.empty:
        raise ValueError("report cannot be empty")

    structured_fraction = float(np.mean(report["structure_score"] >= structure_threshold))
    diagnostics = SpatialStructureDiagnostics(
        n_dimensions=int(len(report)),
        structured_fraction=structured_fraction,
        mean_structure_score=float(report["structure_score"].mean()),
        median_nugget_ratio=float(report["nugget_ratio"].median()),
        mean_practical_range=float(report["practical_range"].mean()),
        top_dimension=str(report.iloc[0]["name"]),
    )
    return diagnostics, report
