"""Configuration objects and defaults for embedding workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


def default_embedding_bands(n_bands: int = 64) -> List[str]:
    """Return canonical AlphaEarth embedding band names."""
    return [f"A{i:02d}" for i in range(n_bands)]


@dataclass(frozen=True)
class GeoSupportEmbeddingConfig:
    """Runtime settings for LULC sampling and embedding evaluation."""

    project_id: str
    years: Sequence[int] = field(default_factory=lambda: tuple(range(2017, 2025)))
    train_years: Sequence[int] = field(default_factory=lambda: tuple(range(2017, 2023)))
    test_years: Sequence[int] = field(default_factory=lambda: (2023, 2024))
    per_class: int = 75
    seed: int = 42
    sample_scale: int = 30
    lulc_asset: str = "projects/mapbiomas-public/assets/brazil/lulc/v1"
    emb_collection: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    emb_bands: Sequence[str] = field(default_factory=default_embedding_bands)
    lulc_column: str = "lulc"
    year_column: str = "year"

    def with_years(self, years: Sequence[int]) -> "GeoSupportEmbeddingConfig":
        """Return a copy with a new year sequence."""
        return GeoSupportEmbeddingConfig(
            project_id=self.project_id,
            years=years,
            train_years=self.train_years,
            test_years=self.test_years,
            per_class=self.per_class,
            seed=self.seed,
            sample_scale=self.sample_scale,
            lulc_asset=self.lulc_asset,
            emb_collection=self.emb_collection,
            emb_bands=self.emb_bands,
            lulc_column=self.lulc_column,
            year_column=self.year_column,
        )
