"""Phase 1 foundations: Earth observation data access helpers."""

from geosupportfm.foundations.alphaearth import (
    ALPHAEARTH_COLLECTION_ID,
    default_alphaearth_bands,
    load_alphaearth_embeddings,
    sample_alphaearth_embeddings,
    year_bounds,
)
from geosupportfm.foundations.landsat import (
    LANDSAT_COLLECTION_IDS,
    common_landsat_bands,
    load_landsat_sr,
)
from geosupportfm.foundations.sentinel2 import (
    SENTINEL2_SR_COLLECTION_ID,
    default_sentinel2_bands,
    load_sentinel2_sr,
)

__all__ = [
    "ALPHAEARTH_COLLECTION_ID",
    "LANDSAT_COLLECTION_IDS",
    "SENTINEL2_SR_COLLECTION_ID",
    "default_alphaearth_bands",
    "default_sentinel2_bands",
    "load_alphaearth_embeddings",
    "sample_alphaearth_embeddings",
    "load_landsat_sr",
    "load_sentinel2_sr",
    "year_bounds",
    "common_landsat_bands",
]
