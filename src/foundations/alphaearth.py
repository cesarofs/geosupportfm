"""AlphaEarth / Satellite Embedding loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from geosupportfm.foundations._ee import require_ee

ALPHAEARTH_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_ALPHAEARTH_BAND_COUNT = 64
DEFAULT_ALPHAEARTH_SCALE = 10


def validate_year(year: int) -> int:
    if not isinstance(year, int):
        raise TypeError("year must be an integer")
    if year < 1984:
        raise ValueError("year must be >= 1984")
    return year


def year_bounds(year: int) -> tuple[str, str]:
    """Return ISO date strings for the annual Earth Engine window."""
    validate_year(year)
    return f"{year}-01-01", f"{year + 1}-01-01"


def default_alphaearth_bands(count: int = DEFAULT_ALPHAEARTH_BAND_COUNT) -> list[str]:
    """Return the canonical AlphaEarth band names A01..A64."""
    if count <= 0:
        raise ValueError("count must be positive")
    return [f"A{i:02d}" for i in range(1, count + 1)]


def load_alphaearth_embeddings(
    year: int,
    region=None,
    bands: Sequence[str] | None = None,
):
    """Load the annual AlphaEarth embedding image for a given year.

    Parameters
    ----------
    year:
        Annual composite year.
    region:
        Optional Earth Engine geometry used to filter and clip.
    bands:
        Optional subset of embedding band names.
    """
    ee = require_ee()
    start, end = year_bounds(year)
    band_list = list(bands) if bands is not None else default_alphaearth_bands()

    image = (
        ee.ImageCollection(ALPHAEARTH_COLLECTION_ID)
        .filterDate(start, end)
    )
    if region is not None:
        image = image.filterBounds(region)

    image = ee.Image(image.mosaic()).select(band_list)
    if region is not None:
        image = image.clip(region)
    return image


def sample_alphaearth_embeddings(
    points,
    year: int,
    region=None,
    scale: int = DEFAULT_ALPHAEARTH_SCALE,
    bands: Sequence[str] | None = None,
):
    """Sample AlphaEarth embeddings at a FeatureCollection of points."""
    ee = require_ee()
    image = load_alphaearth_embeddings(year=year, region=region, bands=bands)
    return image.sampleRegions(
        collection=points,
        scale=scale,
        geometries=True,
        tileScale=4,
    )
