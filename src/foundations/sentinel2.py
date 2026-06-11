"""Sentinel-2 surface reflectance loaders."""

from __future__ import annotations

from typing import Sequence

from geosupportfm.foundations._ee import require_ee

SENTINEL2_SR_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SENTINEL2_SCALE = 10


def default_sentinel2_bands() -> list[str]:
    return list(DEFAULT_SENTINEL2_BANDS)


def mask_sentinel2_clouds(image):
    """Mask S2 cloud / cirrus pixels using QA60 when available."""
    ee = require_ee()
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def load_sentinel2_sr(
    start_date: str,
    end_date: str,
    region=None,
    cloud_cover: int = 20,
    bands: Sequence[str] | None = None,
    composite: bool = True,
):
    """Load Sentinel-2 SR harmonized imagery for a date window."""
    ee = require_ee()
    band_list = list(bands) if bands is not None else default_sentinel2_bands()

    collection = (
        ee.ImageCollection(SENTINEL2_SR_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover))
    )
    if region is not None:
        collection = collection.filterBounds(region)

    collection = collection.map(mask_sentinel2_clouds)
    image = ee.Image(collection.median() if composite else collection.mosaic()).select(band_list)
    if region is not None:
        image = image.clip(region)
    return image
