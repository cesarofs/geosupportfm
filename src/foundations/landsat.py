"""Landsat Collection 2 Level-2 loaders."""

from __future__ import annotations

from typing import Mapping, Sequence

from geosupportfm.foundations._ee import require_ee

LANDSAT_COLLECTION_IDS = {
    "landsat5": "LANDSAT/LT05/C02/T1_L2",
    "landsat7": "LANDSAT/LE07/C02/T1_L2",
    "landsat8": "LANDSAT/LC08/C02/T1_L2",
    "landsat9": "LANDSAT/LC09/C02/T1_L2",
}

COMMON_LANDSAT_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

_BAND_MAPS = {
    "landsat5": {
        "SR_B1": "blue",
        "SR_B2": "green",
        "SR_B3": "red",
        "SR_B4": "nir",
        "SR_B5": "swir1",
        "SR_B7": "swir2",
    },
    "landsat7": {
        "SR_B1": "blue",
        "SR_B2": "green",
        "SR_B3": "red",
        "SR_B4": "nir",
        "SR_B5": "swir1",
        "SR_B7": "swir2",
    },
    "landsat8": {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
    },
    "landsat9": {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
    },
}


def common_landsat_bands() -> list[str]:
    return list(COMMON_LANDSAT_BANDS)


def _scale_optical(image):
    return image.multiply(0.0000275).add(-0.2)


def _scale_thermal(image):
    return image.multiply(0.00341802).add(149.0)


def mask_landsat_clouds(image):
    """Mask clouds, cloud shadows, snow, and fill pixels."""
    qa = image.select("QA_PIXEL")
    fill = 1 << 0
    dilated = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    snow = 1 << 5
    mask = (
        qa.bitwiseAnd(fill).eq(0)
        .And(qa.bitwiseAnd(dilated).eq(0))
        .And(qa.bitwiseAnd(cirrus).eq(0))
        .And(qa.bitwiseAnd(cloud).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow).eq(0))
        .And(qa.bitwiseAnd(snow).eq(0))
    )
    return image.updateMask(mask)


def harmonize_landsat_sr(image, sensor: str):
    """Scale and rename Landsat bands to a common six-band layout."""
    ee = require_ee()
    sensor_key = sensor.lower()
    if sensor_key not in _BAND_MAPS:
        raise ValueError(f"Unknown sensor: {sensor}")

    band_map = _BAND_MAPS[sensor_key]
    optical_source = list(band_map.keys())
    optical = _scale_optical(image.select(optical_source))
    renamed = optical.rename(list(band_map.values()))
    return renamed


def load_landsat_sr(
    start_date: str,
    end_date: str,
    region=None,
    sensors: Sequence[str] = ("landsat5", "landsat7", "landsat8", "landsat9"),
    cloud_cover: int = 30,
    composite: bool = True,
):
    """Load a harmonized Landsat surface reflectance composite."""
    ee = require_ee()
    collections = []

    for sensor in sensors:
        key = sensor.lower()
        if key not in LANDSAT_COLLECTION_IDS:
            raise ValueError(f"Unknown sensor: {sensor}")
        collection = (
            ee.ImageCollection(LANDSAT_COLLECTION_IDS[key])
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUD_COVER", cloud_cover))
        )
        if region is not None:
            collection = collection.filterBounds(region)
        collection = collection.map(mask_landsat_clouds)
        collection = collection.map(lambda img, s=key: harmonize_landsat_sr(img, s))
        collections.append(collection)

    merged = ee.ImageCollection(collections).flatten()
    image = ee.Image(merged.median() if composite else merged.mosaic()).select(COMMON_LANDSAT_BANDS)
    if region is not None:
        image = image.clip(region)
    return image
