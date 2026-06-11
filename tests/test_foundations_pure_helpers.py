import pytest

from geosupportfm.foundations.alphaearth import default_alphaearth_bands, year_bounds
from geosupportfm.foundations.landsat import common_landsat_bands, LANDSAT_COLLECTION_IDS
from geosupportfm.foundations.sentinel2 import default_sentinel2_bands


def test_alphaearth_year_bounds():
    assert year_bounds(2024) == ("2024-01-01", "2025-01-01")


def test_alphaearth_default_bands():
    bands = default_alphaearth_bands()
    assert len(bands) == 64
    assert bands[0] == "A01"
    assert bands[-1] == "A64"


def test_sentinel2_default_bands():
    assert default_sentinel2_bands() == ["B2", "B3", "B4", "B8", "B11", "B12"]


def test_landsat_collection_ids_and_bands():
    assert set(LANDSAT_COLLECTION_IDS) == {"landsat5", "landsat7", "landsat8", "landsat9"}
    assert common_landsat_bands() == ["blue", "green", "red", "nir", "swir1", "swir2"]
