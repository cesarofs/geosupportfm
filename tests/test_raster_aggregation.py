import numpy as np
import pytest

from geosupportfm.support.raster_aggregation import aggregate_array, aggregate_raster_stack


def test_aggregate_array_mean():
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    out = aggregate_array(arr, factor=2, reducer="mean")
    assert out.shape == (1, 1)
    assert out[0, 0] == 2.5


def test_aggregate_array_sum():
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    out = aggregate_array(arr, factor=2, reducer="sum")
    assert out[0, 0] == 10.0


def test_aggregate_raster_stack():
    arr1 = np.array([[1, 2], [3, 4]], dtype=float)
    arr2 = np.array([[5, 6], [7, 8]], dtype=float)
    out = aggregate_raster_stack([arr1, arr2], factor=2, reducer="mean")
    assert len(out) == 2
    assert out[0][0, 0] == 2.5
    assert out[1][0, 0] == 6.5


def test_aggregate_array_rejects_bad_factor():
    arr = np.ones((2, 2))
    with pytest.raises(ValueError):
        aggregate_array(arr, factor=0)
