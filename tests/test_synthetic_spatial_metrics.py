from __future__ import annotations

import numpy as np

from geosupportfm.benchmarks.synthetic_spatial_metrics import expm1_safe, log1p_safe, metrics_dict, rmse


def test_safe_transforms_round_trip_positive_values():
    values = np.array([0.0, 1.0, 10.0, 25.0])
    transformed = log1p_safe(values)
    recovered = expm1_safe(transformed)
    assert np.allclose(recovered, values)


def test_safe_transforms_clip_negative_values():
    values = np.array([-5.0, 0.0, 3.0])
    transformed = log1p_safe(values)
    recovered = expm1_safe(transformed)
    assert recovered[0] == 0.0
    assert recovered[1] == 0.0
    assert recovered[2] > 0.0


def test_metrics_dict_contains_expected_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 3.5])
    m = metrics_dict(y_true, y_pred, prefix="Test_")
    assert set(m) == {"Test_RMSE", "Test_MAE", "Test_R2"}
    assert rmse(y_true, y_true) == 0.0
