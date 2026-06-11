from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from geosupportfm.benchmarks.synthetic_spatial_data import DEFAULT_RF_FEATURES, build_spatial_blocks, simulate_dataset
from geosupportfm.benchmarks.synthetic_spatial_models import RFOKHybridRegressor, fit_ordinary_kriging, oof_rf_predictions


def test_fit_ordinary_kriging_raises_helpful_error_without_pykrige():
    with pytest.raises(ImportError, match="Install pykrige first"):
        fit_ordinary_kriging([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])


def test_oof_rf_predictions_shape():
    df = simulate_dataset(n_points=48, seed=7)
    groups = build_spatial_blocks(df, n_blocks=4, seed=7)
    rf = RandomForestRegressor(n_estimators=20, random_state=7)
    preds = oof_rf_predictions(rf, df, df["rainfall_station"], groups, DEFAULT_RF_FEATURES[:4])
    assert preds.shape == (len(df),)


def test_hybrid_regressor_with_fake_kriging(monkeypatch):
    df = simulate_dataset(n_points=32, seed=9)
    groups = build_spatial_blocks(df, n_blocks=4, seed=9)
    X = df[DEFAULT_RF_FEATURES[:4] + ["x", "y"]].copy()
    X["__group__"] = groups
    y = df["rainfall_station"].to_numpy()

    class FakeOK:
        def __init__(self, *args, **kwargs):
            self.values = np.asarray(args[2], dtype=float)

        def execute(self, mode, x, y):
            return np.zeros(len(x)), np.zeros(len(x))

    module = __import__("geosupportfm.benchmarks.synthetic_spatial_models", fromlist=["dummy"])
    monkeypatch.setattr(module, "_load_ordinary_kriging_class", lambda: FakeOK)

    rf = RandomForestRegressor(n_estimators=10, random_state=9)
    model = RFOKHybridRegressor(rf_model=rf, rf_features=DEFAULT_RF_FEATURES[:4])
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.all(pred >= 0)
