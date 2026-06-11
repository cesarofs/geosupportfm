from __future__ import annotations

import numpy as np

from geosupportfm.benchmarks.synthetic_spatial_data import (
    DEFAULT_RF_FEATURES,
    build_spatial_blocks,
    exponential_covariance,
    simulate_dataset,
    split_train_test,
)


def test_simulate_dataset_has_expected_columns():
    df = simulate_dataset(n_points=40, seed=1)
    assert len(df) == 40
    for col in ["x", "y", "rainfall_station", *DEFAULT_RF_FEATURES]:
        assert col in df.columns


def test_exponential_covariance_is_symmetric_and_pd_like():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cov = exponential_covariance(coords, sill=2.0, range_param=3.0, nugget=0.1)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) > 0)


def test_build_spatial_blocks_and_split_preserve_groups():
    df = simulate_dataset(n_points=60, seed=2)
    groups = build_spatial_blocks(df, n_blocks=4, seed=2)
    assert len(np.unique(groups)) == 4

    tr_idx, te_idx = split_train_test(df, groups, test_size=0.25, seed=2)
    assert len(set(tr_idx).intersection(set(te_idx))) == 0
    assert len(tr_idx) + len(te_idx) == len(df)
