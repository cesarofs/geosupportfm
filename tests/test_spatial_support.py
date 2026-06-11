import numpy as np
import pandas as pd

from geosupportfm.support.spatial_support import (
    aggregate_by_group,
    assign_spatial_blocks,
    grouped_train_test_split,
    validate_coordinates,
)


def test_validate_coordinates_shape():
    coords = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = validate_coordinates(coords)
    assert out.shape == (2, 2)


def test_assign_spatial_blocks_is_reproducible():
    coords = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    blocks1 = assign_spatial_blocks(coords, n_blocks=2, random_state=7)
    blocks2 = assign_spatial_blocks(coords, n_blocks=2, random_state=7)
    assert np.array_equal(blocks1, blocks2)
    assert set(blocks1) <= {0, 1}


def test_grouped_train_test_split_uses_groups():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "group": [0, 0, 1, 1]})
    tr_idx, te_idx = grouped_train_test_split(df, "group", test_size=0.5, random_state=42)
    assert len(set(df.loc[tr_idx, "group"]).intersection(set(df.loc[te_idx, "group"]))) == 0


def test_aggregate_by_group_mean():
    df = pd.DataFrame({"group": [0, 0, 1], "a": [1.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0]})
    out = aggregate_by_group(df, "group", ["a", "b"], reducer="mean")
    assert out.loc[out["group"] == 0, "a"].iloc[0] == 2.0
    assert out.loc[out["group"] == 1, "b"].iloc[0] == 6.0
