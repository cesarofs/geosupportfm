"""Pure-Python spatial support utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit


def validate_coordinates(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2)")
    if len(arr) == 0:
        raise ValueError("coords cannot be empty")
    return arr


def assign_spatial_blocks(coords: np.ndarray, n_blocks: int, random_state: int = 42) -> np.ndarray:
    """Assign each coordinate pair to a spatial block using KMeans."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    arr = validate_coordinates(coords)
    model = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    return model.fit_predict(arr)


def grouped_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Split a dataframe into grouped train/test indices."""
    if group_col not in df.columns:
        raise KeyError(f"{group_col} not in dataframe")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(df, groups=df[group_col]))


def aggregate_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_cols: Sequence[str],
    reducer: str = "mean",
) -> pd.DataFrame:
    """Aggregate numeric columns by a group label."""
    if group_col not in df.columns:
        raise KeyError(f"{group_col} not in dataframe")
    missing = [col for col in value_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if reducer not in {"mean", "sum", "median", "min", "max"}:
        raise ValueError("Unsupported reducer")

    grouped = df.groupby(group_col, dropna=False)[list(value_cols)]
    result = getattr(grouped, reducer)().reset_index()
    return result
