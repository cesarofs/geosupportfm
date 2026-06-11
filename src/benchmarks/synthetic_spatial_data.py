"""Synthetic spatial data generation and splitting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_RF_FEATURES = [
    "elevation",
    "t2m",
    "rh2m",
    "ps",
    "ws2m",
    "wd2m",
    "allsky_sfc_sw_dwn",
    "satellite_chirps",
    "satellite_merge",
    "t2m_range",
    "dewpoint_deficit",
    "satellite_gap",
    "ws2m_sq",
    "rh_temp_interaction",
]


def exponential_covariance(
    coords: np.ndarray,
    sill: float = 1.0,
    range_param: float = 18.0,
    nugget: float = 0.08,
) -> np.ndarray:
    """Build an exponential covariance matrix from 2D coordinates."""
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    cov = sill * np.exp(-dist / range_param)
    return cov + np.eye(len(coords)) * nugget


def simulate_dataset(n_points: int = 600, seed: int = 42) -> pd.DataFrame:
    """Simulate a spatial regression dataset with a geostatistical signal."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 100, size=n_points)
    y = rng.uniform(0, 100, size=n_points)
    coords = np.column_stack([x, y])

    cov = exponential_covariance(coords, sill=35.0, range_param=16.0, nugget=2.5)
    spatial_residual = rng.multivariate_normal(mean=np.zeros(n_points), cov=cov)

    elevation = (
        450
        + 1.8 * (100 - y)
        + 0.8 * np.sin(x / 8.0) * 40
        + 0.6 * np.cos(y / 10.0) * 25
        + rng.normal(0, 12, size=n_points)
    )

    t2m = 29.0 - 0.015 * elevation + 0.015 * x + rng.normal(0, 1.4, size=n_points)
    rh2m = 82.0 - 0.55 * (t2m - 24.0) + rng.normal(0, 3.5, size=n_points)
    ps = 101.5 - 0.0016 * elevation + rng.normal(0, 0.25, size=n_points)
    ws2m = np.clip(2.5 + 0.008 * elevation / 10.0 + rng.normal(0, 0.8, size=n_points), 0.1, None)
    wd2m = (180 + 1.5 * x + rng.normal(0, 25, size=n_points)) % 360
    allsky_sfc_sw_dwn = 18.0 + 0.14 * x - 0.05 * elevation / 100.0 + rng.normal(0, 1.8, size=n_points)

    rainfall_signal = (
        70.0
        + 0.12 * elevation
        + 0.90 * rh2m
        - 1.65 * t2m
        - 1.25 * ws2m
        + 0.55 * allsky_sfc_sw_dwn
        + 0.12 * np.maximum(elevation - 550.0, 0.0)
        + 0.025 * (rh2m * allsky_sfc_sw_dwn) / 10.0
    )

    rainfall_station = np.clip(
        rainfall_signal + 0.55 * spatial_residual + rng.normal(0, 7.5, size=n_points),
        0,
        None,
    )

    satellite_chirps = np.clip(
        rainfall_station + 0.80 * spatial_residual + rng.normal(0, 18.0, size=n_points),
        0,
        None,
    )

    satellite_merge = np.clip(
        rainfall_station + 0.35 * spatial_residual + rng.normal(0, 10.0, size=n_points),
        0,
        None,
    )

    df = pd.DataFrame(
        {
            "id": np.arange(n_points),
            "x": x,
            "y": y,
            "elevation": elevation,
            "t2m": t2m,
            "rh2m": rh2m,
            "ps": ps,
            "ws2m": ws2m,
            "wd2m": wd2m,
            "allsky_sfc_sw_dwn": allsky_sfc_sw_dwn,
            "satellite_chirps": satellite_chirps,
            "satellite_merge": satellite_merge,
            "rainfall_station": rainfall_station,
        }
    )

    df["t2m_max"] = df["t2m"] + 3.0 + rng.normal(0, 0.7, size=n_points)
    df["t2m_min"] = df["t2m"] - 4.0 + rng.normal(0, 0.7, size=n_points)
    df["t2mdew"] = df["t2m"] - (100.0 - df["rh2m"]) / 8.0 + rng.normal(0, 0.5, size=n_points)
    df["t2m_range"] = df["t2m_max"] - df["t2m_min"]
    df["dewpoint_deficit"] = df["t2m"] - df["t2mdew"]
    df["satellite_gap"] = df["satellite_merge"] - df["satellite_chirps"]
    df["ws2m_sq"] = df["ws2m"] ** 2
    df["rh_temp_interaction"] = df["rh2m"] * df["t2m"]
    return df


def build_spatial_blocks(df: pd.DataFrame, n_blocks: int = 8, seed: int = 42) -> np.ndarray:
    """Cluster coordinates into spatial blocks using KMeans."""
    coords = df[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def split_train_test(
    df: pd.DataFrame,
    groups: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into train/test while preserving block structure."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(df, groups=groups))
