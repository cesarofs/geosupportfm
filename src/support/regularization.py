"""Numerical change-of-support covariance regularization.

Spatial supports are represented by weighted quadrature offsets around each
location.  Point-to-point, point-to-block, and block-to-block covariances are
then the corresponding weighted averages of the punctual LMC covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geosupportfm.coregionalization.lmc import LinearModelOfCoregionalization


@dataclass(frozen=True)
class SpatialSupport:
    """Weighted discretization offsets relative to a support centroid."""

    offsets: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if offsets.ndim != 2 or len(offsets) == 0 or len(offsets) != len(weights):
            raise ValueError("offsets must be (n_points, n_dimensions) and match weights")
        if np.any(~np.isfinite(offsets)) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("support offsets and weights must be finite and weights non-negative")
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("support weights must have a positive sum")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "weights", weights / total)

    @property
    def n_dimensions(self) -> int:
        return int(self.offsets.shape[1])

    @classmethod
    def point(cls, n_dimensions: int = 2) -> "SpatialSupport":
        if n_dimensions < 1:
            raise ValueError("n_dimensions must be positive")
        return cls(np.zeros((1, n_dimensions)), np.ones(1))

    @classmethod
    def rectangle(
        cls,
        size: Sequence[float],
        discretization: int | Sequence[int] = 5,
    ) -> "SpatialSupport":
        """Create midpoint quadrature for a rectangular 2D or 3D block."""
        extent = np.asarray(tuple(size), dtype=float)
        if extent.ndim != 1 or len(extent) not in {2, 3} or np.any(extent <= 0):
            raise ValueError("size must contain two or three positive extents")
        if isinstance(discretization, (int, np.integer)):
            counts = np.repeat(int(discretization), len(extent))
        else:
            counts = np.asarray(tuple(discretization), dtype=int)
        if counts.shape != extent.shape or np.any(counts < 1):
            raise ValueError("discretization must provide positive counts for every dimension")
        axes = [(-width / 2.0) + (np.arange(n) + 0.5) * width / n for width, n in zip(extent, counts)]
        mesh = np.meshgrid(*axes, indexing="ij")
        offsets = np.column_stack([axis.reshape(-1) for axis in mesh])
        return cls(offsets, np.ones(len(offsets)))


@dataclass(frozen=True)
class SupportRegularizationResult:
    """A regularized covariance/variogram profile and variance corrections."""

    lags: np.ndarray
    covariance: np.ndarray
    variogram: np.ndarray
    point_sill: np.ndarray
    block_sill: np.ndarray
    variance_ratio: np.ndarray
    standard_deviation_ratio: np.ndarray


def _coords(coords: np.ndarray, n_dimensions: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_dimensions or np.any(~np.isfinite(arr)):
        raise ValueError(f"coordinates must have shape (n_locations, {n_dimensions})")
    return arr


def _supports(value: SpatialSupport | Sequence[SpatialSupport], n: int) -> tuple[SpatialSupport, ...]:
    if isinstance(value, SpatialSupport):
        return (value,) * n
    result = tuple(value)
    if len(result) != n or not all(isinstance(item, SpatialSupport) for item in result):
        raise ValueError("a support sequence must contain one SpatialSupport per location")
    return result


def support_covariance_matrix(
    lmc: LinearModelOfCoregionalization,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    support_a: SpatialSupport | Sequence[SpatialSupport],
    support_b: SpatialSupport | Sequence[SpatialSupport],
    component: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Integrate an LMC into an observation-major support covariance matrix."""
    first_support = support_a if isinstance(support_a, SpatialSupport) else tuple(support_a)[0]
    dimensions = first_support.n_dimensions
    a = _coords(coords_a, dimensions)
    b = _coords(coords_b, dimensions)
    supports_a = _supports(support_a, len(a))
    supports_b = _supports(support_b, len(b))
    if any(s.n_dimensions != dimensions for s in supports_a + supports_b):
        raise ValueError("all supports must use the coordinate dimensionality")

    p = lmc.n_variables
    result = np.empty((len(a) * p, len(b) * p), dtype=float)
    for i, (center_a, shape_a) in enumerate(zip(a, supports_a)):
        points_a = center_a + shape_a.offsets
        for j, (center_b, shape_b) in enumerate(zip(b, supports_b)):
            points_b = center_b + shape_b.offsets
            delta = points_a[:, None, :] - points_b[None, :, :]
            distances = np.sqrt(np.sum(delta * delta, axis=2))
            covariance = lmc.covariance_values(distances, component=component)
            averaged = np.einsum("a,abij,b->ij", shape_a.weights, covariance, shape_b.weights)
            result[i * p : (i + 1) * p, j * p : (j + 1) * p] = averaged
    return result


def regularize_lmc(
    lmc: LinearModelOfCoregionalization,
    support: SpatialSupport,
    lags: np.ndarray,
    direction: Sequence[float] | None = None,
    component: int | Sequence[int] | None = None,
) -> SupportRegularizationResult:
    """Regularize an LMC over a support and evaluate its block variogram.

    Lags are applied along ``direction``. For isotropic LMC structures the
    direction changes block geometry only when the support is anisotropic.
    """
    lag_values = np.asarray(lags, dtype=float).reshape(-1)
    if np.any(~np.isfinite(lag_values)) or np.any(lag_values < 0):
        raise ValueError("lags must be finite and non-negative")
    if direction is None:
        direction_vector = np.zeros(support.n_dimensions)
        direction_vector[0] = 1.0
    else:
        direction_vector = np.asarray(tuple(direction), dtype=float)
        if direction_vector.shape != (support.n_dimensions,) or np.linalg.norm(direction_vector) == 0:
            raise ValueError("direction must be a non-zero vector matching the support dimensions")
        direction_vector /= np.linalg.norm(direction_vector)

    origin = np.zeros((1, support.n_dimensions))
    covariance = np.empty((len(lag_values), lmc.n_variables, lmc.n_variables))
    for i, lag in enumerate(lag_values):
        shifted = (direction_vector * lag).reshape(1, -1)
        covariance[i] = support_covariance_matrix(
            lmc, origin, shifted, support, support, component=component
        ).reshape(lmc.n_variables, lmc.n_variables)
    block_sill = support_covariance_matrix(
        lmc, origin, origin, support, support, component=component
    ).reshape(lmc.n_variables, lmc.n_variables)
    point_sill = lmc.component_sill(component)
    point_variance = np.diag(point_sill)
    block_variance = np.diag(block_sill)
    variance_ratio = np.divide(
        block_variance,
        point_variance,
        out=np.full_like(block_variance, np.nan),
        where=point_variance > 0,
    )
    return SupportRegularizationResult(
        lags=lag_values,
        covariance=covariance,
        variogram=block_sill[None, :, :] - covariance,
        point_sill=point_sill,
        block_sill=block_sill,
        variance_ratio=variance_ratio,
        standard_deviation_ratio=np.sqrt(np.maximum(variance_ratio, 0.0)),
    )
