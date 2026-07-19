"""Support-aware factorial cokriging for multivariate LMC models.

The solver follows the block matrix formulation of Ma et al. (2014).  The
left-hand covariance contains every LMC structure.  Factorial estimates use
only the requested structure on the right-hand side, producing a spatial-scale
filter. Ordinary cokriging uses the complete LMC on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from geosupportfm.coregionalization.lmc import LinearModelOfCoregionalization
from geosupportfm.support.regularization import SpatialSupport, support_covariance_matrix


@dataclass(frozen=True)
class FactorialCokrigingResult:
    """Predictions and numerical diagnostics from a cokriging solve."""

    estimates: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    lagrange_multipliers: np.ndarray
    condition_number: float
    component: int | tuple[int, ...] | None


def _validate_inputs(
    obs_coords: np.ndarray,
    obs_data: np.ndarray,
    target_coords: np.ndarray,
    lmc: LinearModelOfCoregionalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_coords = np.asarray(obs_coords, dtype=float)
    target = np.asarray(target_coords, dtype=float)
    data = np.asarray(obs_data, dtype=float)
    if observed_coords.ndim != 2 or target.ndim != 2 or observed_coords.shape[1] != target.shape[1]:
        raise ValueError("observation and target coordinates must be aligned 2D coordinate arrays")
    if data.shape != (len(observed_coords), lmc.n_variables):
        raise ValueError("obs_data must have shape (n_observations, lmc.n_variables)")
    if len(observed_coords) == 0 or len(target) == 0:
        raise ValueError("observation and target coordinates cannot be empty")
    if np.any(~np.isfinite(observed_coords)) or np.any(~np.isfinite(target)) or np.any(~np.isfinite(data)):
        raise ValueError("coordinates and observations must be finite")
    return observed_coords, data, target


def _constraint_matrix(n_locations: int, n_variables: int) -> np.ndarray:
    """Return per-variable constant drift columns for observation-major data."""
    return np.tile(np.eye(n_variables), (n_locations, 1))


def _solve(
    obs_coords: np.ndarray,
    obs_data: np.ndarray,
    target_coords: np.ndarray,
    lmc: LinearModelOfCoregionalization,
    component: int | Sequence[int] | None,
    obs_support: SpatialSupport | Sequence[SpatialSupport],
    target_support: SpatialSupport | Sequence[SpatialSupport],
    constraints: Literal["none", "ordinary", "zero_mean"],
    ridge: float,
) -> FactorialCokrigingResult:
    observed_coords, data, target = _validate_inputs(obs_coords, obs_data, target_coords, lmc)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    p = lmc.n_variables
    c_data = support_covariance_matrix(
        lmc, observed_coords, observed_coords, obs_support, obs_support
    )
    c_data = 0.5 * (c_data + c_data.T) + np.eye(len(c_data)) * ridge
    c_target = support_covariance_matrix(
        lmc, observed_coords, target, obs_support, target_support, component=component
    )
    condition_number = float(np.linalg.cond(c_data))

    if constraints == "none":
        weights = np.linalg.solve(c_data, c_target)
        multipliers = np.empty((0, c_target.shape[1]))
        target_constraints = np.empty((0, c_target.shape[1]))
    elif constraints in {"ordinary", "zero_mean"}:
        design = _constraint_matrix(len(observed_coords), p)
        zeros = np.zeros((p, p))
        system = np.block([[c_data, design], [design.T, zeros]])
        target_constraints = np.zeros((p, len(target) * p))
        if constraints == "ordinary":
            target_constraints = np.tile(np.eye(p), len(target))
        rhs = np.vstack([c_target, target_constraints])
        solution = np.linalg.solve(system, rhs)
        weights = solution[: len(c_data)]
        multipliers = solution[len(c_data) :]
        condition_number = float(np.linalg.cond(system))
    else:
        raise ValueError("constraints must be 'none', 'ordinary', or 'zero_mean'")

    estimates = (data.reshape(-1) @ weights).reshape(len(target), p)
    if isinstance(target_support, SpatialSupport):
        target_supports = (target_support,) * len(target)
    else:
        target_supports = tuple(target_support)
        if len(target_supports) != len(target):
            raise ValueError("target_support must contain one support per target")
    variances = np.empty((len(target), p), dtype=float)
    for location in range(len(target)):
        local_target = target[location : location + 1]
        marginal_covariance = support_covariance_matrix(
            lmc,
            local_target,
            local_target,
            target_supports[location],
            target_supports[location],
            component=component,
        )
        for variable in range(p):
            column = location * p + variable
            marginal = marginal_covariance[variable, variable]
            constraint_term = (
                float(target_constraints[:, column] @ multipliers[:, column])
                if len(multipliers)
                else 0.0
            )
            variances[location, variable] = max(
                0.0,
                float(marginal - weights[:, column] @ c_target[:, column] - constraint_term),
            )
    selected = None if component is None else (
        int(component) if isinstance(component, (int, np.integer)) else tuple(int(i) for i in component)
    )
    return FactorialCokrigingResult(estimates, variances, weights, multipliers, condition_number, selected)


def factorial_cokriging(
    obs_coords: np.ndarray,
    obs_data: np.ndarray,
    target_coords: np.ndarray,
    lmc: LinearModelOfCoregionalization,
    component: int | Sequence[int],
    obs_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    target_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    constraints: Literal["none", "zero_mean"] = "zero_mean",
    ridge: float = 1e-8,
) -> FactorialCokrigingResult:
    """Estimate one or more zero-mean LMC scale components.

    ``zero_mean`` constraints make every variable-specific weight sum zero,
    removing unknown constants as required for a factorial component. Use
    ``none`` when observations have already been centered under a known mean.
    """
    dimensions = np.asarray(obs_coords).shape[1]
    point = SpatialSupport.point(dimensions)
    return _solve(
        obs_coords,
        obs_data,
        target_coords,
        lmc,
        component,
        point if obs_support is None else obs_support,
        point if target_support is None else target_support,
        constraints,
        ridge,
    )


def ordinary_cokriging(
    obs_coords: np.ndarray,
    obs_data: np.ndarray,
    target_coords: np.ndarray,
    lmc: LinearModelOfCoregionalization,
    obs_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    target_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    ridge: float = 1e-8,
) -> FactorialCokrigingResult:
    """Perform ordinary point or block cokriging with all LMC structures."""
    dimensions = np.asarray(obs_coords).shape[1]
    point = SpatialSupport.point(dimensions)
    return _solve(
        obs_coords,
        obs_data,
        target_coords,
        lmc,
        None,
        point if obs_support is None else obs_support,
        point if target_support is None else target_support,
        "ordinary",
        ridge,
    )


def simple_cokriging(
    obs_coords: np.ndarray,
    obs_data: np.ndarray,
    target_coords: np.ndarray,
    lmc: LinearModelOfCoregionalization,
    obs_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    target_support: SpatialSupport | Sequence[SpatialSupport] | None = None,
    ridge: float = 1e-8,
) -> FactorialCokrigingResult:
    """Perform point or block cokriging for centered data with a known mean."""
    dimensions = np.asarray(obs_coords).shape[1]
    point = SpatialSupport.point(dimensions)
    return _solve(
        obs_coords,
        obs_data,
        target_coords,
        lmc,
        None,
        point if obs_support is None else obs_support,
        point if target_support is None else target_support,
        "none",
        ridge,
    )
