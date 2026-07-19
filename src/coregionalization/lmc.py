"""Linear Model of Coregionalization (LMC) construction and fitting.

An LMC represents a matrix-valued semivariogram as

    Gamma(h) = sum_k B_k g_k(h),

where every coregionalization matrix ``B_k`` is positive semidefinite.  The
positive-semidefinite constraint is enforced during fitting through a
Cholesky parameterization, which guarantees a permissible multivariate model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_SUPPORTED_MODELS = {"nugget", "exponential", "spherical", "gaussian"}


def _distances(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def normalized_covariance(h: np.ndarray, model: str, range_param: float = 1.0) -> np.ndarray:
    """Evaluate a unit-sill isotropic covariance model.

    ``range_param`` is the practical range for exponential and Gaussian
    structures and the compact support for a spherical structure.
    """
    distance = np.asarray(h, dtype=float)
    if np.any(distance < 0) or np.any(~np.isfinite(distance)):
        raise ValueError("h must contain finite non-negative distances")
    name = str(model).lower()
    if name not in _SUPPORTED_MODELS:
        raise ValueError(f"unsupported covariance model: {model}")
    if name == "nugget":
        return np.isclose(distance, 0.0, rtol=0.0, atol=1e-12).astype(float)
    if not np.isfinite(range_param) or range_param <= 0:
        raise ValueError("range_param must be positive for structured models")

    ratio = distance / float(range_param)
    if name == "exponential":
        return np.exp(-3.0 * ratio)
    if name == "gaussian":
        return np.exp(-3.0 * ratio * ratio)
    return np.where(ratio < 1.0, 1.0 - 1.5 * ratio + 0.5 * ratio**3, 0.0)


def normalized_variogram(h: np.ndarray, model: str, range_param: float = 1.0) -> np.ndarray:
    """Evaluate the unit-sill variogram paired with ``normalized_covariance``."""
    return 1.0 - normalized_covariance(h, model=model, range_param=range_param)


def _validate_psd(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("coregionalization matrices must be square")
    if np.any(~np.isfinite(arr)) or not np.allclose(arr, arr.T, atol=tolerance):
        raise ValueError("coregionalization matrices must be finite and symmetric")
    if float(np.min(np.linalg.eigvalsh(arr))) < -tolerance:
        raise ValueError("coregionalization matrices must be positive semidefinite")
    return 0.5 * (arr + arr.T)


@dataclass(frozen=True)
class LMCStructure:
    """One nested spatial structure and its coregionalization matrix."""

    model: str
    range_param: float
    coregionalization: np.ndarray

    def __post_init__(self) -> None:
        name = str(self.model).lower()
        if name not in _SUPPORTED_MODELS:
            raise ValueError(f"unsupported covariance model: {self.model}")
        if name != "nugget" and (not np.isfinite(self.range_param) or self.range_param <= 0):
            raise ValueError("range_param must be positive for structured models")
        object.__setattr__(self, "model", name)
        object.__setattr__(self, "coregionalization", _validate_psd(self.coregionalization))

    def covariance(self, h: np.ndarray) -> np.ndarray:
        return normalized_covariance(h, self.model, self.range_param)

    def variogram(self, h: np.ndarray) -> np.ndarray:
        return normalized_variogram(h, self.model, self.range_param)


@dataclass(frozen=True)
class LinearModelOfCoregionalization:
    """A permissible nested multivariate covariance and variogram model."""

    structures: tuple[LMCStructure, ...]

    def __post_init__(self) -> None:
        structures = tuple(self.structures)
        if not structures:
            raise ValueError("at least one LMC structure is required")
        n_variables = structures[0].coregionalization.shape[0]
        if any(s.coregionalization.shape != (n_variables, n_variables) for s in structures):
            raise ValueError("all coregionalization matrices must have the same shape")
        object.__setattr__(self, "structures", structures)

    @property
    def n_variables(self) -> int:
        return int(self.structures[0].coregionalization.shape[0])

    @property
    def sill_matrix(self) -> np.ndarray:
        return np.sum([s.coregionalization for s in self.structures], axis=0)

    def component_sill(self, component: int | Sequence[int] | None = None) -> np.ndarray:
        """Return the sill matrix for the selected component or component sum."""
        return np.sum([s.coregionalization for s in self._select(component)], axis=0)

    def _select(self, component: int | Sequence[int] | None) -> tuple[LMCStructure, ...]:
        if component is None:
            return self.structures
        indices = (component,) if isinstance(component, (int, np.integer)) else tuple(component)
        if not indices or any(i < 0 or i >= len(self.structures) for i in indices):
            raise IndexError("component index is outside the LMC")
        return tuple(self.structures[int(i)] for i in indices)

    def covariance_values(self, h: np.ndarray, component: int | Sequence[int] | None = None) -> np.ndarray:
        """Return matrix-valued covariance with shape ``h.shape + (p, p)``."""
        distance = np.asarray(h, dtype=float)
        result = np.zeros(distance.shape + (self.n_variables, self.n_variables), dtype=float)
        for structure in self._select(component):
            result += structure.covariance(distance)[..., None, None] * structure.coregionalization
        return result

    def variogram_values(self, h: np.ndarray, component: int | Sequence[int] | None = None) -> np.ndarray:
        """Return matrix-valued semivariogram with shape ``h.shape + (p, p)``."""
        distance = np.asarray(h, dtype=float)
        result = np.zeros(distance.shape + (self.n_variables, self.n_variables), dtype=float)
        for structure in self._select(component):
            result += structure.variogram(distance)[..., None, None] * structure.coregionalization
        return result

    def covariance_matrix(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        component: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Build an observation-major covariance matrix.

        Rows are ordered ``[location_0 variable_0, location_0 variable_1, ...]``.
        """
        a = _validate_coords(coords_a)
        b = _validate_coords(coords_b)
        blocks = self.covariance_values(_distances(a, b), component=component)
        return blocks.transpose(0, 2, 1, 3).reshape(len(a) * self.n_variables, len(b) * self.n_variables)


@dataclass(frozen=True)
class LMCFitResult:
    """Result and diagnostics from constrained weighted least-squares fitting."""

    lmc: LinearModelOfCoregionalization
    fitted_variograms: np.ndarray
    residuals: np.ndarray
    cost: float
    success: bool
    message: str


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 1 or len(arr) < 1 or np.any(~np.isfinite(arr)):
        raise ValueError("coordinates must be a non-empty finite 2D array")
    return arr


def empirical_variogram_matrix(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 12,
    max_distance: float | None = None,
    pair_min: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate direct and cross-semivariograms for all variables.

    Returns ``(lag_centers, gamma, pair_counts)`` where ``gamma`` has shape
    ``(n_lags, n_variables, n_variables)``. Empty lag bins contain NaNs.
    """
    xy = _validate_coords(coords)
    z = np.asarray(values, dtype=float)
    if z.ndim != 2 or len(z) != len(xy) or np.any(~np.isfinite(z)):
        raise ValueError("values must be a finite array with shape (n_samples, n_variables)")
    if n_lags < 1 or pair_min < 1 or len(xy) < 2:
        raise ValueError("n_lags and pair_min must be positive and at least two samples are required")

    i, j = np.triu_indices(len(xy), k=1)
    distances = _distances(xy, xy)[i, j]
    upper = float(np.max(distances)) if max_distance is None else float(max_distance)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("max_distance must be positive")
    edges = np.linspace(0.0, upper, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full((n_lags, z.shape[1], z.shape[1]), np.nan)
    counts = np.zeros(n_lags, dtype=int)
    differences = z[i] - z[j]
    cross_products = 0.5 * differences[:, :, None] * differences[:, None, :]
    for lag in range(n_lags):
        mask = (distances >= edges[lag]) & (
            (distances <= edges[lag + 1]) if lag == n_lags - 1 else (distances < edges[lag + 1])
        )
        counts[lag] = int(mask.sum())
        if counts[lag] >= pair_min:
            gamma[lag] = np.mean(cross_products[mask], axis=0)
    return centers, gamma, counts


def _lower_to_psd(parameters: np.ndarray, n_variables: int) -> np.ndarray:
    lower = np.zeros((n_variables, n_variables), dtype=float)
    lower[np.tril_indices(n_variables)] = parameters
    return lower @ lower.T


def _psd_seed(matrix: np.ndarray, scale: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T) / max(scale, 1.0)
    values, vectors = np.linalg.eigh(symmetric)
    positive = (vectors * np.sqrt(np.maximum(values, 1e-8))) @ vectors.T
    lower = np.linalg.cholesky(positive @ positive.T + np.eye(len(matrix)) * 1e-10)
    return lower[np.tril_indices(len(matrix))]


def fit_lmc(
    lags: np.ndarray,
    empirical: np.ndarray,
    structures: Sequence[LMCStructure],
    weights: np.ndarray | None = None,
    fit_ranges: bool = False,
    max_nfev: int = 5000,
) -> LMCFitResult:
    """Fit nested LMC structures by constrained weighted least squares.

    The supplied structures provide model types, initial ranges, and matrix
    dimensions. Coregionalization matrices are always fitted. Structured
    ranges are fitted as positive parameters when ``fit_ranges=True``.
    """
    from scipy.optimize import least_squares

    h = np.asarray(lags, dtype=float).reshape(-1)
    observed = np.asarray(empirical, dtype=float)
    seeds = tuple(structures)
    if not seeds:
        raise ValueError("at least one initial structure is required")
    p = seeds[0].coregionalization.shape[0]
    if observed.shape != (len(h), p, p):
        raise ValueError("empirical must have shape (n_lags, n_variables, n_variables)")
    if any(s.coregionalization.shape != (p, p) for s in seeds):
        raise ValueError("all structures must match the empirical variable count")
    valid = np.isfinite(observed)
    valid &= valid.transpose(0, 2, 1)
    lower_mask = np.tril(np.ones((p, p), dtype=bool))[None, :, :] & valid
    if not np.any(lower_mask):
        raise ValueError("empirical contains no finite lower-triangular values")

    if weights is None:
        weight_array = np.ones_like(observed)
    else:
        raw_weights = np.asarray(weights, dtype=float)
        if raw_weights.ndim == 1 and len(raw_weights) == len(h):
            weight_array = np.broadcast_to(raw_weights[:, None, None], observed.shape)
        elif raw_weights.shape == observed.shape:
            weight_array = raw_weights
        else:
            raise ValueError("weights must have shape (n_lags,) or empirical.shape")
        if np.any(weight_array[lower_mask] < 0) or np.any(~np.isfinite(weight_array[lower_mask])):
            raise ValueError("weights must be finite and non-negative")

    n_lower = p * (p + 1) // 2
    finite_matrices = observed[np.all(np.isfinite(observed), axis=(1, 2))]
    sill_seed = finite_matrices[-1] if len(finite_matrices) else np.eye(p)
    parameters = np.concatenate([_psd_seed(sill_seed, len(seeds)) for _ in seeds])
    structured_indices = [i for i, s in enumerate(seeds) if s.model != "nugget"]
    if fit_ranges:
        parameters = np.concatenate([parameters, np.log([seeds[i].range_param for i in structured_indices])])

    def unpack(theta: np.ndarray) -> LinearModelOfCoregionalization:
        matrices = [_lower_to_psd(theta[k * n_lower : (k + 1) * n_lower], p) for k in range(len(seeds))]
        fitted_ranges = {i: seeds[i].range_param for i in range(len(seeds))}
        if fit_ranges:
            start = len(seeds) * n_lower
            fitted_ranges.update({idx: float(np.exp(theta[start + k])) for k, idx in enumerate(structured_indices)})
        return LinearModelOfCoregionalization(
            tuple(LMCStructure(s.model, fitted_ranges[i], matrices[i]) for i, s in enumerate(seeds))
        )

    scale = np.sqrt(weight_array[lower_mask])

    def objective(theta: np.ndarray) -> np.ndarray:
        model_values = unpack(theta).variogram_values(h)
        return (model_values[lower_mask] - observed[lower_mask]) * scale

    solution = least_squares(objective, parameters, max_nfev=max_nfev)
    lmc = unpack(solution.x)
    fitted = lmc.variogram_values(h)
    residuals = fitted - observed
    return LMCFitResult(lmc, fitted, residuals, float(solution.cost), bool(solution.success), str(solution.message))
