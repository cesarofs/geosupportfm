import numpy as np
import pytest

from geosupportfm.coregionalization import (
    LMCStructure,
    LinearModelOfCoregionalization,
    empirical_variogram_matrix,
    fit_lmc,
)


def make_lmc():
    return LinearModelOfCoregionalization(
        (
            LMCStructure("exponential", 10.0, np.array([[2.0, 0.5], [0.5, 1.5]])),
            LMCStructure("exponential", 50.0, np.array([[4.0, 2.5], [2.5, 3.0]])),
        )
    )


def test_lmc_covariance_is_symmetric_and_psd():
    lmc = make_lmc()
    coords = np.array([[0.0, 0.0], [2.0, 1.0], [10.0, 5.0]])
    covariance = lmc.covariance_matrix(coords, coords)
    assert covariance.shape == (6, 6)
    assert np.allclose(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-10


def test_lmc_rejects_non_psd_coregionalization():
    with pytest.raises(ValueError, match="positive semidefinite"):
        LMCStructure("exponential", 10.0, np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_empirical_variogram_matrix_contains_direct_and_cross_terms():
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    coords = np.column_stack([x.ravel(), y.ravel()])
    values = np.column_stack([coords[:, 0], 2.0 * coords[:, 0] + coords[:, 1]])
    lags, gamma, counts = empirical_variogram_matrix(coords, values, n_lags=4, pair_min=1)
    assert gamma.shape == (4, 2, 2)
    assert np.allclose(gamma, gamma.transpose(0, 2, 1), equal_nan=True)
    assert np.all(counts > 0)
    assert np.all(lags > 0)


def test_fit_lmc_reproduces_permissible_nested_model():
    expected = make_lmc()
    lags = np.linspace(1.0, 80.0, 20)
    empirical = expected.variogram_values(lags)
    seeds = (
        LMCStructure("exponential", 10.0, np.eye(2)),
        LMCStructure("exponential", 50.0, np.eye(2)),
    )
    result = fit_lmc(lags, empirical, seeds)
    assert result.success
    assert np.allclose(result.fitted_variograms, empirical, atol=1e-4)
    assert all(np.min(np.linalg.eigvalsh(s.coregionalization)) >= -1e-10 for s in result.lmc.structures)

