import numpy as np

from geosupportfm.coregionalization import LMCStructure, LinearModelOfCoregionalization
from geosupportfm.kriging import factorial_cokriging, ordinary_cokriging, simple_cokriging
from geosupportfm.support import SpatialSupport


def make_lmc():
    return LinearModelOfCoregionalization(
        (
            LMCStructure("exponential", 5.0, np.array([[1.5, 0.3], [0.3, 1.0]])),
            LMCStructure("exponential", 30.0, np.array([[3.0, 1.4], [1.4, 2.0]])),
        )
    )


def observations():
    coords = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
    values = np.column_stack([np.sin(coords[:, 0] / 4) + coords[:, 1] / 10, coords[:, 0] / 10 + np.cos(coords[:, 1])])
    return coords, values


def test_factorial_components_are_additive_for_known_centered_mean():
    coords, values = observations()
    lmc = make_lmc()
    targets = np.array([[2.0, 2.0], [7.0, 3.0]])
    short = factorial_cokriging(coords, values, targets, lmc, component=0, constraints="none")
    long = factorial_cokriging(coords, values, targets, lmc, component=1, constraints="none")
    total = simple_cokriging(coords, values, targets, lmc)
    assert np.allclose(short.estimates + long.estimates, total.estimates)
    assert short.weights.shape == (len(coords) * 2, len(targets) * 2)


def test_zero_mean_factor_weights_remove_variable_specific_constants():
    coords, values = observations()
    lmc = make_lmc()
    result = factorial_cokriging(coords, values + np.array([100.0, -50.0]), coords[:2], lmc, component=0)
    design = np.tile(np.eye(2), (len(coords), 1))
    assert np.allclose(design.T @ result.weights, 0.0, atol=1e-10)
    baseline = factorial_cokriging(coords, values, coords[:2], lmc, component=0)
    assert np.allclose(result.estimates, baseline.estimates)


def test_ordinary_cokriging_preserves_variable_specific_constants():
    coords, _ = observations()
    lmc = make_lmc()
    constants = np.tile(np.array([4.0, -2.0]), (len(coords), 1))
    result = ordinary_cokriging(coords, constants, np.array([[3.0, 3.0], [8.0, 4.0]]), lmc)
    assert np.allclose(result.estimates, np.array([[4.0, -2.0], [4.0, -2.0]]))
    assert np.all(result.variances >= 0)


def test_factorial_block_estimation_runs_with_change_of_support():
    coords, values = observations()
    lmc = make_lmc()
    block = SpatialSupport.rectangle((4.0, 4.0), discretization=3)
    result = factorial_cokriging(
        coords, values, np.array([[3.0, 3.0]]), lmc, component=1, target_support=block
    )
    assert result.estimates.shape == (1, 2)
    assert result.variances.shape == (1, 2)
    assert np.all(np.isfinite(result.estimates))
