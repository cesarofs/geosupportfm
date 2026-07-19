import numpy as np

from geosupportfm.coregionalization import LMCStructure, LinearModelOfCoregionalization
from geosupportfm.support.embedding_aggregation import aggregate_embeddings_to_blocks
from geosupportfm.support.regularization import SpatialSupport, regularize_lmc, support_covariance_matrix


def make_lmc():
    return LinearModelOfCoregionalization(
        (LMCStructure("exponential", 20.0, np.array([[2.0, 0.6], [0.6, 1.0]])),)
    )


def test_point_support_matches_punctual_lmc():
    lmc = make_lmc()
    coords = np.array([[0.0, 0.0], [4.0, 0.0]])
    point = SpatialSupport.point(2)
    assert np.allclose(
        support_covariance_matrix(lmc, coords, coords, point, point),
        lmc.covariance_matrix(coords, coords),
    )


def test_block_regularization_reduces_variance_and_starts_at_zero():
    lmc = make_lmc()
    block = SpatialSupport.rectangle((10.0, 10.0), discretization=5)
    result = regularize_lmc(lmc, block, lags=np.array([0.0, 10.0, 20.0]))
    assert np.all(result.variance_ratio > 0)
    assert np.all(result.variance_ratio < 1)
    assert np.allclose(result.variogram[0], 0.0)
    assert np.allclose(result.block_sill, result.block_sill.T)


def test_point_to_block_covariance_has_expected_shape():
    lmc = make_lmc()
    points = SpatialSupport.point(2)
    blocks = SpatialSupport.rectangle((8.0, 8.0), discretization=3)
    covariance = support_covariance_matrix(
        lmc,
        np.array([[0.0, 0.0], [5.0, 0.0]]),
        np.array([[2.0, 1.0]]),
        points,
        blocks,
    )
    assert covariance.shape == (4, 2)


def test_block_embedding_aggregation_weighting_and_unit_norm():
    coords = np.array([[1.0, 1.0], [2.0, 2.0], [11.0, 1.0]])
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    result = aggregate_embeddings_to_blocks(
        coords,
        embeddings,
        block_size=(10.0, 10.0),
        origin=(0.0, 0.0),
        sample_weights=np.array([1.0, 3.0, 1.0]),
        unit_normalize=True,
    )
    assert np.array_equal(result.counts, [2, 1])
    assert np.allclose(np.linalg.norm(result.embeddings, axis=1), 1.0)
    assert result.embeddings[0, 1] > result.embeddings[0, 0]

