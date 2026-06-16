import numpy as np

from geosupportfm.variograms.core import empirical_cross_variogram, empirical_variogram, variogram_summary
from geosupportfm.variograms.diagnostics import spatial_structure_diagnostics
from geosupportfm.variograms.latent import latent_variography_report


def make_grid(n=8):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    return coords


def test_empirical_variogram_returns_frame():
    coords = make_grid()
    values = coords[:, 0].astype(float)
    frame = empirical_variogram(coords, values, n_lags=6, pair_min=1)
    assert list(frame.columns) == ["lag_center", "semivariance", "npairs"]
    assert len(frame) == 6
    assert frame["npairs"].sum() > 0


def test_cross_variogram_matches_self_variogram_shape():
    coords = make_grid()
    a = coords[:, 0].astype(float)
    b = coords[:, 0].astype(float)
    v = empirical_variogram(coords, a, n_lags=5, pair_min=1)
    cv = empirical_cross_variogram(coords, a, b, n_lags=5, pair_min=1)
    assert len(v) == len(cv)
    assert np.isclose(v["lag_center"].iloc[0], cv["lag_center"].iloc[0])


def test_variogram_summary_has_expected_fields():
    coords = make_grid()
    values = coords[:, 0].astype(float)
    frame = empirical_variogram(coords, values, n_lags=5, pair_min=1)
    summary = variogram_summary(frame)
    assert summary.sill >= 0
    assert 0 <= summary.nugget_ratio <= 1
    assert 0 <= summary.structure_score <= 1


def test_latent_variography_report_ranks_dimensions():
    coords = make_grid()
    emb = np.column_stack([
        coords[:, 0].astype(float),
        np.random.default_rng(0).normal(size=len(coords)),
        coords[:, 0].astype(float) * 0.5,
    ])
    report = latent_variography_report(coords, emb, n_lags=5, pair_min=1)
    assert list(report.columns) == [
        "dimension",
        "name",
        "nugget",
        "sill",
        "practical_range",
        "nugget_ratio",
        "structure_score",
        "monotonicity",
    ]
    assert len(report) == 3
    assert report.iloc[0]["structure_score"] >= report.iloc[-1]["structure_score"]


def test_spatial_structure_diagnostics_returns_summary_and_report():
    coords = make_grid()
    emb = np.column_stack([
        coords[:, 0].astype(float),
        coords[:, 1].astype(float),
    ])
    summary, report = spatial_structure_diagnostics(coords, emb, n_lags=4, pair_min=1)
    assert summary.n_dimensions == 2
    assert 0 <= summary.structured_fraction <= 1
    assert len(report) == 2
