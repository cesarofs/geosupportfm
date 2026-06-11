from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from geosupportfm.benchmarks.synthetic_spatial_analysis import compute_pca_projection
from geosupportfm.visualization.synthetic_spatial_plots import (
    plot_feature_importance,
    plot_map,
    plot_pca_projection,
    plot_scatter,
)


def test_plot_functions_write_files(tmp_path):
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    importance_df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.4, 0.2]})
    proj = compute_pca_projection(np.random.RandomState(0).randn(12, 3), np.arange(12) % 3, label_name="lulc")

    p1 = plot_scatter(y_true, y_pred, "scatter", tmp_path / "scatter.png")
    p2 = plot_map(df, [0.1, 0.2, 0.3], "map", tmp_path / "map.png")
    p3 = plot_feature_importance(importance_df, "imp", tmp_path / "importance.png")
    p4 = plot_pca_projection(proj, "lulc", "pca", tmp_path / "pca.png")

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4.exists()
