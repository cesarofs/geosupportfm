import pandas as pd

from geosupportfm.visualization.embedding_plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_scatter,
    plot_similarity_heatmap,
)


def test_plot_functions_return_figures():
    df = pd.DataFrame(
        {
            "A00": [1.0, 1.1, 3.0, 3.2],
            "A01": [0.1, 0.2, 0.9, 1.0],
            "lulc": [1, 1, 2, 2],
        }
    )
    fig1, _ = plot_pca_scatter(df, ["A00", "A01"])
    fig2, _ = plot_feature_importance(pd.Series([0.5, 0.2], index=["A00", "A01"]))
    fig3, _ = plot_confusion_matrix(pd.DataFrame([[2, 0], [0, 2]], index=[1, 2], columns=[1, 2]))
    fig4, _ = plot_similarity_heatmap(pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=[1, 2], columns=[1, 2]))

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
