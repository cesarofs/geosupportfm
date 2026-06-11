"""Matplotlib plots for embedding experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_pca_scatter(
    df: pd.DataFrame,
    embedding_columns: Sequence[str],
    label_column: str = "lulc",
    max_classes: int = 20,
    title: str = "PCA of Satellite Embeddings, training sample",
):
    """Create a 2D PCA scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(df.loc[:, embedding_columns].to_numpy(dtype=np.float32))

    plot_df = pd.DataFrame(
        {
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
            label_column: df.loc[:, label_column].to_numpy(),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in sorted(plot_df[label_column].unique())[:max_classes]:
        subset = plot_df[plot_df[label_column] == cls]
        ax.scatter(subset["pc1"], subset["pc2"], s=8, alpha=0.35, label=str(cls))

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_feature_importance(importance: pd.Series, top_n: int = 15, title: str = "Embedding Axis Importance"):
    """Plot the strongest feature importances."""
    top = importance.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index.astype(str), top.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = "Confusion Matrix"):
    """Plot a confusion matrix heatmap with matplotlib only."""
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm_df.to_numpy(), interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns.astype(str), rotation=90)
    ax.set_yticklabels(cm_df.index.astype(str))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_similarity_heatmap(similarity_df: pd.DataFrame, title: str = "Class Centroid Cosine Similarity"):
    """Plot a class similarity matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity_df.to_numpy(), vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_xticks(range(len(similarity_df.columns)))
    ax.set_yticks(range(len(similarity_df.index)))
    ax.set_xticklabels(similarity_df.columns.astype(str), rotation=90)
    ax.set_yticklabels(similarity_df.index.astype(str))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig, ax
