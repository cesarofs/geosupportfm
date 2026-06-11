"""Embedding workflows for GeoSupportFM."""

from .config import GeoSupportEmbeddingConfig, default_embedding_bands
from .analysis import (
    EvaluationResult,
    axis_importance,
    centroid_cosine_similarity,
    class_centroids,
    count_samples_by_year_class,
    evaluate_classifier,
    feature_matrix,
    fit_random_forest,
    prepare_dataframe,
    split_by_year,
    top_similar_pairs,
)

__all__ = [
    "GeoSupportEmbeddingConfig",
    "default_embedding_bands",
    "EvaluationResult",
    "axis_importance",
    "centroid_cosine_similarity",
    "class_centroids",
    "count_samples_by_year_class",
    "evaluate_classifier",
    "feature_matrix",
    "fit_random_forest",
    "prepare_dataframe",
    "split_by_year",
    "top_similar_pairs",
]
