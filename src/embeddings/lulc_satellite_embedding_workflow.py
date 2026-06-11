"""End-to-end workflow for MapBiomas and satellite embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .analysis import (
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
    label_vector,
)
from .config import GeoSupportEmbeddingConfig
from .gee_sampling import build_samples, initialize_earth_engine, read_local_sample_csvs, export_yearly_samples_to_drive
from ..visualization.embedding_plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_scatter,
    plot_similarity_heatmap,
)


def load_samples_from_csvs(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Load and concatenate sampled CSV files from disk."""
    frames = []
    for path in csv_paths:
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No usable CSV files were found.")
    return pd.concat(frames, ignore_index=True)


def run_local_analysis(df: pd.DataFrame, config: GeoSupportEmbeddingConfig, output_dir: str | Path = ".") -> dict[str, object]:
    """Run the full analysis from a preloaded DataFrame."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataframe(df, config.emb_bands, config.lulc_column, config.year_column)
    prepared.to_csv(output_path / "brazil_lulc_embedding_samples.csv", index=False)

    train_df, test_df = split_by_year(prepared, config.train_years, config.test_years, year_column=config.year_column)
    x_train = feature_matrix(train_df, config.emb_bands)
    y_train = label_vector(train_df, config.lulc_column)
    x_test = feature_matrix(test_df, config.emb_bands)
    y_test = label_vector(test_df, config.lulc_column)

    model = fit_random_forest(x_train, y_train, seed=config.seed)
    result, _ = evaluate_classifier(model, x_test, y_test)
    result.confusion_matrix.to_csv(output_path / "confusion_matrix.csv")

    importance = axis_importance(model, config.emb_bands)
    importance.to_csv(output_path / "embedding_axis_importance.csv")

    centroids = class_centroids(train_df, config.emb_bands, config.lulc_column)
    centroid_similarity = centroid_cosine_similarity(centroids)
    centroid_similarity.to_csv(output_path / "class_centroid_cosine_similarity.csv")

    sample_counts = count_samples_by_year_class(prepared, config.year_column, config.lulc_column)
    sample_counts.to_csv(output_path / "sample_counts_by_year_class.csv", index=False)

    plot_pca_scatter(train_df, config.emb_bands, config.lulc_column)
    plot_feature_importance(importance)
    plot_confusion_matrix(result.confusion_matrix)
    plot_similarity_heatmap(centroid_similarity)

    return {
        "model": model,
        "evaluation": result,
        "feature_importance": importance,
        "centroid_similarity": centroid_similarity,
        "sample_counts": sample_counts,
    }


def run_from_drive_exports(config: GeoSupportEmbeddingConfig, csv_dir: str | Path, output_dir: str | Path = ".") -> dict[str, object]:
    """Load Drive exports from disk and run the analysis."""
    csv_paths = read_local_sample_csvs(csv_dir)
    df = load_samples_from_csvs(csv_paths)
    return run_local_analysis(df, config, output_dir=output_dir)


def export_yearly_samples(config: GeoSupportEmbeddingConfig, folder: str = "gee_lulc_embedding") -> list[object]:
    """Export yearly samples to Drive."""
    ee_module = initialize_earth_engine(config.project_id)
    return export_yearly_samples_to_drive(ee_module, config, folder=folder)


def build_feature_collection(config: GeoSupportEmbeddingConfig):
    """Build the merged Earth Engine FeatureCollection for smaller experiments."""
    ee_module = initialize_earth_engine(config.project_id)
    return build_samples(ee_module, config)


def main() -> None:
    """Example CLI-style entrypoint."""
    config = GeoSupportEmbeddingConfig(project_id="your-gcp-project-id")
    export_yearly_samples(config)


if __name__ == "__main__":
    main()
