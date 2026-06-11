"""Pure analysis utilities for GeoSupportFM embedding experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class EvaluationResult:
    balanced_accuracy: float
    macro_f1: float
    classification_report_text: str
    confusion_matrix: pd.DataFrame


def prepare_dataframe(df: pd.DataFrame, embedding_columns: Sequence[str], label_column: str, year_column: str) -> pd.DataFrame:
    """Validate and normalize the analysis table."""
    required = list(embedding_columns) + [label_column, year_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cleaned = df.dropna(subset=required).copy()
    cleaned[label_column] = cleaned[label_column].astype(int)
    cleaned[year_column] = cleaned[year_column].astype(int)
    return cleaned


def split_by_year(
    df: pd.DataFrame,
    train_years: Sequence[int],
    test_years: Sequence[int],
    year_column: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table into train and test subsets by year."""
    train = df[df[year_column].isin(train_years)].copy()
    test = df[df[year_column].isin(test_years)].copy()
    return train, test


def feature_matrix(df: pd.DataFrame, embedding_columns: Sequence[str]) -> np.ndarray:
    """Extract the model matrix from a DataFrame."""
    return df.loc[:, embedding_columns].to_numpy(dtype=np.float32)


def label_vector(df: pd.DataFrame, label_column: str = "lulc") -> np.ndarray:
    """Extract the target vector from a DataFrame."""
    return df.loc[:, label_column].to_numpy(dtype=int)


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_estimators: int = 500,
) -> RandomForestClassifier:
    """Train a random forest classifier with stable defaults."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_classifier(
    model: RandomForestClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: Sequence[int] | None = None,
) -> tuple[EvaluationResult, np.ndarray]:
    """Evaluate a fitted classifier and return metrics plus predictions."""
    predictions = model.predict(x_test)
    if labels is None:
        labels_array = np.unique(np.concatenate([np.asarray(model.classes_), np.asarray(y_test)]))
    else:
        labels_array = np.asarray(list(labels))
    cm = confusion_matrix(y_test, predictions, labels=labels_array)
    cm_df = pd.DataFrame(cm, index=labels_array, columns=labels_array)
    report = classification_report(y_test, predictions, labels=labels_array, zero_division=0)
    result = EvaluationResult(
        balanced_accuracy=balanced_accuracy_score(y_test, predictions),
        macro_f1=f1_score(y_test, predictions, average="macro"),
        classification_report_text=report,
        confusion_matrix=cm_df,
    )
    return result, predictions


def axis_importance(model: RandomForestClassifier, embedding_columns: Sequence[str]) -> pd.Series:
    """Return feature importance ranked from strongest to weakest."""
    return pd.Series(model.feature_importances_, index=embedding_columns).sort_values(ascending=False)


def class_centroids(df: pd.DataFrame, embedding_columns: Sequence[str], label_column: str = "lulc") -> pd.DataFrame:
    """Compute class centroids in embedding space."""
    return df.groupby(label_column)[list(embedding_columns)].mean()


def centroid_cosine_similarity(centroids: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between class centroids."""
    similarity = cosine_similarity(centroids.to_numpy())
    return pd.DataFrame(similarity, index=centroids.index, columns=centroids.index)


def top_similar_pairs(similarity_df: pd.DataFrame, top_n: int = 10) -> list[tuple[int, int, float]]:
    """Return the top off-diagonal similarity pairs."""
    values = similarity_df.to_numpy().copy()
    np.fill_diagonal(values, -np.inf)
    pairs: list[tuple[int, int, float]] = []
    for _ in range(min(top_n, values.shape[0] * max(values.shape[1] - 1, 0) // 2)):
        i, j = np.unravel_index(np.argmax(values), values.shape)
        pairs.append((int(similarity_df.index[i]), int(similarity_df.columns[j]), float(values[i, j])))
        values[i, j] = -np.inf
        values[j, i] = -np.inf
    return pairs


def count_samples_by_year_class(df: pd.DataFrame, year_column: str = "year", label_column: str = "lulc") -> pd.DataFrame:
    """Count rows per year and class."""
    return (
        df.groupby([year_column, label_column])
        .size()
        .rename("n")
        .reset_index()
        .sort_values([year_column, "n"], ascending=[True, False])
    )
