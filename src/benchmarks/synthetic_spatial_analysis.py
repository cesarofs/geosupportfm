"""Analysis helpers for benchmark outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate_regression_models(y_true, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Create a compact comparison table for regression predictions."""
    rows = []
    y_true = np.asarray(y_true, dtype=float)
    for model_name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=float)
        rows.append(
            {
                "Model": model_name,
                "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "MAE": float(np.mean(np.abs(y_true - y_pred))),
                "R2": float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE")


def permutation_importance_table(
    model,
    X,
    y,
    feature_names: list[str],
    random_state: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Return a sorted feature-importance table from permutation importance."""
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance", ascending=False)


def compute_pca_projection(
    X,
    labels,
    n_components: int = 2,
    random_state: int = 42,
    label_name: str = "label",
) -> pd.DataFrame:
    """Project feature space into two principal components for plotting."""
    pca = PCA(n_components=n_components, random_state=random_state)
    z = pca.fit_transform(np.asarray(X, dtype=np.float32))
    if n_components != 2:
        raise ValueError("This helper currently expects n_components=2.")
    return pd.DataFrame({"pc1": z[:, 0], "pc2": z[:, 1], label_name: np.asarray(labels)})


def classification_summary(y_true, y_pred) -> dict[str, object]:
    """Retained for future classification benchmarks."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
