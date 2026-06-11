"""Plotting helpers for synthetic spatial benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_scatter(y_true, y_pred, title: str, path: str | Path) -> Path:
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.35)
    lo = min(float(min(y_true)), float(min(y_pred)))
    hi = max(float(max(y_true)), float(max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_map(df: pd.DataFrame, values, title: str, path: str | Path) -> Path:
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sc = ax.scatter(df["x"], df["y"], c=values, cmap="coolwarm", s=34, alpha=0.85)
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_feature_importance(importance_df: pd.DataFrame, title: str, path: str | Path, top_n: int = 12) -> Path:
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    top = importance_df.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Permutation importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_pca_projection(
    projection_df: pd.DataFrame,
    label_column: str,
    title: str,
    path: str | Path,
    max_classes: int = 20,
) -> Path:
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = sorted(projection_df[label_column].unique())[:max_classes]
    for cls in classes:
        sub = projection_df[projection_df[label_column] == cls]
        ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.35, label=str(cls))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
