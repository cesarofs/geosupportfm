"""Plotting helpers for variography diagnostics."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_variogram_curve(frame: pd.DataFrame, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["lag_center"], frame["semivariance"], marker="o")
    ax.set_xlabel("Lag distance")
    ax.set_ylabel("Semivariance")
    ax.set_title(title or "Empirical variogram")
    return ax


def plot_cross_variogram(frame: pd.DataFrame, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["lag_center"], frame["semivariance"], marker="o")
    ax.set_xlabel("Lag distance")
    ax.set_ylabel("Cross-semivariance")
    ax.set_title(title or "Empirical cross-variogram")
    return ax


def plot_dimension_rank(report: pd.DataFrame, top_n: int = 20, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    frame = report.head(top_n).iloc[::-1]
    ax.barh(frame["name"], frame["structure_score"])
    ax.set_xlabel("Structure score")
    ax.set_ylabel("Latent dimension")
    ax.set_title(title or "Latent dimensions ranked by spatial structure")
    return ax
