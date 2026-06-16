"""Metrics and target transforms for the synthetic spatial benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}RMSE": rmse(y_true, y_pred),
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}R2": float(r2_score(y_true, y_pred)),
    }


def log1p_safe(y):
    y = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y, 0, None))


def expm1_safe(y):
    y = np.asarray(y, dtype=float)
    return np.clip(np.expm1(y), 0, None)
