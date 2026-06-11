"""Model fitting utilities for the synthetic spatial benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from .synthetic_spatial_metrics import expm1_safe, log1p_safe


def _load_ordinary_kriging_class():
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ImportError("Install pykrige first: pip install pykrige") from exc
    return OrdinaryKriging


def _group_kfold(groups, max_splits: int = 5) -> GroupKFold:
    groups_arr = np.asarray(groups)
    n_groups = len(np.unique(groups_arr))
    if n_groups < 2:
        raise ValueError("Need at least two spatial groups for grouped cross-validation.")
    return GroupKFold(n_splits=min(max_splits, n_groups))


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train,
    groups_train,
    random_state: int = 42,
    n_iter: int = 12,
    max_splits: int = 5,
):
    """Fit a tuned Random Forest wrapped in a target transform."""
    base = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model = TransformedTargetRegressor(
        regressor=base,
        func=log1p_safe,
        inverse_func=expm1_safe,
    )
    search = RandomizedSearchCV(
        model,
        param_distributions={
            "regressor__n_estimators": [300, 500, 800],
            "regressor__max_depth": [None, 8, 12, 16, 24],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__min_samples_split": [2, 4, 8],
            "regressor__max_features": ["sqrt", 0.5, 0.8],
        },
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=_group_kfold(groups_train, max_splits=max_splits),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_


def fit_ordinary_kriging(x_train, y_train, z_train, variogram_model: str = "spherical"):
    """Fit an Ordinary Kriging model with lazy pykrige import."""
    OrdinaryKriging = _load_ordinary_kriging_class()
    return OrdinaryKriging(
        x_train,
        y_train,
        z_train,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        coordinates_type="euclidean",
    )


def predict_ordinary_kriging(ok_model, x, y):
    """Predict values and kriging variance at points."""
    z, v = ok_model.execute("points", x, y)
    return np.asarray(z).ravel(), np.asarray(v).ravel()


def oof_rf_predictions(
    rf_model,
    X: pd.DataFrame,
    y,
    groups,
    rf_features: list[str],
    max_splits: int = 5,
) -> np.ndarray:
    """Compute out-of-fold RF predictions for residual kriging."""
    y_arr = np.asarray(y, dtype=float)
    oof = np.zeros(len(X), dtype=float)
    cv = _group_kfold(groups, max_splits=max_splits)

    for tr_idx, va_idx in cv.split(X, y_arr, groups=np.asarray(groups)):
        fold_model = clone(rf_model)
        fold_model.fit(X.iloc[tr_idx][rf_features], y_arr[tr_idx])
        oof[va_idx] = fold_model.predict(X.iloc[va_idx][rf_features])

    return oof


@dataclass
class RFOKHybridRegressor(BaseEstimator, RegressorMixin):
    """Random Forest plus Ordinary Kriging on out-of-fold residuals."""

    rf_model: object
    rf_features: list[str]
    x_col: str = "x"
    y_col: str = "y"
    group_col: str = "__group__"
    max_splits: int = 5

    def fit(self, X: pd.DataFrame, y):
        X = X.copy()
        y_arr = np.asarray(y, dtype=float)

        self.rf_model_ = clone(self.rf_model)
        self.rf_model_.fit(X[self.rf_features], y_arr)

        groups = X[self.group_col].to_numpy()
        oof_rf = oof_rf_predictions(
            self.rf_model_,
            X,
            y_arr,
            groups,
            self.rf_features,
            max_splits=self.max_splits,
        )
        residuals = y_arr - oof_rf

        OrdinaryKriging = _load_ordinary_kriging_class()
        self.ok_resid_ = OrdinaryKriging(
            X[self.x_col].to_numpy(dtype=float),
            X[self.y_col].to_numpy(dtype=float),
            residuals,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
            coordinates_type="euclidean",
        )
        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        rf_pred = np.asarray(self.rf_model_.predict(X[self.rf_features]), dtype=float)
        resid_pred, _ = self.ok_resid_.execute(
            "points",
            X[self.x_col].to_numpy(dtype=float),
            X[self.y_col].to_numpy(dtype=float),
        )
        return np.clip(rf_pred + np.asarray(resid_pred).ravel(), 0, None)
