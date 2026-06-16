"""Executable synthetic benchmark combining RF, kriging, and hybrid residual modeling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from .synthetic_spatial_analysis import compute_pca_projection, permutation_importance_table, evaluate_regression_models
from .synthetic_spatial_data import DEFAULT_RF_FEATURES, build_spatial_blocks, simulate_dataset, split_train_test
from .synthetic_spatial_metrics import expm1_safe, log1p_safe, metrics_dict, rmse
from .synthetic_spatial_models import RFOKHybridRegressor, fit_ordinary_kriging, fit_random_forest, predict_ordinary_kriging
from ..visualization.synthetic_spatial_plots import plot_feature_importance, plot_map, plot_pca_projection, plot_scatter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyntheticSpatialCaseStudyConfig:
    output_dir: Path = Path("synthetic_case_study_outputs")
    random_state: int = 42
    n_points: int = 600
    n_blocks: int = 8
    test_size: float = 0.25
    rf_features: list[str] = field(default_factory=lambda: list(DEFAULT_RF_FEATURES))


def _prepare_data(config: SyntheticSpatialCaseStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = simulate_dataset(n_points=config.n_points, seed=config.random_state)
    df["block"] = build_spatial_blocks(df, n_blocks=config.n_blocks, seed=config.random_state)
    tr_idx, te_idx = split_train_test(df, df["block"], test_size=config.test_size, seed=config.random_state)
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    return df, train, test


def _save_outputs(
    config: SyntheticSpatialCaseStudyConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    rf_test_pred,
    ok_test_pred,
    hybrid_test_pred,
    feature_importance: pd.DataFrame,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    out = test.copy()
    out["pred_rf"] = rf_test_pred
    out["pred_ok"] = ok_test_pred
    out["pred_hybrid"] = hybrid_test_pred
    out["residual_hybrid"] = out["rainfall_station"] - out["pred_hybrid"]
    pred_path = config.output_dir / "test_predictions.csv"
    out.to_csv(pred_path, index=False)

    fi_path = config.output_dir / "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)

    plot_scatter(test["rainfall_station"], rf_test_pred, "Observed vs Predicted - Random Forest", config.output_dir / "obs_vs_pred_rf.png")
    plot_scatter(test["rainfall_station"], ok_test_pred, "Observed vs Predicted - Ordinary Kriging", config.output_dir / "obs_vs_pred_ok.png")
    plot_scatter(test["rainfall_station"], hybrid_test_pred, "Observed vs Predicted - RF + OK residuals", config.output_dir / "obs_vs_pred_hybrid.png")
    plot_map(test, test["rainfall_station"].values - hybrid_test_pred, "Hybrid residual map", config.output_dir / "hybrid_residual_map.png")

    projection = compute_pca_projection(train[config.rf_features], train["rainfall_station"], label_name="rainfall_station")
    plot_pca_projection(projection, "rainfall_station", "PCA of synthetic covariates", config.output_dir / "pca_projection.png")

    plot_feature_importance(feature_importance, "Top predictors in Random Forest", config.output_dir / "feature_importance.png")

    return {
        "predictions": pred_path,
        "feature_importance": fi_path,
    }


def run_case_study(config: SyntheticSpatialCaseStudyConfig | None = None) -> dict[str, object]:
    """Run the full benchmark and persist outputs."""
    cfg = config or SyntheticSpatialCaseStudyConfig()
    t0 = time.time()

    logger.info("Generating synthetic dataset")
    df, train, test = _prepare_data(cfg)

    logger.info("Training Random Forest")
    rf = fit_random_forest(train[cfg.rf_features], train["rainfall_station"], train["block"], random_state=cfg.random_state)
    rf_test_pred = np.clip(rf.predict(test[cfg.rf_features]), 0, None)

    logger.info("Training Ordinary Kriging")
    ok = fit_ordinary_kriging(train["x"].to_numpy(), train["y"].to_numpy(), log1p_safe(train["rainfall_station"]))
    ok_test_log, ok_test_var = predict_ordinary_kriging(ok, test["x"].to_numpy(), test["y"].to_numpy())
    ok_test_pred = expm1_safe(ok_test_log)

    logger.info("Training hybrid model")
    train_h = train[cfg.rf_features + ["x", "y", "block"]].copy()
    test_h = test[cfg.rf_features + ["x", "y", "block"]].copy()
    train_h["__group__"] = train["block"].values
    test_h["__group__"] = test["block"].values

    hybrid = RFOKHybridRegressor(
        rf_model=clone(rf),
        rf_features=cfg.rf_features,
    )
    hybrid.fit(train_h, train["rainfall_station"])
    hybrid_test_pred = hybrid.predict(test_h)

    comparison = evaluate_regression_models(
        test["rainfall_station"],
        {
            "RandomForest": rf_test_pred,
            "OrdinaryKriging": ok_test_pred,
            "RF + OK residuals": hybrid_test_pred,
        },
    )

    feature_importance = permutation_importance_table(
        rf,
        test[cfg.rf_features],
        test["rainfall_station"],
        cfg.rf_features,
        random_state=cfg.random_state,
    )

    saved_paths = _save_outputs(
        cfg,
        train,
        test,
        rf_test_pred,
        ok_test_pred,
        hybrid_test_pred,
        feature_importance,
    )

    comparison_path = cfg.output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n=== Synthetic spatial case study ===\n")
    print(f"Rows: {len(df)}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))
    logger.info("Total runtime: %.1f seconds", time.time() - t0)

    return {
        "config": cfg,
        "data": df,
        "train": train,
        "test": test,
        "comparison": comparison,
        "saved_paths": saved_paths,
        "rf": rf,
        "ok": ok,
        "hybrid": hybrid,
        "ok_test_variance": ok_test_var,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_case_study()


if __name__ == "__main__":
    main()
