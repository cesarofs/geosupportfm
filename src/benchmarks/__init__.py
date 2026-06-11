"""Benchmark utilities for GeoSupportFM."""

from .synthetic_spatial_case_study import SyntheticSpatialCaseStudyConfig, run_case_study
from .synthetic_spatial_data import (
    DEFAULT_RF_FEATURES,
    build_spatial_blocks,
    exponential_covariance,
    simulate_dataset,
    split_train_test,
)
from .synthetic_spatial_metrics import expm1_safe, log1p_safe, metrics_dict, rmse
from .synthetic_spatial_models import (
    RFOKHybridRegressor,
    fit_ordinary_kriging,
    fit_random_forest,
    oof_rf_predictions,
    predict_ordinary_kriging,
)
