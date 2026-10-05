"""Seasonal-naive baseline and a global LightGBM one-step forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import Dataset, early_stopping, log_evaluation, train

from ride_pricing.config import (
    LGB_EARLY_STOPPING_ROUNDS,
    LGB_LEARNING_RATE,
    LGB_MIN_CHILD_SAMPLES,
    LGB_N_ESTIMATORS,
    LGB_N_JOBS,
    LGB_NUM_LEAVES,
    LGB_REG_LAMBDA,
    SEED,
)
from ride_pricing.features import feature_matrix


def seasonal_naive(lag_168: np.ndarray) -> np.ndarray:
    """Same zone, same hour, one week earlier. Caller must pass an already shifted lag."""
    pred = np.asarray(lag_168, dtype=float)
    if np.isnan(pred).any():
        raise ValueError("seasonal naive lag contains missing values")
    return pred


def _params() -> dict:
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": LGB_LEARNING_RATE,
        "num_leaves": LGB_NUM_LEAVES,
        "min_child_samples": LGB_MIN_CHILD_SAMPLES,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l2": LGB_REG_LAMBDA,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": LGB_N_JOBS,
        "verbosity": -1,
        "verbose": -1,
    }


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_estimators: int = LGB_N_ESTIMATORS,
    early_stopping_rounds: int = LGB_EARLY_STOPPING_ROUNDS,
):
    """Fit on train. Early stopping uses validation rows only."""
    if train_df.empty or val_df.empty:
        raise ValueError("train and validation frames must be non-empty")
    train_x = feature_matrix(train_df)
    val_x = feature_matrix(val_df)
    train_set = Dataset(
        train_x,
        label=train_df["trips"].to_numpy(dtype=np.float32),
        categorical_feature=["zone_id"],
        free_raw_data=False,
    )
    val_set = Dataset(
        val_x,
        label=val_df["trips"].to_numpy(dtype=np.float32),
        categorical_feature=["zone_id"],
        reference=train_set,
        free_raw_data=False,
    )
    return train(
        _params(),
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            early_stopping(early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )


def predict_nonnegative(model, df: pd.DataFrame) -> np.ndarray:
    iteration = getattr(model, "best_iteration", 0) or None
    raw = model.predict(feature_matrix(df), num_iteration=iteration)
    return np.clip(np.asarray(raw, dtype=float), 0.0, None)
