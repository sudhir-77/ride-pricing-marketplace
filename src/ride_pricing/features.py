"""Lag features that use only trips from earlier hours."""

from __future__ import annotations

import pandas as pd

from ride_pricing.config import FEATURE_COLUMNS, MAX_LAG_HOURS, TEST_START, TRAIN_START, VAL_START


def assert_hourly_grid(df: pd.DataFrame) -> None:
    """Each zone must be a gap-free hourly series so shift-based lags stay aligned."""
    if df.empty:
        raise ValueError("panel is empty")
    ordered = df.sort_values(["zone_id", "ts"])
    diffs = ordered.groupby("zone_id", sort=False)["ts"].diff().dropna()
    step = pd.Timedelta(hours=1)
    if not (diffs == step).all():
        raise ValueError("panel is not a complete hourly grid; lag features would leak across gaps")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags and a trailing 24-hour mean. Row t never sees trips at t or later."""
    assert_hourly_grid(df)
    out = df.sort_values(["zone_id", "ts"]).reset_index(drop=True).copy()
    grouped = out.groupby("zone_id", sort=False)["trips"]
    out["lag_1"] = grouped.shift(1)
    out["lag_2"] = grouped.shift(2)
    out["lag_3"] = grouped.shift(3)
    out["lag_24"] = grouped.shift(24)
    out["lag_168"] = grouped.shift(168)
    out["lag_336"] = grouped.shift(336)
    out["lag_672"] = grouped.shift(672)
    out["roll_mean_24"] = grouped.transform(lambda s: s.shift(1).rolling(24, min_periods=24).mean())
    return out


def supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that do not yet have the longest lag. Those hours stay in the panel as history."""
    needed = ["lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "lag_336", "lag_672", "roll_mean_24"]
    return df.dropna(subset=needed).reset_index(drop=True)


def assign_split(ts: pd.Series) -> pd.Series:
    """Label rows history / train / val / test. Boundaries are fixed calendar times."""
    stamps = pd.to_datetime(ts)
    labels = pd.Series("history", index=stamps.index, dtype="object")
    labels.loc[stamps >= pd.Timestamp(TRAIN_START)] = "train"
    labels.loc[stamps >= pd.Timestamp(VAL_START)] = "val"
    labels.loc[stamps >= pd.Timestamp(TEST_START)] = "test"
    return labels


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    frame = df.loc[:, cols].copy()
    frame["zone_id"] = frame["zone_id"].astype("int32")
    for col in cols:
        if col != "zone_id":
            frame[col] = frame[col].astype("float32")
    return frame


def max_lag_hours() -> int:
    return MAX_LAG_HOURS
