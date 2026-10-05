import numpy as np
import pandas as pd
import pytest

from ride_pricing.features import add_lag_features, assert_hourly_grid, assign_split


def _hourly(trips: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(trips), freq="h")
    return pd.DataFrame(
        {
            "zone_id": np.int16(1),
            "ts": ts,
            "trips": np.array(trips, dtype=np.int32),
            "hour": ts.hour.astype("int8"),
            "dow": ts.dayofweek.astype("int8"),
            "month": ts.month.astype("int8"),
            "is_weekend": (ts.dayofweek >= 5).astype("int8"),
        }
    )


def test_lags_use_only_earlier_hours():
    trips = list(range(1, 31))
    out = add_lag_features(_hourly(trips))
    # Position 10 (value 11) must see the previous value 10 and the one before that, 9.
    assert out.iloc[10]["lag_1"] == 10
    assert out.iloc[10]["lag_2"] == 9
    assert out.iloc[10]["lag_3"] == 8
    assert np.isnan(out.iloc[10]["lag_24"])
    # roll_mean_24 at the last row is the mean of the previous 24 trips, excluding the current row.
    assert out.iloc[29]["roll_mean_24"] == pytest.approx(float(np.mean(trips[5:29])))
    assert out.iloc[29]["lag_1"] == trips[28]


def test_gap_in_panel_is_rejected():
    frame = _hourly([1, 2, 3, 4])
    frame.loc[2, "ts"] = frame.loc[2, "ts"] + pd.Timedelta(hours=5)
    with pytest.raises(ValueError):
        assert_hourly_grid(frame)


def test_splits_do_not_overlap_and_test_starts_in_april():
    ts = pd.date_range("2024-01-01", "2024-04-30 23:00", freq="h")
    labels = assign_split(pd.Series(ts))
    assert set(labels.unique()) <= {"history", "train", "val", "test"}
    train_ts = set(ts[labels.to_numpy() == "train"])
    val_ts = set(ts[labels.to_numpy() == "val"])
    test_ts = set(ts[labels.to_numpy() == "test"])
    assert train_ts.isdisjoint(val_ts)
    assert train_ts.isdisjoint(test_ts)
    assert val_ts.isdisjoint(test_ts)
    assert min(test_ts) == pd.Timestamp("2024-04-01")
    assert max(train_ts) < pd.Timestamp("2024-03-18")
    assert max(val_ts) < pd.Timestamp("2024-04-01")
    assert min(train_ts) == pd.Timestamp("2024-01-29")
