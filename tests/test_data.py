import pandas as pd
import pytest

from ride_pricing.data import (
    aggregate_zone_hour,
    file_sha256,
    parquet_row_count,
    tlc_url,
    verify_file,
    write_synthetic_trips,
)


def test_sha256_and_row_count_round_trip(tmp_path):
    path = write_synthetic_trips(tmp_path / "trips.parquet", seed=1)
    digest = file_sha256(path)
    assert len(digest) == 64
    assert parquet_row_count(path) > 0
    verify_file(path, digest, parquet_row_count(path))
    with pytest.raises(RuntimeError):
        verify_file(path, "0" * 64, 10)


def test_aggregation_counts_filters_and_calendar(tmp_path):
    rows = [
        ("2024-01-01 00:10:00", 1),
        ("2024-01-01 00:40:00", 1),
        ("2024-01-01 01:05:00", 1),
        ("2024-01-01 03:00:00", 1),  # outside the half-open window
        ("2024-01-01 02:15:00", 999),  # outside zone range
        ("2024-01-01 02:15:00", 2),
    ]
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime([r[0] for r in rows]),
            "PULocationID": [r[1] for r in rows],
        }
    )
    path = tmp_path / "tiny.parquet"
    frame.to_parquet(path, index=False)
    panel = aggregate_zone_hour(
        [path],
        start=pd.Timestamp("2024-01-01 00:00:00"),
        end=pd.Timestamp("2024-01-01 03:00:00"),
        temp_dir=tmp_path / "tmp",
    )
    panel = panel.sort_values(["zone_id", "ts"]).reset_index(drop=True)
    assert list(panel["zone_id"].unique()) == [1, 2]
    z1 = panel.loc[panel["zone_id"] == 1, "trips"].tolist()
    z2 = panel.loc[panel["zone_id"] == 2, "trips"].tolist()
    assert z1 == [2, 1, 0]
    assert z2 == [0, 0, 1]
    assert panel["hour"].tolist() == [0, 1, 2, 0, 1, 2]
    # 2024-01-01 was a Monday, so the weekend flag is off.
    assert set(panel["is_weekend"].tolist()) == {0}
    assert int(panel["trips"].sum()) == 4


def test_tlc_url_points_at_the_public_parquet():
    url = tlc_url("2024-01")
    assert url.endswith("yellow_tripdata_2024-01.parquet")
    assert url.startswith("https://")
