"""Download TLC yellow-taxi files and aggregate them to a zone-hour panel with SQL."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from ride_pricing.config import (
    PANEL_END,
    PANEL_START,
    SEED,
    TLC_BASE_URL,
    TLC_MONTHS,
    TLC_ROWS,
    TLC_SHA256,
    USER_AGENT,
)


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: Path, pause_seconds: float = 0.0) -> None:
    """Stream a file to dest. A partial file is replaced only after the download finishes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    temporary.replace(dest)


def parquet_row_count(path: Path) -> int:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET threads TO 4")
        count = con.execute("SELECT count(*) FROM read_parquet(?)", [path.as_posix()]).fetchone()
    finally:
        con.close()
    if count is None:
        raise RuntimeError(f"could not count rows in {path}")
    return int(count[0])


def verify_file(path: Path, expected_hash: str, expected_rows: int) -> None:
    if len(expected_hash) != 64 or expected_rows <= 0:
        raise RuntimeError(
            f"checksum for {path.name} is not pinned; record sha256 and row count before a full run"
        )
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"sha256 mismatch for {path.name}: {actual_hash} != {expected_hash}")
    actual_rows = parquet_row_count(path)
    if actual_rows != expected_rows:
        raise RuntimeError(f"row-count mismatch for {path.name}: {actual_rows} != {expected_rows}")


def verify_tlc_file(path: Path, month: str) -> None:
    verify_file(path, TLC_SHA256.get(month, ""), int(TLC_ROWS.get(month, 0)))


def tlc_url(month: str) -> str:
    return f"{TLC_BASE_URL}/yellow_tripdata_{month}.parquet"


def ensure_tlc_files(cache_dir: Path) -> list[Path]:
    """Download the four monthly files if needed, then verify checksum and row count."""
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, month in enumerate(TLC_MONTHS):
        dest = raw_dir / f"yellow_tripdata_{month}.parquet"
        if not dest.exists():
            print(f"downloading {tlc_url(month)}", flush=True)
            download_file(tlc_url(month), dest, pause_seconds=3.0 if index else 0.0)
        else:
            print(f"using cached {dest.name}", flush=True)
        verify_tlc_file(dest, month)
        paths.append(dest)
    return paths


def write_synthetic_trips(dest: Path, seed: int = SEED) -> Path:
    """Small TLC-shaped parquet so the SQL path runs without a network download."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(PANEL_START, PANEL_END - pd.Timedelta(hours=1), freq="h")
    zones = np.arange(1, 7, dtype=np.int32)
    ts_grid = np.repeat(hours.to_numpy(), len(zones))
    zone_grid = np.tile(zones, len(hours))
    hour = pd.DatetimeIndex(ts_grid).hour.to_numpy()
    dow = pd.DatetimeIndex(ts_grid).dayofweek.to_numpy()
    rate = 3.0 + zone_grid.astype(float)
    rate = rate + np.isin(hour, [8, 9, 17, 18, 19]).astype(float) * 7.0
    rate = rate + (dow >= 5).astype(float) * -1.0
    rate = np.clip(rate, 0.5, None)
    counts = rng.poisson(rate)
    pickup = np.repeat(ts_grid, counts)
    zone = np.repeat(zone_grid, counts)
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(pickup),
            "PULocationID": zone.astype("int32"),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    return dest


def aggregate_zone_hour(
    parquet_paths: list[Path],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Aggregate trips to a complete zone-hour panel and add calendar fields in SQL.

    Only PULocationID and tpep_pickup_datetime are read. Zones outside 1-263 and
    pickups outside [start, end) are dropped. Missing zone-hours are filled with 0.
    """
    if not parquet_paths:
        raise ValueError("no parquet paths")
    start_ts = pd.Timestamp(PANEL_START if start is None else start)
    end_ts = pd.Timestamp(PANEL_END if end is None else end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")
    hours = pd.DataFrame(
        {"ts": pd.date_range(start_ts, end_ts - pd.Timedelta(hours=1), freq="h")}
    )
    file_list = ", ".join("'" + path.resolve().as_posix().replace("'", "''") + "'" for path in parquet_paths)
    start_sql = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    end_sql = end_ts.strftime("%Y-%m-%d %H:%M:%S")
    query = f"""
        WITH counts AS (
            SELECT
                CAST(PULocationID AS INTEGER) AS zone_id,
                date_trunc('hour', tpep_pickup_datetime) AS ts,
                COUNT(*)::INTEGER AS trips
            FROM read_parquet([{file_list}])
            WHERE tpep_pickup_datetime >= TIMESTAMP '{start_sql}'
              AND tpep_pickup_datetime < TIMESTAMP '{end_sql}'
              AND PULocationID BETWEEN 1 AND 263
            GROUP BY 1, 2
        ),
        zones AS (
            SELECT DISTINCT zone_id FROM counts
        )
        SELECT
            z.zone_id,
            h.ts,
            COALESCE(c.trips, 0)::INTEGER AS trips,
            EXTRACT(hour FROM h.ts)::INTEGER AS hour,
            EXTRACT(dow FROM h.ts)::INTEGER AS dow,
            EXTRACT(month FROM h.ts)::INTEGER AS month,
            CASE WHEN EXTRACT(dow FROM h.ts) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
        FROM zones z
        CROSS JOIN hours h
        LEFT JOIN counts c
            ON c.zone_id = z.zone_id AND c.ts = h.ts
        ORDER BY z.zone_id, h.ts
    """
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET threads TO 4")
        con.execute("SET memory_limit = '1500MB'")
        if temp_dir is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = '{temp_dir.resolve().as_posix()}'")
        con.register("hours", hours)
        frame = con.execute(query).fetchdf()
    finally:
        con.close()
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["zone_id"] = frame["zone_id"].astype("int16")
    frame["trips"] = frame["trips"].astype("int32")
    frame["hour"] = frame["hour"].astype("int8")
    frame["dow"] = frame["dow"].astype("int8")
    frame["month"] = frame["month"].astype("int8")
    frame["is_weekend"] = frame["is_weekend"].astype("int8")
    return frame
