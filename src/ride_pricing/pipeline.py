"""End-to-end run: panel, forecast, surge search, switchback coverage."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from ride_pricing.config import (
    BASE_FARE_USD,
    COVERAGE_MIN,
    LGB_EARLY_STOPPING_ROUNDS,
    LGB_EARLY_STOPPING_ROUNDS_QUICK,
    LGB_N_ESTIMATORS,
    LGB_N_ESTIMATORS_QUICK,
    N_BLOCKS,
    N_BLOCKS_QUICK,
    N_SWITCHBACK_EXPERIMENTS,
    N_SWITCHBACK_EXPERIMENTS_QUICK,
    N_UNITS_PER_BLOCK,
    N_UNITS_PER_BLOCK_QUICK,
    OFFPEAK_SUPPLY_RATIO,
    PEAK_SUPPLY_RATIO,
    PROJECT_SLUG,
    RIDER_ELASTICITY,
    SEED,
    SUPPLY_ELASTICITY,
    TEST_START,
    TOP_N_ZONES,
    TRAIN_START,
    TREATMENT_MULTIPLIER,
    TRIPS_LIFT_MIN,
    VAL_START,
    WAPE_IMPROVEMENT_MIN,
    WAPE_MAX,
)
from ride_pricing.data import aggregate_zone_hour, ensure_tlc_files, write_synthetic_trips
from ride_pricing.experiments import run_switchback_coverage, switchback_bias
from ride_pricing.features import add_lag_features, assign_split, supervised_frame
from ride_pricing.forecast import predict_nonnegative, seasonal_naive, train_lightgbm
from ride_pricing.metrics import (
    assert_finite,
    make_criterion,
    relative_improvement,
    wape,
    weighted_average,
)
from ride_pricing.optimize import optimize_surge
from ride_pricing.reporting import (
    FORECAST_CSV,
    NAIVE_CSV,
    SURGE_CSV,
    SWITCHBACK_CSV,
    TOP_ZONES_CSV,
    plot_forecast_example,
    plot_multipliers,
    plot_switchback,
    plot_wape,
    render_resume_notes,
    results_markdown,
    update_readme,
    write_frame,
    write_json,
)
from ride_pricing.simulator import build_market_frame, user_level_and_market_effects

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PeakMemory:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="peak-rss", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            rss = self._proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        info = self._proc.memory_info()
        peak_wset = int(getattr(info, "peak_wset", 0) or 0)
        self.peak = max(self.peak, peak_wset, info.rss)
        return self.peak / (1024**3)


def _row_id(zone_id: int, ts: pd.Timestamp) -> str:
    return f"{int(zone_id)}|{pd.Timestamp(ts).isoformat()}"


def _hardware(peak_rss_gb: float) -> dict:
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(float(peak_rss_gb), 4),
    }


def run_pipeline(
    *,
    quick: bool = False,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
    update_docs: bool = True,
    project_root: Path | None = None,
) -> dict:
    """Reproduce every artifact. `quick` uses a synthetic panel and fewer simulations."""
    np.random.seed(SEED)
    started = time.perf_counter()
    tracker = _PeakMemory()
    root = PROJECT_ROOT if project_root is None else project_root
    out = (root / "results") if output_dir is None else Path(output_dir)
    cache = (root / "data") if cache_dir is None else Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        print("CUDA was requested. This project trains on CPU only.", flush=True)

    if quick:
        print("building synthetic trip file for quick mode", flush=True)
        parquet_paths = [write_synthetic_trips(cache / "quick" / "synthetic_trips.parquet")]
    else:
        parquet_paths = ensure_tlc_files(cache)

    print("aggregating zone-hour demand", flush=True)
    panel = aggregate_zone_hour(parquet_paths, temp_dir=cache / "tmp")
    print(f"panel rows={len(panel)} zones={panel['zone_id'].nunique()} trips={int(panel['trips'].sum())}", flush=True)

    featured = supervised_frame(add_lag_features(panel))
    featured["split"] = assign_split(featured["ts"])
    if featured.duplicated(["zone_id", "ts"]).any():
        raise RuntimeError("duplicate zone-hour rows")
    train = featured.loc[featured["split"] == "train"].copy()
    val = featured.loc[featured["split"] == "val"].copy()
    test = featured.loc[featured["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise RuntimeError("a split is empty; check the calendar bounds and the panel")
    if train["ts"].max() >= pd.Timestamp(VAL_START) or val["ts"].max() >= pd.Timestamp(TEST_START):
        raise RuntimeError("split boundaries overlap")
    if test["ts"].min() < pd.Timestamp(TEST_START):
        raise RuntimeError("test split contains rows before April")

    zone_totals = train.groupby("zone_id", sort=False)["trips"].sum().sort_values(ascending=False)
    n_top = min(TOP_N_ZONES, int(zone_totals.shape[0]))
    if not quick and n_top < TOP_N_ZONES:
        raise RuntimeError(f"expected at least {TOP_N_ZONES} zones with training trips, found {n_top}")
    top_zones = zone_totals.head(n_top)
    top_ids = set(int(z) for z in top_zones.index.tolist())

    manifest = {
        "train_start": pd.Timestamp(TRAIN_START).isoformat(),
        "val_start": pd.Timestamp(VAL_START).isoformat(),
        "test_start": pd.Timestamp(TEST_START).isoformat(),
        "top_zones": [int(z) for z in top_zones.index.tolist()],
        "quick": quick,
    }
    manifest_path = cache / "splits" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("training LightGBM (validation is March only)", flush=True)
    n_estimators = LGB_N_ESTIMATORS_QUICK if quick else LGB_N_ESTIMATORS
    stopping = LGB_EARLY_STOPPING_ROUNDS_QUICK if quick else LGB_EARLY_STOPPING_ROUNDS
    model = train_lightgbm(train, val, n_estimators=n_estimators, early_stopping_rounds=stopping)
    best_iteration = int(getattr(model, "best_iteration", 0) or n_estimators)

    val_pred = predict_nonnegative(model, val)
    test_pred = predict_nonnegative(model, test)
    val_naive = seasonal_naive(val["lag_168"].to_numpy())
    test_naive = seasonal_naive(test["lag_168"].to_numpy())

    test_out = test.loc[:, ["zone_id", "ts", "trips"]].copy()
    test_out["pred_lightgbm"] = test_pred
    test_out["pred_seasonal_naive"] = test_naive
    test_top = test_out.loc[test_out["zone_id"].astype(int).isin(top_ids)].copy()
    val_top_mask = val["zone_id"].astype(int).isin(top_ids).to_numpy()
    wape_model = wape(test_top["trips"].to_numpy(), test_top["pred_lightgbm"].to_numpy())
    wape_naive = wape(test_top["trips"].to_numpy(), test_top["pred_seasonal_naive"].to_numpy())
    improvement = relative_improvement(wape_naive, wape_model)
    wape_model_val = wape(val.loc[val_top_mask, "trips"].to_numpy(), val_pred[val_top_mask])
    wape_naive_val = wape(val.loc[val_top_mask, "trips"].to_numpy(), val_naive[val_top_mask])

    forecast_csv = pd.DataFrame(
        {
            "row_id": [_row_id(z, t) for z, t in zip(test_top["zone_id"], test_top["ts"])],
            "zone_id": test_top["zone_id"].astype(int).to_numpy(),
            "ts": pd.to_datetime(test_top["ts"]).map(lambda t: pd.Timestamp(t).isoformat()),
            "actual": test_top["trips"].to_numpy(dtype=float),
            "pred_lightgbm": test_top["pred_lightgbm"].to_numpy(dtype=float),
            "pred_seasonal_naive": test_top["pred_seasonal_naive"].to_numpy(dtype=float),
        }
    )
    # Metrics are computed from the same columns that are written to disk.
    wape_model = wape(forecast_csv["actual"].to_numpy(), forecast_csv["pred_lightgbm"].to_numpy())
    wape_naive = wape(forecast_csv["actual"].to_numpy(), forecast_csv["pred_seasonal_naive"].to_numpy())
    improvement = relative_improvement(wape_naive, wape_model)

    print("optimizing surge multipliers", flush=True)
    market_input = pd.DataFrame(
        {
            "zone_id": forecast_csv["zone_id"].to_numpy(),
            "ts": pd.to_datetime(forecast_csv["ts"]),
            "base_demand": forecast_csv["pred_lightgbm"].to_numpy(dtype=float),
        }
    )
    market = build_market_frame(market_input)
    solution = optimize_surge(market["base_demand"].to_numpy(), market["base_supply"].to_numpy())
    surge_csv = pd.DataFrame(
        {
            "row_id": forecast_csv["row_id"].to_numpy(),
            "zone_id": market["zone_id"].astype(int).to_numpy(),
            "ts": forecast_csv["ts"].to_numpy(),
            "base_demand": market["base_demand"].to_numpy(dtype=float),
            "base_supply": market["base_supply"].to_numpy(dtype=float),
            "supply_ratio": market["supply_ratio"].to_numpy(dtype=float),
            "multiplier": np.asarray(solution["multiplier"], dtype=float),
            "completed_optimized": np.asarray(solution["completed_optimized"], dtype=float),
            "completed_flat": np.asarray(solution["completed_flat"], dtype=float),
            "rider_price_optimized": np.asarray(solution["rider_price_optimized"], dtype=float),
            "rider_price_flat": np.asarray(solution["rider_price_flat"], dtype=float),
        }
    )
    trips_opt = float(surge_csv["completed_optimized"].sum())
    trips_flat = float(surge_csv["completed_flat"].sum())
    if trips_flat <= 0:
        raise RuntimeError("flat pricing produced no completed trips")
    trips_lift = trips_opt / trips_flat - 1.0
    price_opt = weighted_average(
        surge_csv["rider_price_optimized"].to_numpy(),
        surge_csv["completed_optimized"].to_numpy(),
    )
    price_flat = weighted_average(
        surge_csv["rider_price_flat"].to_numpy(),
        surge_csv["completed_flat"].to_numpy(),
    )

    print("simulating switchback experiments", flush=True)
    naive_estimate, true_cell = user_level_and_market_effects(
        surge_csv["base_demand"].to_numpy(),
        surge_csv["base_supply"].to_numpy(),
        TREATMENT_MULTIPLIER,
    )
    naive_csv = pd.DataFrame(
        {
            "row_id": surge_csv["row_id"].to_numpy(),
            "zone_id": surge_csv["zone_id"].to_numpy(),
            "ts": surge_csv["ts"].to_numpy(),
            "naive_estimate": naive_estimate,
            "true_effect": true_cell,
        }
    )
    true_effect = float(naive_csv["true_effect"].mean())
    naive_bias = float(abs(naive_csv["naive_estimate"].mean() - true_effect))
    n_experiments = N_SWITCHBACK_EXPERIMENTS_QUICK if quick else N_SWITCHBACK_EXPERIMENTS
    n_blocks = N_BLOCKS_QUICK if quick else N_BLOCKS
    n_units = N_UNITS_PER_BLOCK_QUICK if quick else N_UNITS_PER_BLOCK
    switch_df = run_switchback_coverage(
        true_effect=true_effect,
        n_experiments=n_experiments,
        n_blocks=n_blocks,
        n_units=n_units,
        seed=SEED,
    )
    coverage = float(switch_df["covered"].mean())
    sb_bias = switchback_bias(switch_df["estimate"].to_numpy(), true_effect)
    if naive_bias <= 0:
        raise RuntimeError("naive A/B bias is zero; the interference comparison is not informative")
    bias_ratio = sb_bias / naive_bias

    top_csv = pd.DataFrame(
        {
            "row_id": [str(int(z)) for z in top_zones.index.tolist()],
            "zone_id": [int(z) for z in top_zones.index.tolist()],
            "train_trips": [float(v) for v in top_zones.tolist()],
            "rank": list(range(1, n_top + 1)),
        }
    )

    criteria = [
        make_criterion(
            "April zone-hour demand WAPE (top 50 zones)",
            wape_model,
            "<=",
            WAPE_MAX,
        ),
        make_criterion(
            "April zone-hour demand WAPE (top 50 zones) improvement vs seasonal-naive",
            improvement,
            ">=",
            WAPE_IMPROVEMENT_MIN,
        ),
        make_criterion(
            "Completed trips under optimized surge vs. flat pricing",
            trips_lift,
            ">=",
            TRIPS_LIFT_MIN,
        ),
        make_criterion(
            "Average rider price under optimized surge vs flat pricing",
            price_opt,
            "<=",
            price_flat,
        ),
        make_criterion(
            "Switchback analysis 95% CI coverage of the true effect",
            coverage,
            ">=",
            COVERAGE_MIN,
        ),
        make_criterion(
            "Switchback bias vs one third of naive user-level A/B bias",
            sb_bias,
            "<",
            naive_bias / 3.0,
        ),
    ]
    mean_multiplier = float(surge_csv["multiplier"].mean()) if trips_opt <= 0 else float(
        np.average(surge_csv["multiplier"].to_numpy(), weights=surge_csv["completed_optimized"].to_numpy())
    )
    payload = {
        "project": PROJECT_SLUG,
        "success_criteria": criteria,
        "metrics": {
            "forecast": {
                "april_top50_wape_lightgbm": wape_model,
                "april_top50_wape_seasonal_naive": wape_naive,
                "wape_improvement_vs_naive": improvement,
                "val_wape_lightgbm": wape_model_val,
                "val_wape_seasonal_naive": wape_naive_val,
                "n_top_zones": n_top,
                "best_iteration": best_iteration,
            },
            "simulator": {
                "completed_trips_optimized": trips_opt,
                "completed_trips_flat": trips_flat,
                "trips_lift": trips_lift,
                "avg_rider_price_optimized": price_opt,
                "avg_rider_price_flat": price_flat,
                "mean_multiplier": mean_multiplier,
                "rider_elasticity": RIDER_ELASTICITY,
                "supply_elasticity": SUPPLY_ELASTICITY,
                "base_fare_usd": BASE_FARE_USD,
                "peak_supply_ratio": PEAK_SUPPLY_RATIO,
                "offpeak_supply_ratio": OFFPEAK_SUPPLY_RATIO,
            },
            "switchback": {
                "ci_coverage": coverage,
                "n_experiments": n_experiments,
                "n_blocks": n_blocks,
                "n_units_per_block": n_units,
                "true_effect": true_effect,
                "mean_estimate": float(switch_df["estimate"].mean()),
                "switchback_bias": sb_bias,
                "naive_ab_bias": naive_bias,
                "bias_ratio": bias_ratio,
                "treatment_multiplier": TREATMENT_MULTIPLIER,
            },
        },
        "baseline": {
            "april_top50_wape_seasonal_naive": wape_naive,
            "completed_trips_flat": trips_flat,
            "avg_rider_price_flat": price_flat,
            "flat_multiplier": 1.0,
        },
        "splits": {
            "zone_hour_panel": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            }
        },
        "seeds": [SEED],
        "hardware": _hardware(0.0),
        "runtime_seconds": 0.0,
    }
    write_frame(out / FORECAST_CSV, forecast_csv)
    write_frame(out / TOP_ZONES_CSV, top_csv)
    write_frame(out / SURGE_CSV, surge_csv)
    write_frame(out / SWITCHBACK_CSV, switch_df)
    write_frame(out / NAIVE_CSV, naive_csv)
    plot_wape(out / "wape_comparison.png", wape_model, wape_naive)
    example_zone = int(test_top.groupby("zone_id")["trips"].sum().idxmax())
    example = forecast_csv.loc[forecast_csv["zone_id"] == example_zone].copy()
    example["ts"] = pd.to_datetime(example["ts"])
    plot_forecast_example(out / "forecast_example_zone.png", example)
    plot_multipliers(out / "surge_multiplier_hist.png", surge_csv["multiplier"].to_numpy())
    plot_switchback(
        out / "switchback_estimates.png",
        switch_df["estimate"].to_numpy(),
        true_effect,
        coverage,
    )
    payload["hardware"] = _hardware(tracker.stop())
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    assert_finite(payload)
    write_json(out / "metrics.json", payload)
    if update_docs and not quick:
        (root / "RESUME_NOTES.md").write_text(render_resume_notes(payload), encoding="utf-8")
        update_readme(root / "README.md", results_markdown(payload))
    for item in criteria:
        flag = "PASS" if item["passed"] else "FAIL"
        print(f"{flag} {item['name']}: {item['value']:.6f} ({item['op']} {item['threshold']})", flush=True)
    print(f"runtime_seconds={payload['runtime_seconds']}", flush=True)
    return payload
