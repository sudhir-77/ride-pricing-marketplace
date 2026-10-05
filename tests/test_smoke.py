import json
from pathlib import Path

import pandas as pd
import pytest

from ride_pricing.metrics import relative_improvement, wape, weighted_average
from ride_pricing.pipeline import run_pipeline


def test_quick_pipeline_writes_recomputable_artifacts(tmp_path: Path):
    out = tmp_path / "results"
    cache = tmp_path / "data"
    payload = run_pipeline(
        quick=True,
        output_dir=out,
        cache_dir=cache,
        update_docs=False,
        project_root=tmp_path,
    )
    assert payload["project"] == "ride-pricing-marketplace"
    assert payload["runtime_seconds"] < 180
    assert payload["hardware"]["peak_rss_gb"] < 3.0
    assert payload["seeds"] == [0]
    for name in ("train", "val", "test"):
        assert payload["splits"]["zone_hour_panel"][name] > 0
    assert (out / "metrics.json").exists()
    on_disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["success_criteria"]

    forecast = pd.read_csv(out / "forecast_test_predictions.csv")
    assert {"row_id", "zone_id", "ts", "actual", "pred_lightgbm", "pred_seasonal_naive"} <= set(forecast.columns)
    assert forecast["row_id"].is_unique
    model_wape = wape(forecast["actual"], forecast["pred_lightgbm"])
    naive_wape = wape(forecast["actual"], forecast["pred_seasonal_naive"])
    assert model_wape == pytest.approx(payload["metrics"]["forecast"]["april_top50_wape_lightgbm"], rel=1e-6)
    assert naive_wape == pytest.approx(payload["metrics"]["forecast"]["april_top50_wape_seasonal_naive"], rel=1e-6)
    assert relative_improvement(naive_wape, model_wape) == pytest.approx(
        payload["metrics"]["forecast"]["wape_improvement_vs_naive"], rel=1e-6
    )

    # No test timestamp appears in the saved training window.
    manifest = json.loads((cache / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    test_ts = pd.to_datetime(forecast["ts"])
    assert test_ts.min() >= pd.Timestamp(manifest["test_start"])

    surge = pd.read_csv(out / "surge_zone_hour.csv")
    lift = surge["completed_optimized"].sum() / surge["completed_flat"].sum() - 1.0
    price_opt = weighted_average(surge["rider_price_optimized"], surge["completed_optimized"])
    price_flat = weighted_average(surge["rider_price_flat"], surge["completed_flat"])
    assert lift == pytest.approx(payload["metrics"]["simulator"]["trips_lift"], rel=1e-6, abs=1e-8)
    assert price_opt == pytest.approx(payload["metrics"]["simulator"]["avg_rider_price_optimized"], rel=1e-6)
    assert price_flat == pytest.approx(payload["metrics"]["simulator"]["avg_rider_price_flat"], rel=1e-6)
    assert price_opt <= price_flat + 1e-6

    experiments = pd.read_csv(out / "switchback_experiments.csv")
    covered = (
        (experiments["ci_low"] <= experiments["true_effect"])
        & (experiments["true_effect"] <= experiments["ci_high"])
    ).mean()
    assert covered == pytest.approx(payload["metrics"]["switchback"]["ci_coverage"], abs=1e-9)
    assert (out / "wape_comparison.png").exists()
    assert (out / "forecast_example_zone.png").exists()
