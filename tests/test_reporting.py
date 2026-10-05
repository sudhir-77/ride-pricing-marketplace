from pathlib import Path

from ride_pricing.config import TLC_MONTHS, TLC_ROWS, TLC_SHA256
from ride_pricing.reporting import render_resume_notes, results_markdown, update_readme


def _payload() -> dict:
    return {
        "project": "ride-pricing-marketplace",
        "success_criteria": [
            {
                "name": "April zone-hour demand WAPE (top 50 zones)",
                "value": 0.14612673860775385,
                "op": "<=",
                "threshold": 0.25,
                "threshold_high": None,
                "passed": True,
            }
        ],
        "metrics": {
            "forecast": {
                "april_top50_wape_lightgbm": 0.14612673860775385,
                "april_top50_wape_seasonal_naive": 0.19296595644565825,
                "wape_improvement_vs_naive": 0.24273306390754443,
                "val_wape_lightgbm": 0.14744705686606388,
                "val_wape_seasonal_naive": 0.18527970979495112,
            },
            "simulator": {
                "completed_trips_optimized": 10.0,
                "completed_trips_flat": 8.0,
                "trips_lift": 0.25,
                "avg_rider_price_optimized": 18.5,
                "avg_rider_price_flat": 20.0,
            },
            "switchback": {
                "ci_coverage": 0.942,
                "switchback_bias": 0.011451099797499031,
                "naive_ab_bias": 4.796914548233538,
                "bias_ratio": 0.0023871802764791577,
            },
        },
        "hardware": {"peak_rss_gb": 0.43},
        "runtime_seconds": 7.0,
    }


def test_tlc_files_are_pinned():
    for month in TLC_MONTHS:
        assert len(TLC_SHA256[month]) == 64
        assert int(TLC_SHA256[month], 16) >= 0
        assert TLC_ROWS[month] > 1_000_000


def test_resume_notes_use_metric_digits_and_plain_words():
    text = render_resume_notes(_payload())
    assert "0.1461" in text
    assert "0.1930" in text
    assert "0.2427" in text
    assert "0.9420" in text
    lowered = text.lower()
    for banned in ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded"):
        assert banned not in lowered
    assert "\u2014" not in text


def test_results_table_and_readme_markers(tmp_path: Path):
    table = results_markdown(_payload())
    assert "0.146127" in table
    assert "20.000000000000004" not in table
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- RESULTS_START -->\nold\n<!-- RESULTS_END -->\nafter\n", encoding="utf-8")
    update_readme(readme, table)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("before")
    assert text.rstrip().endswith("after")
    assert "0.146127" in text
