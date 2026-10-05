"""Metric definitions used by the pipeline and by the tests.

An outside reviewer can recompute every reported forecast and pricing number
from the CSVs under results/ with the functions in this module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: sum(|y - yhat|) / sum(|y|)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    denom = np.abs(actual).sum()
    if denom <= 0:
        raise ValueError("WAPE denominator is zero")
    return float(np.abs(actual - predicted).sum() / denom)


def relative_improvement(baseline: float, model: float) -> float:
    """Fractional reduction in a lower-is-better metric. 0.15 means 15% better."""
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return float((baseline - model) / baseline)


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights sum to zero")
    return float(np.dot(values, weights) / total)


def compare(value: float, op: str, threshold: float, threshold_high: float | None = None) -> bool:
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == ">":
        return value > threshold
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        return threshold <= value <= threshold_high
    raise ValueError(f"unknown op {op}")


def make_criterion(
    name: str,
    value: float,
    op: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict[str, Any]:
    value_f = float(value)
    threshold_f = float(threshold)
    high = None if threshold_high is None else float(threshold_high)
    return {
        "name": name,
        "value": value_f,
        "op": op,
        "threshold": threshold_f,
        "threshold_high": high,
        "passed": bool(compare(value_f, op, threshold_f, high)),
    }


def assert_finite(obj: Any, path: str = "metrics") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_finite(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_finite(value, f"{path}[{i}]")
        return
    if isinstance(obj, (float, np.floating)):
        if not math.isfinite(float(obj)):
            raise ValueError(f"non-finite number at {path}")
    if isinstance(obj, (np.floating, np.integer)):
        return


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
