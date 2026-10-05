import math

import numpy as np
import pytest

from ride_pricing.metrics import compare, make_criterion, relative_improvement, wape, weighted_average


def test_wape_matches_hand_calculation():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 10.0, 60.0])
    # absolute errors 0 + 10 + 30 = 40; sum of actuals = 60
    assert wape(actual, predicted) == pytest.approx(40.0 / 60.0)


def test_wape_rejects_zero_denominator():
    with pytest.raises(ValueError):
        wape(np.array([0.0, 0.0]), np.array([1.0, 1.0]))


def test_relative_improvement_hand_calculation():
    # (0.40 - 0.30) / 0.40 = 0.25
    assert relative_improvement(0.40, 0.30) == pytest.approx(0.25)


def test_weighted_average_hand_calculation():
    assert weighted_average(np.array([10.0, 30.0]), np.array([1.0, 3.0])) == pytest.approx(25.0)


def test_compare_ops():
    assert compare(0.2, "<=", 0.25)
    assert not compare(0.3, "<=", 0.25)
    assert compare(0.2, ">=", 0.15)
    assert compare(1.0, "<", 3.0)
    assert compare(0.5, "between", 0.2, 0.7)
    assert not compare(0.1, "between", 0.2, 0.7)


def test_criterion_passed_flag_matches_operator():
    item = make_criterion("wape", 0.2, "<=", 0.25)
    assert item["passed"] is True
    assert item["threshold_high"] is None
    failed = make_criterion("lift", 0.01, ">=", 0.05)
    assert failed["passed"] is False
    assert math.isfinite(item["value"])
