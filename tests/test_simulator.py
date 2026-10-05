import numpy as np
import pandas as pd
import pytest

from ride_pricing.config import (
    BASE_FARE_USD,
    OFFPEAK_SUPPLY_RATIO,
    PEAK_SUPPLY_RATIO,
    RIDER_ELASTICITY,
    SUPPLY_ELASTICITY,
)
from ride_pricing.metrics import weighted_average
from ride_pricing.optimize import optimize_surge
from ride_pricing.simulator import (
    build_market_frame,
    clearing_multiplier,
    marketplace_outcomes,
    supply_ratio_for_hours,
    user_level_and_market_effects,
)


def test_marketplace_identity_at_flat_price():
    demand, supply, completed = marketplace_outcomes(
        np.array([100.0, 40.0]),
        np.array([80.0, 90.0]),
        np.array([1.0, 1.0]),
    )
    assert demand.tolist() == [100.0, 40.0]
    assert supply.tolist() == [80.0, 90.0]
    assert completed.tolist() == [80.0, 40.0]


def test_price_moves_demand_and_supply_in_opposite_directions():
    base_d = np.array([100.0])
    base_s = np.array([80.0])
    _, _, flat = marketplace_outcomes(base_d, base_s, np.array([1.0]))
    demand_hi, supply_hi, _ = marketplace_outcomes(base_d, base_s, np.array([1.5]))
    demand_lo, supply_lo, _ = marketplace_outcomes(base_d, base_s, np.array([0.8]))
    assert demand_hi[0] < 100.0 < demand_lo[0]
    assert supply_lo[0] < 80.0 < supply_hi[0]
    assert flat[0] == pytest.approx(80.0)


def test_completed_never_exceeds_either_side():
    rng = np.random.default_rng(0)
    d0 = rng.uniform(10, 200, size=30)
    s0 = rng.uniform(10, 200, size=30)
    multiplier = rng.uniform(0.7, 1.8, size=30)
    demand, supply, completed = marketplace_outcomes(d0, s0, multiplier)
    assert np.all(completed <= demand + 1e-8)
    assert np.all(completed <= supply + 1e-8)


def test_clearing_multiplier_hand_values():
    assert clearing_multiplier(100.0, 100.0) == pytest.approx(1.0)
    # m = (50/100) ** (1 / (-0.5 - 1.0)) = 0.5 ** (1 / -1.5) = 2 ** (2/3)
    expected = 2.0 ** (2.0 / 3.0)
    found = clearing_multiplier(100.0, 50.0, rider_elasticity=-0.5, supply_elasticity=1.0)
    assert found == pytest.approx(expected)
    demand, supply, _ = marketplace_outcomes(
        np.array([100.0]),
        np.array([50.0]),
        np.array([found]),
        rider_elasticity=-0.5,
        supply_elasticity=1.0,
    )
    assert demand[0] == pytest.approx(supply[0])


def test_known_multiplier_completed_trips():
    demand, supply, completed = marketplace_outcomes(
        np.array([100.0]),
        np.array([80.0]),
        np.array([1.5]),
        rider_elasticity=-0.5,
        supply_elasticity=1.0,
    )
    assert demand[0] == pytest.approx(100.0 / np.sqrt(1.5))
    assert supply[0] == pytest.approx(120.0)
    assert completed[0] == pytest.approx(100.0 / np.sqrt(1.5))


def test_peak_hours_are_shorter_on_supply_than_offpeak():
    hours = np.arange(24)
    ratio = supply_ratio_for_hours(hours)
    assert ratio[8] == PEAK_SUPPLY_RATIO
    assert ratio[3] == OFFPEAK_SUPPLY_RATIO
    assert PEAK_SUPPLY_RATIO < 1.0 < OFFPEAK_SUPPLY_RATIO


def test_optimizer_beats_flat_on_a_mixed_day_without_raising_price():
    # One representative day. Weights are a rough diurnal shape, not fit to April.
    weights = np.array(
        [
            0.020, 0.015, 0.010, 0.008, 0.008, 0.012,
            0.025, 0.045, 0.055, 0.050,
            0.045, 0.048, 0.050, 0.050, 0.048, 0.050,
            0.055, 0.060, 0.058, 0.050,
            0.045, 0.040, 0.035, 0.028,
        ]
    )
    weights = weights / weights.sum()
    demand = weights * 10_000.0
    frame = pd.DataFrame(
        {
            "zone_id": 1,
            "ts": pd.date_range("2024-04-02", periods=24, freq="h"),
            "base_demand": demand,
        }
    )
    market = build_market_frame(frame)
    solution = optimize_surge(market["base_demand"].to_numpy(), market["base_supply"].to_numpy())
    trips_opt = float(solution["completed_optimized"].sum())
    trips_flat = float(solution["completed_flat"].sum())
    price_opt = weighted_average(solution["rider_price_optimized"], solution["completed_optimized"])
    price_flat = weighted_average(solution["rider_price_flat"], solution["completed_flat"])
    assert trips_opt >= trips_flat * 1.05
    assert price_opt <= price_flat + 1e-8
    assert price_flat == pytest.approx(BASE_FARE_USD)


def test_user_level_contrast_is_biased_for_a_citywide_discount():
    d0 = np.array([100.0, 100.0, 80.0])
    s0 = np.array([70.0, 140.0, 50.0])
    naive, true = user_level_and_market_effects(
        d0, s0, 0.85, rider_elasticity=RIDER_ELASTICITY, supply_elasticity=SUPPLY_ELASTICITY
    )
    assert abs(naive.mean() - true.mean()) > 1.0
    # Shared supply makes the user-level contrast larger than the citywide effect
    # on a tight cell: treated riders crowd out control riders.
    tight_naive, tight_true = user_level_and_market_effects(
        np.array([100.0]),
        np.array([60.0]),
        0.85,
    )
    assert tight_naive[0] > tight_true[0]
