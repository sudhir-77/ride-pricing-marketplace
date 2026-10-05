"""Marketplace simulator.

Completed trips are the minimum of rider demand and driver supply. A higher
multiplier reduces demand and raises supply. Parameters are assumptions; see
config.py for the citations and the hour-of-day supply ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ride_pricing.config import (
    OFFPEAK_SUPPLY_RATIO,
    PEAK_HOURS,
    PEAK_SUPPLY_RATIO,
    RIDER_ELASTICITY,
    SUPPLY_ELASTICITY,
)


def supply_ratio_for_hours(hours: np.ndarray) -> np.ndarray:
    hours = np.asarray(hours, dtype=int)
    ratio = np.full(hours.shape, OFFPEAK_SUPPLY_RATIO, dtype=float)
    ratio[np.isin(hours, PEAK_HOURS)] = PEAK_SUPPLY_RATIO
    return ratio


def marketplace_outcomes(
    base_demand: np.ndarray,
    base_supply: np.ndarray,
    multiplier: np.ndarray | float,
    rider_elasticity: float = RIDER_ELASTICITY,
    supply_elasticity: float = SUPPLY_ELASTICITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return demand, supply, and completed trips at the given multipliers.

    demand = base_demand * multiplier ** rider_elasticity
    supply = base_supply * multiplier ** supply_elasticity
    completed = min(demand, supply)
    """
    d0 = np.asarray(base_demand, dtype=float)
    s0 = np.asarray(base_supply, dtype=float)
    multiplier_arr = np.asarray(multiplier, dtype=float)
    if np.any(multiplier_arr <= 0):
        raise ValueError("multiplier must be positive")
    demand = d0 * np.power(multiplier_arr, rider_elasticity)
    supply = s0 * np.power(multiplier_arr, supply_elasticity)
    completed = np.minimum(demand, supply)
    return demand, supply, completed


def clearing_multiplier(
    base_demand: float,
    base_supply: float,
    rider_elasticity: float = RIDER_ELASTICITY,
    supply_elasticity: float = SUPPLY_ELASTICITY,
) -> float:
    """Multiplier that equates demand and supply. Requires base_demand > 0."""
    if base_demand <= 0 or base_supply <= 0:
        raise ValueError("clearing multiplier needs positive demand and supply")
    gap = rider_elasticity - supply_elasticity
    if gap == 0:
        raise ValueError("elasticities must differ")
    return float((base_supply / base_demand) ** (1.0 / gap))


def build_market_frame(forecast: pd.DataFrame) -> pd.DataFrame:
    """Attach assumed driver supply to forecasted base demand.

    `forecast` must contain zone_id, ts, and base_demand (trips at multiplier 1).
    """
    out = forecast.copy()
    out["base_demand"] = np.clip(out["base_demand"].to_numpy(dtype=float), 0.0, None)
    hours = pd.to_datetime(out["ts"]).dt.hour.to_numpy()
    out["supply_ratio"] = supply_ratio_for_hours(hours)
    out["base_supply"] = out["base_demand"].to_numpy(dtype=float) * out["supply_ratio"].to_numpy(dtype=float)
    return out


def user_level_and_market_effects(
    base_demand: np.ndarray,
    base_supply: np.ndarray,
    treatment_multiplier: float,
    rider_elasticity: float = RIDER_ELASTICITY,
    supply_elasticity: float = SUPPLY_ELASTICITY,
    fraction_treated: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive user-level contrast versus the true citywide effect, per zone-hour.

    In the mixed market, half of demand faces the treatment price and half faces
    the flat price. They share one driver pool, which responds to the average
    multiplier. Treated riders who request more (a discount) crowd out control
    riders, so the user-level difference is not the effect of treating everyone.

    The naive total-trip contrast is completed_treated / f - completed_control / (1-f).
    The true effect is completed(all treated) - completed(all flat).
    Both are in completed trips for that zone-hour. This is the infinite-sample
    (continuum of riders) bias, not a noisy draw.
    """
    if not 0.0 < fraction_treated < 1.0:
        raise ValueError("fraction_treated must be between 0 and 1")
    d0 = np.asarray(base_demand, dtype=float)
    s0 = np.asarray(base_supply, dtype=float)
    treated_demand = fraction_treated * d0 * np.power(treatment_multiplier, rider_elasticity)
    control_demand = (1.0 - fraction_treated) * d0
    average_multiplier = fraction_treated * treatment_multiplier + (1.0 - fraction_treated)
    shared_supply = s0 * np.power(average_multiplier, supply_elasticity)
    total_demand = treated_demand + control_demand
    fill = np.ones_like(total_demand)
    positive = total_demand > 0
    fill[positive] = np.minimum(1.0, shared_supply[positive] / total_demand[positive])
    completed_treated = treated_demand * fill
    completed_control = control_demand * fill
    naive = np.zeros_like(d0)
    naive[positive] = (
        completed_treated[positive] / fraction_treated
        - completed_control[positive] / (1.0 - fraction_treated)
    )
    _, _, completed_all_treated = marketplace_outcomes(
        d0, s0, np.full_like(d0, treatment_multiplier), rider_elasticity, supply_elasticity
    )
    _, _, completed_flat = marketplace_outcomes(
        d0, s0, np.ones_like(d0), rider_elasticity, supply_elasticity
    )
    true_effect = completed_all_treated - completed_flat
    return naive, true_effect
