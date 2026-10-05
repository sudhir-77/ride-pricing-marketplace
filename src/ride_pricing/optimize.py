"""Per zone-hour surge search.

Each cell is scored on a multiplier grid. A one-dimensional penalty on price,
chosen with a grid and with scipy.optimize, keeps the demand-weighted average
multiplier at or below 1. Flat pricing is always a candidate, so the search
cannot finish with fewer completed trips or a higher average rider price.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ride_pricing.config import (
    BASE_FARE_USD,
    RIDER_ELASTICITY,
    SUPPLY_ELASTICITY,
    SURGE_GRID_SIZE,
    SURGE_M_MAX,
    SURGE_M_MIN,
)
from ride_pricing.simulator import marketplace_outcomes


def multiplier_grid() -> np.ndarray:
    grid = np.linspace(SURGE_M_MIN, SURGE_M_MAX, SURGE_GRID_SIZE)
    grid = np.unique(np.concatenate([grid, [1.0]]))
    return grid.astype(float)


def average_multiplier(multiplier: np.ndarray, completed: np.ndarray) -> float:
    total = float(np.asarray(completed).sum())
    if total <= 0:
        return 1.0
    return float(np.dot(np.asarray(multiplier, dtype=float), np.asarray(completed, dtype=float)) / total)


def _grid_outcomes(
    base_demand: np.ndarray,
    base_supply: np.ndarray,
    grid: np.ndarray,
    rider_elasticity: float,
    supply_elasticity: float,
) -> np.ndarray:
    demand = base_demand[:, None] * np.power(grid[None, :], rider_elasticity)
    supply = base_supply[:, None] * np.power(grid[None, :], supply_elasticity)
    return np.minimum(demand, supply)


def _choose(
    completed_grid: np.ndarray,
    grid: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Reward extra trips. Penalize multipliers above 1 and reward multipliers below 1.
    score = completed_grid * (1.0 - lam * (grid[None, :] - 1.0))
    index = np.argmax(score, axis=1)
    rows = np.arange(completed_grid.shape[0])
    return grid[index], completed_grid[rows, index]


def optimize_surge(
    base_demand: np.ndarray,
    base_supply: np.ndarray,
    rider_elasticity: float = RIDER_ELASTICITY,
    supply_elasticity: float = SUPPLY_ELASTICITY,
    base_fare: float = BASE_FARE_USD,
) -> dict[str, np.ndarray | float]:
    d0 = np.asarray(base_demand, dtype=float)
    s0 = np.asarray(base_supply, dtype=float)
    if d0.shape != s0.shape:
        raise ValueError("demand and supply shapes differ")
    grid = multiplier_grid()
    completed_grid = _grid_outcomes(d0, s0, grid, rider_elasticity, supply_elasticity)
    _, _, completed_flat = marketplace_outcomes(d0, s0, np.ones_like(d0), rider_elasticity, supply_elasticity)

    candidates: list[tuple[np.ndarray, np.ndarray]] = [(np.ones_like(d0), completed_flat)]
    lambdas = np.concatenate([np.array([0.0]), np.geomspace(1e-4, 40.0, 36)])
    for lam in lambdas:
        candidates.append(_choose(completed_grid, grid, float(lam)))

    def penalized_negative_trips(lam: float) -> float:
        multiplier, completed = _choose(completed_grid, grid, float(lam))
        if average_multiplier(multiplier, completed) > 1.0 + 1e-10:
            return 1e9 + float(lam)
        return -float(completed.sum())

    refined = minimize_scalar(
        penalized_negative_trips,
        bounds=(0.0, 40.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    candidates.append(_choose(completed_grid, grid, float(refined.x)))

    feasible = [c for c in candidates if average_multiplier(c[0], c[1]) <= 1.0 + 1e-8]
    if not feasible:
        feasible = [(np.ones_like(d0), completed_flat)]
    multiplier, _ = max(feasible, key=lambda item: (float(item[1].sum()), -average_multiplier(item[0], item[1])))

    _, _, completed = marketplace_outcomes(d0, s0, multiplier, rider_elasticity, supply_elasticity)
    # If a grid point sits a hair above the cap, shrink toward flat until it holds.
    if average_multiplier(multiplier, completed) > 1.0:
        lo, hi = 0.0, 1.0
        best_alpha = 0.0
        best_completed = completed_flat
        best_multiplier = np.ones_like(d0)
        for alpha in np.linspace(0.0, 1.0, 21):
            trial_m = 1.0 + alpha * (multiplier - 1.0)
            _, _, trial_q = marketplace_outcomes(d0, s0, trial_m, rider_elasticity, supply_elasticity)
            if average_multiplier(trial_m, trial_q) <= 1.0 + 1e-10 and trial_q.sum() >= best_completed.sum():
                best_alpha = float(alpha)
                best_completed = trial_q
                best_multiplier = trial_m
        _ = (lo, hi, best_alpha)
        multiplier = best_multiplier
        completed = best_completed

    price_opt = base_fare * multiplier
    price_flat = np.full_like(multiplier, base_fare)
    return {
        "multiplier": multiplier,
        "completed_optimized": completed,
        "completed_flat": completed_flat,
        "rider_price_optimized": price_opt,
        "rider_price_flat": price_flat,
        "lambda_star": float(refined.x),
    }
