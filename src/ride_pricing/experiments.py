"""Switchback experiments and the biased user-level comparison.

Switchback assignment is constant inside a time block and shared by every unit
in that block. The analysis is OLS of the outcome on treatment with cluster-robust
standard errors at the block. The data-generating process matches that design,
and the treatment coefficient is the known average effect taken from the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ride_pricing.config import SEED


def fit_cluster_ate(outcome: np.ndarray, treat: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """Point estimate and 95% CI for the treatment coefficient, clustered by block."""
    exog = sm.add_constant(np.asarray(treat, dtype=float))
    result = sm.OLS(np.asarray(outcome, dtype=float), exog).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(clusters)},
    )
    estimate = float(result.params[1])
    low, high = result.conf_int(alpha=0.05)[1]
    return estimate, float(low), float(high)


def run_switchback_coverage(
    true_effect: float,
    n_experiments: int,
    n_blocks: int,
    n_units: int,
    seed: int = SEED,
    sigma_block: float | None = None,
    sigma_unit: float | None = None,
) -> pd.DataFrame:
    """Monte Carlo the switchback design. Each row is one experiment."""
    if n_experiments < 1 or n_blocks < 4 or n_units < 2:
        raise ValueError("switchback design is too small")
    scale = max(1.0, abs(float(true_effect)))
    if sigma_block is None:
        sigma_block = 0.45 * scale
    if sigma_unit is None:
        sigma_unit = 0.45 * scale
    master = np.random.default_rng(seed)
    rows: list[tuple[int, float, float, float, float, int]] = []
    for experiment_id in range(n_experiments):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        treat_block = rng.integers(0, 2, size=n_blocks)
        if treat_block.min() == treat_block.max():
            treat_block = treat_block.copy()
            treat_block[0] = 1 - int(treat_block[0])
        block_shock = rng.normal(0.0, sigma_block, size=n_blocks)
        unit_noise = rng.normal(0.0, sigma_unit, size=(n_blocks, n_units))
        treat = np.repeat(treat_block, n_units)
        shock = np.repeat(block_shock, n_units)
        outcome = float(true_effect) * treat + shock + unit_noise.ravel()
        clusters = np.repeat(np.arange(n_blocks), n_units)
        estimate, low, high = fit_cluster_ate(outcome, treat, clusters)
        covered = int(low <= float(true_effect) <= high)
        rows.append((experiment_id, estimate, low, high, float(true_effect), covered))
    return pd.DataFrame(
        rows,
        columns=["experiment_id", "estimate", "ci_low", "ci_high", "true_effect", "covered"],
    )


def switchback_bias(estimates: np.ndarray, true_effect: float) -> float:
    return float(abs(np.mean(np.asarray(estimates, dtype=float)) - float(true_effect)))
