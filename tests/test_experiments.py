import numpy as np

from ride_pricing.experiments import fit_cluster_ate, run_switchback_coverage, switchback_bias
from ride_pricing.simulator import user_level_and_market_effects


def test_cluster_interval_is_ordered_around_the_estimate():
    rng = np.random.default_rng(0)
    n_blocks, n_units, tau = 40, 6, 3.5
    treat_block = rng.integers(0, 2, size=n_blocks)
    treat = np.repeat(treat_block, n_units)
    shock = np.repeat(rng.normal(0, 0.4, size=n_blocks), n_units)
    noise = rng.normal(0, 0.4, size=n_blocks * n_units)
    outcome = tau * treat + shock + noise
    clusters = np.repeat(np.arange(n_blocks), n_units)
    estimate, low, high = fit_cluster_ate(outcome, treat, clusters)
    assert low < estimate < high
    assert abs(estimate - tau) < 1.5


def test_switchback_coverage_and_bias_versus_naive():
    d0 = np.array([120.0, 90.0, 150.0, 80.0])
    s0 = np.array([70.0, 160.0, 90.0, 50.0])
    naive, true = user_level_and_market_effects(d0, s0, 0.85)
    true_effect = float(true.mean())
    naive_bias = abs(float(naive.mean()) - true_effect)
    experiments = run_switchback_coverage(
        true_effect=true_effect,
        n_experiments=40,
        n_blocks=30,
        n_units=4,
        seed=0,
        sigma_block=0.5,
        sigma_unit=0.5,
    )
    coverage = experiments["covered"].mean()
    assert coverage >= 0.80
    sb_bias = switchback_bias(experiments["estimate"].to_numpy(), true_effect)
    assert sb_bias < naive_bias / 3.0
    recomputed = (
        (experiments["ci_low"] <= experiments["true_effect"])
        & (experiments["true_effect"] <= experiments["ci_high"])
    ).mean()
    assert recomputed == coverage


def test_switchback_frame_has_one_row_per_experiment():
    frame = run_switchback_coverage(1.0, n_experiments=5, n_blocks=12, n_units=3, seed=1)
    assert len(frame) == 5
    assert list(frame["experiment_id"]) == [0, 1, 2, 3, 4]
    assert frame["true_effect"].nunique() == 1
