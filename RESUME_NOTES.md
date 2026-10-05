# Resume notes

## Draft bullets

- Forecast NYC zone-hour taxi demand with LightGBM at 0.1461 WAPE on held-out April, versus 0.1930 for a seasonal-naive baseline (relative reduction 0.2427).
- Built a marketplace simulator and surge-pricing optimizer that raised simulated completed trips by 0.2131 over flat pricing, with average rider price 18.7555 versus 20.0000 under the flat multiplier.
- Designed switchback experiments that held absolute interference bias to 0.0115 (0.0024 of the naive user-level A/B bias of 4.7969), with 0.9420 coverage of the true effect by nominal 95% CIs across 500 simulations.

## Skills and tools

Python, SQL, DuckDB, pandas, NumPy, LightGBM, SciPy, statsmodels, matplotlib, pytest. Time-series forecasting, marketplace simulation, constrained optimization, and experiment analysis with clustered standard errors.

## Project summary

This project forecasts yellow-taxi pickups by zone and hour, then uses those forecasts as base demand in a small marketplace simulator. A grid search sets a surge multiplier for each zone-hour, and simulated switchback experiments check that a block-randomized analysis recovers the known pricing effect while a user-level A/B test does not.

## GitHub description

Forecast NYC taxi zone-hour demand, optimize a simulated surge multiplier, and compare switchback experiments with a user-level A/B test.
