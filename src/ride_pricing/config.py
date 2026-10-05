"""Fixed configuration. Forecast thresholds are not chosen from the April test set."""

from __future__ import annotations

from datetime import datetime

# Master seed. Experiment i uses a stream derived from this seed.
SEED = 0

PROJECT_SLUG = "ride-pricing-marketplace"

# Panel window. Test is April 2024 and is never used to train or to stop trees.
PANEL_START = datetime(2024, 1, 1)
PANEL_END = datetime(2024, 5, 1)  # exclusive
# lag_672 needs 28 days of history before the first supervised row.
TRAIN_START = datetime(2024, 1, 29)
VAL_START = datetime(2024, 3, 18)
TEST_START = datetime(2024, 4, 1)

TOP_N_ZONES = 50
MAX_LAG_HOURS = 672

# LightGBM settings are precommitted. Early stopping watches March validation only.
LGB_N_ESTIMATORS = 400
LGB_N_ESTIMATORS_QUICK = 40
LGB_LEARNING_RATE = 0.05
LGB_NUM_LEAVES = 63
LGB_MIN_CHILD_SAMPLES = 50
LGB_EARLY_STOPPING_ROUNDS = 40
LGB_EARLY_STOPPING_ROUNDS_QUICK = 8
LGB_N_JOBS = 4
LGB_REG_LAMBDA = 1.0

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "lag_336",
    "lag_672",
    "roll_mean_24",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "zone_id",
]

# Marketplace assumptions. Not estimated from TLC fares.
# Rider elasticity -0.5 matches the neighborhood of UberX elasticities in
# Cohen, Hahn, Hall, Levitt, and Metcalfe (2016). Driver supply elasticity 1.0
# is a moderate positive response, in the range discussed for ride-hail labor
# supply (Chen, Chevalier, Rossi, and Oehlsen, 2019). Hour-of-day supply ratios
# are assumptions: commute peaks are short of drivers, other hours have slack.
RIDER_ELASTICITY = -0.5
SUPPLY_ELASTICITY = 1.0
BASE_FARE_USD = 20.0
# Citywide price change tested in the switchback / naive A/B comparison.
TREATMENT_MULTIPLIER = 0.85

# Weekday commute peaks (local hour). Supply base = ratio * base demand.
PEAK_HOURS = (7, 8, 9, 16, 17, 18, 19)
PEAK_SUPPLY_RATIO = 0.62
OFFPEAK_SUPPLY_RATIO = 1.55

SURGE_M_MIN = 0.70
SURGE_M_MAX = 1.80
SURGE_GRID_SIZE = 45

N_SWITCHBACK_EXPERIMENTS = 500
N_SWITCHBACK_EXPERIMENTS_QUICK = 30
N_BLOCKS = 96
N_BLOCKS_QUICK = 24
N_UNITS_PER_BLOCK = 8
N_UNITS_PER_BLOCK_QUICK = 4

# Official TLC yellow trip parquet files (NYC Open Data, public).
# sha256 and rows are properties of those files, filled after download and
# checked on every full run.
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_MONTHS = ("2024-01", "2024-02", "2024-03", "2024-04")

# SHA-256 and row counts of the TLC parquet files from the public cloudfront bucket.
TLC_SHA256: dict[str, str] = {
    "2024-01": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    "2024-02": "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    "2024-03": "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
    "2024-04": "ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89",
}
TLC_ROWS: dict[str, int] = {
    "2024-01": 2964624,
    "2024-02": 3007526,
    "2024-03": 3582628,
    "2024-04": 3514289,
}

WAPE_MAX = 0.25
WAPE_IMPROVEMENT_MIN = 0.15
TRIPS_LIFT_MIN = 0.05
COVERAGE_MIN = 0.92
USER_AGENT = "ride-pricing-marketplace/1.0 (educational; local portfolio run)"
