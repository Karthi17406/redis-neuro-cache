"""Shared configuration for the MiniRedis ML pipeline.

The feature list here is the Python half of the contract
declared in ``MLFeatures.java``. Both halves are checked
against each other by ``tests/test_contract.py``, so the
two cannot drift apart unnoticed.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

# ml/ lives inside src/, so the project root for data
# purposes is the parent of this package.
SRC_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = SRC_DIR / "data"

MODEL_DIR = SRC_DIR / "models"

# The live log written by AccessLogger at runtime.
DEFAULT_ACCESS_LOG = SRC_DIR / "access_log.csv"

# The larger synthetic log produced by WorkloadGenerator.
DEFAULT_WORKLOAD_LOG = DATA_DIR / "workload_log.csv"

DEFAULT_DATASET = DATA_DIR / "ml_dataset.csv"

DEFAULT_MODEL = MODEL_DIR / "eviction_model.joblib"

DEFAULT_METRICS = MODEL_DIR / "metrics.json"


# ---------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------

# Order matters: it must match MLFeatures.NAMES exactly.
FEATURE_NAMES: list[str] = [
    "total_accesses",
    "get_count",
    "set_count",
    "age_seconds",
    "recency_seconds",
    "mean_interval_seconds",
    "ewma_interval_seconds",
    "accesses_per_minute",
    "get_ratio",
    "value_size",
    "recency_over_mean_interval",
]

# Must match KeyStats.EWMA_ALPHA on the Java side.
EWMA_ALPHA = 0.3

LABEL_NAME = "will_be_reused"


# ---------------------------------------------------------------
# Dataset construction defaults
# ---------------------------------------------------------------

# How many points in time the dataset is sampled at.
DEFAULT_SNAPSHOTS = 250

# Fraction of the log replayed before the first snapshot,
# so that early snapshots are not dominated by keys with
# a single access and no measurable interval.
DEFAULT_WARMUP_FRACTION = 0.15

# Fraction of the total time span used as the reuse
# horizon when none is given explicitly.
DEFAULT_HORIZON_FRACTION = 0.05

# Fraction of snapshots (by time, never shuffled) kept
# for testing. A random split would leak the future.
DEFAULT_TEST_FRACTION = 0.25


# ---------------------------------------------------------------
# Service defaults
# ---------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8000
