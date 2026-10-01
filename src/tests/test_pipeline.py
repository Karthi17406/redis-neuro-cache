"""Tests for the log reader, the replay and the dataset.

These work on small logs built in the test rather than
on the real one, so that each property is checked
against a case where the right answer can be worked out
by hand.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from ml.config import FEATURE_NAMES, LABEL_NAME
from ml.data_loader import AccessLogError, load_access_log
from ml.features import (
    KeyStats,
    TrackerReplay,
    build_dataset,
    resolve_horizon_ms,
)

BASE = 1_700_000_000_000


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def write_log(path, rows: list[tuple]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return str(path)


# ---------------------------------------------------------------
# The log reader
# ---------------------------------------------------------------


def test_reads_the_current_five_column_format(tmp_path):
    path = write_log(
        tmp_path / "log.csv",
        [
            ("a", "SET", BASE, "false", 10),
            ("a", "GET", BASE + 100, "true", -1),
        ],
    )

    log = load_access_log(path)

    assert list(log["key"]) == ["a", "a"]
    assert list(log["operation"]) == ["SET", "GET"]
    assert log["hit"].tolist() == [False, True]


def test_reads_the_legacy_three_column_format(tmp_path):
    """An older log has no hit column. Unknown must stay
    distinguishable from a miss, or every legacy row
    would replay as a lookup that failed."""
    path = write_log(
        tmp_path / "log.csv",
        [("a", "SET", BASE), ("a", "GET", BASE + 100)],
    )

    log = load_access_log(path)

    assert len(log) == 2
    assert log["hit"].isna().all()
    assert log["value_size"].isna().all()


def test_sorts_by_timestamp(tmp_path):
    path = write_log(
        tmp_path / "log.csv",
        [
            ("b", "SET", BASE + 500, "false", 1),
            ("a", "SET", BASE, "false", 1),
        ],
    )

    log = load_access_log(path)

    assert list(log["key"]) == ["a", "b"]


def test_malformed_rows_are_dropped(tmp_path):
    path = write_log(
        tmp_path / "log.csv",
        [
            ("a", "SET", BASE, "false", 1),
            ("broken",),
            ("b", "SET", "not-a-number", "false", 1),
        ],
    )

    log = load_access_log(path)

    assert list(log["key"]) == ["a"]


def test_empty_log_is_an_error(tmp_path):
    path = write_log(tmp_path / "log.csv", [])

    with pytest.raises(AccessLogError):
        load_access_log(path)


def test_quoted_keys_survive(tmp_path):
    """AccessLogger quotes a key containing a comma, so
    naive splitting would shift every later column."""
    path = write_log(
        tmp_path / "log.csv",
        [("odd,key", "SET", BASE, "false", 5)],
    )

    log = load_access_log(path)

    assert list(log["key"]) == ["odd,key"]
    assert list(log["value_size"]) == [5]


# ---------------------------------------------------------------
# The replay
# ---------------------------------------------------------------


def test_delete_forgets_the_key():
    replay = TrackerReplay()

    replay.apply("a", "SET", BASE, False, 10)
    replay.apply("a", "DELETE", BASE + 10, True, -1)

    assert "a" not in replay.stats


def test_evict_forgets_the_key():
    """Eviction calls forget in Database.java, so a key
    the cache dropped must leave no statistics behind
    here either."""
    replay = TrackerReplay()

    replay.apply("a", "SET", BASE, False, 10)
    replay.apply("a", "EVICT", BASE + 10, True, -1)

    assert "a" not in replay.stats


def test_a_miss_creates_nothing():
    replay = TrackerReplay()

    replay.apply("ghost", "GET", BASE, False, -1)

    assert replay.stats == {}


def test_a_hit_records():
    replay = TrackerReplay()

    replay.apply("a", "SET", BASE, False, 10)
    replay.apply("a", "GET", BASE + 100, True, -1)

    assert replay.stats["a"].total_accesses == 2


def test_a_key_can_come_back_after_deletion():
    replay = TrackerReplay()

    replay.apply("a", "SET", BASE, False, 10)
    replay.apply("a", "GET", BASE + 50, True, -1)
    replay.apply("a", "DELETE", BASE + 100, True, -1)
    replay.apply("a", "SET", BASE + 200, False, 10)

    # The counters start again, exactly as the Java
    # tracker would after forget.
    assert replay.stats["a"].total_accesses == 1


# ---------------------------------------------------------------
# Feature arithmetic
# ---------------------------------------------------------------


def test_single_access_has_no_interval_history():
    stats = KeyStats(BASE)
    stats.record("SET", BASE, 40)

    vector = dict(
        zip(FEATURE_NAMES, stats.to_feature_vector(BASE + 10_000))
    )

    assert vector["total_accesses"] == 1
    assert vector["age_seconds"] == pytest.approx(10.0)
    assert vector["recency_seconds"] == pytest.approx(10.0)

    # With nothing to average, the age stands in for the
    # interval, which keeps the ratio at 1 rather than
    # dividing by zero.
    assert vector["mean_interval_seconds"] == pytest.approx(10.0)
    assert vector["recency_over_mean_interval"] == pytest.approx(1.0)


def test_regular_reads_give_a_steady_interval():
    stats = KeyStats(BASE)

    for i in range(5):
        stats.record("GET", BASE + i * 1000, -1)

    vector = dict(
        zip(FEATURE_NAMES, stats.to_feature_vector(BASE + 4000))
    )

    assert vector["total_accesses"] == 5
    assert vector["mean_interval_seconds"] == pytest.approx(1.0)
    assert vector["ewma_interval_seconds"] == pytest.approx(1.0)
    assert vector["recency_seconds"] == pytest.approx(0.0)


def test_get_ratio_counts_only_reads():
    stats = KeyStats(BASE)

    stats.record("SET", BASE, 10)
    stats.record("GET", BASE + 100, -1)
    stats.record("GET", BASE + 200, -1)
    stats.record("GET", BASE + 300, -1)

    vector = dict(
        zip(FEATURE_NAMES, stats.to_feature_vector(BASE + 300))
    )

    assert vector["get_count"] == 3
    assert vector["set_count"] == 1
    assert vector["get_ratio"] == pytest.approx(0.75)


def test_value_size_follows_the_latest_write():
    stats = KeyStats(BASE)

    stats.record("SET", BASE, 10)
    stats.record("GET", BASE + 100, -1)
    stats.record("SET", BASE + 200, 999)

    vector = dict(
        zip(FEATURE_NAMES, stats.to_feature_vector(BASE + 200))
    )

    # A GET carries no size, so it must not reset what
    # the last write established.
    assert vector["value_size"] == 999


def test_features_are_always_finite():
    """Every vector is serialised as JSON for the model
    service, which rejects a non finite value. The guards
    against dividing by a zero age have to hold even when
    everything happens at the same millisecond."""
    stats = KeyStats(BASE)

    stats.record("SET", BASE, 0)
    stats.record("GET", BASE, -1)

    vector = stats.to_feature_vector(BASE)

    assert all(np.isfinite(vector))


# ---------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------


def synthetic_log(n_snapshots: int = 40) -> pd.DataFrame:
    """A hot key read constantly and a cold key read once."""
    rows = []

    for i in range(600):
        timestamp = BASE + i * 100

        if i == 0:
            rows.append(("hot:1", "SET", timestamp, False, 10))
            rows.append(("cold:1", "SET", timestamp, False, 10))
        else:
            rows.append(("hot:1", "GET", timestamp, True, -1))

    frame = pd.DataFrame(
        rows,
        columns=["key", "operation", "timestamp", "hit", "value_size"],
    )

    return frame.sort_values("timestamp").reset_index(drop=True)


def test_dataset_has_the_expected_columns():
    dataset, settings = build_dataset(
        synthetic_log(), snapshots=20, horizon_seconds=1.0
    )

    expected = (
        ["snapshot_index", "snapshot_time", "key"]
        + list(FEATURE_NAMES)
        + [LABEL_NAME]
    )

    assert list(dataset.columns) == expected
    assert settings["horizon_seconds"] == pytest.approx(1.0)


def test_label_separates_hot_from_cold():
    dataset, _ = build_dataset(
        synthetic_log(), snapshots=20, horizon_seconds=1.0
    )

    rates = dataset.groupby("key")[LABEL_NAME].mean()

    assert rates["hot:1"] > rates["cold:1"]
    assert rates["cold:1"] == 0.0


def test_features_never_see_the_future():
    """The point of the whole exercise. A row taken at
    snapshot time t must be reproducible from the log up
    to t, with no later row able to change it."""
    log = synthetic_log()

    dataset, _ = build_dataset(
        log, snapshots=10, horizon_seconds=1.0
    )

    row = dataset.iloc[len(dataset) // 2]

    snapshot_ms = int(row["snapshot_time"])

    past = log[log["timestamp"] <= snapshot_ms]

    replay = TrackerReplay()

    for entry in past.itertuples(index=False):
        replay.apply(
            entry.key,
            entry.operation,
            int(entry.timestamp),
            entry.hit,
            entry.value_size,
        )

    rebuilt = replay.stats[row["key"]].to_feature_vector(snapshot_ms)

    stored = [row[name] for name in FEATURE_NAMES]

    assert rebuilt == pytest.approx(stored, rel=1e-12, abs=1e-12)


def test_evictions_do_not_count_as_reuse():
    """A cache that evicts a key must not thereby label
    that key as reused. Otherwise a policy could improve
    its own score simply by evicting more."""
    rows = [
        ("a", "SET", BASE, False, 10),
        ("b", "SET", BASE, False, 10),
        ("b", "GET", BASE + 100, True, -1),
    ]

    # Keep the log alive past the horizon so snapshots
    # have somewhere to land.
    for i in range(1, 60):
        rows.append(("b", "GET", BASE + i * 100, True, -1))

    with_evict = rows + [("a", "EVICT", BASE + 3000, True, -1)]

    columns = ["key", "operation", "timestamp", "hit", "value_size"]

    plain = build_dataset(
        pd.DataFrame(rows, columns=columns),
        snapshots=10,
        horizon_seconds=1.0,
    )[0]

    evicted = build_dataset(
        pd.DataFrame(with_evict, columns=columns),
        snapshots=10,
        horizon_seconds=1.0,
    )[0]

    labels_for_a = evicted[evicted["key"] == "a"][LABEL_NAME]

    assert labels_for_a.sum() == 0

    # And the key stops appearing once it is gone.
    assert len(evicted[evicted["key"] == "a"]) <= len(
        plain[plain["key"] == "a"]
    )


def test_horizon_is_inferred_from_the_log_span():
    log = synthetic_log()

    horizon = resolve_horizon_ms(log, None)

    span = int(log["timestamp"].max() - log["timestamp"].min())

    assert 0 < horizon < span
