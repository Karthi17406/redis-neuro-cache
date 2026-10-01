"""Offline mirror of the live Java statistics tracker.

``KeyStats`` here is a line for line port of
``KeyStats.java``, and ``TrackerReplay`` mirrors what
``Database.java`` does to the tracker on each operation.

Keeping the two in step is the whole point. If the
offline features were computed differently from the
online ones, the model would be trained on one
distribution and queried on another, and the eviction
decisions would quietly degrade to noise.

The dataset is built by replaying the log and stopping
at a series of snapshot times. At each snapshot every
resident key contributes one row, built only from what
had happened by then, labelled with what happens next.
That ordering is what keeps the future out of the
features.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_HORIZON_FRACTION,
    DEFAULT_SNAPSHOTS,
    DEFAULT_WARMUP_FRACTION,
    EWMA_ALPHA,
    FEATURE_NAMES,
    LABEL_NAME,
)


class KeyStats:
    """Port of KeyStats.java. Keep the two in sync."""

    __slots__ = (
        "total_accesses",
        "get_count",
        "set_count",
        "delete_count",
        "first_seen_ms",
        "last_seen_ms",
        "interval_sum_ms",
        "interval_count",
        "ewma_interval_ms",
        "value_size",
    )

    def __init__(self, now_ms: int) -> None:
        self.total_accesses = 0
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.first_seen_ms = now_ms
        self.last_seen_ms = now_ms
        self.interval_sum_ms = 0
        self.interval_count = 0
        self.ewma_interval_ms = 0.0
        self.value_size = 0

    def record(
        self,
        operation: str,
        now_ms: int,
        value_size: int = -1,
    ) -> None:
        if self.total_accesses > 0:
            gap = now_ms - self.last_seen_ms

            if gap >= 0:
                self.interval_sum_ms += gap
                self.interval_count += 1

                if self.interval_count == 1:
                    self.ewma_interval_ms = float(gap)
                else:
                    self.ewma_interval_ms = (
                        EWMA_ALPHA * gap
                        + (1.0 - EWMA_ALPHA) * self.ewma_interval_ms
                    )

        self.total_accesses += 1
        self.last_seen_ms = now_ms

        if operation == "GET":
            self.get_count += 1
        elif operation == "SET":
            self.set_count += 1
        elif operation == "DELETE":
            self.delete_count += 1

        if value_size >= 0:
            self.value_size = value_size

    def to_feature_vector(self, now_ms: int) -> list[float]:
        """Features in MLFeatures.NAMES order."""
        age_seconds = max(0.0, (now_ms - self.first_seen_ms) / 1000.0)

        recency_seconds = max(0.0, (now_ms - self.last_seen_ms) / 1000.0)

        mean_interval_seconds = (
            (self.interval_sum_ms / self.interval_count) / 1000.0
            if self.interval_count > 0
            else age_seconds
        )

        ewma_interval_seconds = (
            self.ewma_interval_ms / 1000.0
            if self.interval_count > 0
            else age_seconds
        )

        accesses_per_minute = self.total_accesses / max(
            age_seconds / 60.0, 1.0 / 60.0
        )

        get_ratio = (
            self.get_count / self.total_accesses
            if self.total_accesses > 0
            else 0.0
        )

        recency_over_mean_interval = recency_seconds / max(
            mean_interval_seconds, 0.001
        )

        return [
            float(self.total_accesses),
            float(self.get_count),
            float(self.set_count),
            age_seconds,
            recency_seconds,
            mean_interval_seconds,
            ewma_interval_seconds,
            accesses_per_minute,
            get_ratio,
            float(self.value_size),
            recency_over_mean_interval,
        ]


@dataclass
class TrackerReplay:
    """Mirror of what Database.java does to the tracker."""

    stats: dict[str, KeyStats] = field(default_factory=dict)

    def apply(
        self,
        key: str,
        operation: str,
        timestamp_ms: int,
        hit: bool | None,
        value_size: int | None,
    ) -> None:
        size = -1 if value_size is None else int(value_size)

        if operation in {"DELETE", "EVICT"}:
            # Database.delete and the eviction path both
            # call forget, so neither leaves statistics
            # behind for a key that is no longer there.
            self.stats.pop(key, None)
            return

        if operation in {"GET", "EXISTS"} and hit is False:
            # Either a plain miss, which should leave no
            # statistics behind, or an expiry, which
            # calls forget. Dropping covers both.
            self.stats.pop(key, None)
            return

        # SET always records, and a hit or legacy row
        # records too.
        entry = self.stats.get(key)

        if entry is None:
            entry = KeyStats(timestamp_ms)
            self.stats[key] = entry

        entry.record(operation, timestamp_ms, size)

    def resident_keys(self) -> list[str]:
        return list(self.stats)


def resolve_horizon_ms(
    frame: pd.DataFrame,
    horizon_seconds: float | None,
) -> int:
    """Pick a reuse horizon, from the log if not given."""
    if horizon_seconds is not None:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        return int(round(horizon_seconds * 1000))

    span_ms = int(frame["timestamp"].max() - frame["timestamp"].min())

    return max(1, int(span_ms * DEFAULT_HORIZON_FRACTION))


def build_dataset(
    frame: pd.DataFrame,
    *,
    snapshots: int = DEFAULT_SNAPSHOTS,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
    horizon_seconds: float | None = None,
    min_accesses: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Replay the log and emit a labelled training set.

    Returns the dataset and a dict of the settings used,
    which the trainer records alongside the model.
    """
    if frame.empty:
        raise ValueError("Cannot build a dataset from an empty log")

    if snapshots < 2:
        raise ValueError("Need at least 2 snapshots")

    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)

    t_min = int(timestamps[0])
    t_max = int(timestamps[-1])
    span_ms = t_max - t_min

    if span_ms <= 0:
        raise ValueError(
            "Access log covers no time span, so no temporal "
            "features can be derived from it"
        )

    horizon_ms = resolve_horizon_ms(frame, horizon_seconds)

    start_ms = t_min + int(span_ms * warmup_fraction)
    end_ms = t_max - horizon_ms

    if end_ms <= start_ms:
        raise ValueError(
            f"Reuse horizon of {horizon_ms} ms is too long for a "
            f"log spanning {span_ms} ms. Use a shorter horizon."
        )

    snapshot_times = np.linspace(
        start_ms, end_ms, snapshots, dtype=np.int64
    )

    # Per key access times, for the forward looking label.
    #
    # Evictions are excluded. The label asks whether a
    # client comes back for the key, and an eviction is
    # the cache acting on its own. Counting it would let
    # a policy manufacture its own positive labels by
    # evicting, which is precisely backwards.
    key_times: dict[str, list[int]] = {}

    requested = frame["operation"].to_numpy() != "EVICT"

    for key, timestamp, is_request in zip(
        frame["key"].to_numpy(), timestamps, requested, strict=True
    ):
        if is_request:
            key_times.setdefault(key, []).append(int(timestamp))

    keys = frame["key"].to_numpy()
    operations = frame["operation"].to_numpy()
    hits = frame["hit"].to_numpy(na_value=None)
    sizes = frame["value_size"].to_numpy(na_value=None)

    replay = TrackerReplay()

    records: list[tuple] = []
    cursor = 0
    n_rows = len(frame)

    for snapshot_index, snapshot_ms in enumerate(snapshot_times):
        snapshot_ms = int(snapshot_ms)

        # Replay everything that happened by now, and
        # nothing that happened after.
        while cursor < n_rows and int(timestamps[cursor]) <= snapshot_ms:
            replay.apply(
                str(keys[cursor]),
                str(operations[cursor]),
                int(timestamps[cursor]),
                hits[cursor],
                sizes[cursor],
            )
            cursor += 1

        deadline = snapshot_ms + horizon_ms

        for key, entry in replay.stats.items():
            if entry.total_accesses < min_accesses:
                continue

            times = key_times[key]

            # First access strictly after the snapshot.
            position = bisect_right(times, snapshot_ms)

            reused = position < len(times) and times[position] <= deadline

            records.append(
                (
                    snapshot_index,
                    snapshot_ms,
                    key,
                    *entry.to_feature_vector(snapshot_ms),
                    int(reused),
                )
            )

    if not records:
        raise ValueError(
            "No training rows produced. The log may be too short "
            "or the warmup fraction too large."
        )

    columns = (
        ["snapshot_index", "snapshot_time", "key"]
        + FEATURE_NAMES
        + [LABEL_NAME]
    )

    dataset = pd.DataFrame(records, columns=columns)

    settings = {
        "snapshots": int(snapshots),
        "warmup_fraction": float(warmup_fraction),
        "horizon_seconds": round(horizon_ms / 1000.0, 4),
        "horizon_ms": int(horizon_ms),
        "min_accesses": int(min_accesses),
        "log_span_seconds": round(span_ms / 1000.0, 3),
        "log_rows": int(n_rows),
        "dataset_rows": int(len(dataset)),
        "feature_names": list(FEATURE_NAMES),
    }

    return dataset, settings


def features_from_log(
    frame: pd.DataFrame,
    at_ms: int | None = None,
) -> pd.DataFrame:
    """Replay a whole log and return one row per key.

    Used to score a live log rather than to train, and by
    the compatibility wrapper in ``create_features.py``.
    """
    replay = TrackerReplay()

    for row in frame.itertuples(index=False):
        replay.apply(
            str(row.key),
            str(row.operation),
            int(row.timestamp),
            None if pd.isna(row.hit) else bool(row.hit),
            None if pd.isna(row.value_size) else int(row.value_size),
        )

    if at_ms is None:
        at_ms = int(frame["timestamp"].max()) if not frame.empty else 0

    rows = [
        (key, *entry.to_feature_vector(at_ms))
        for key, entry in replay.stats.items()
    ]

    return pd.DataFrame(rows, columns=["key"] + FEATURE_NAMES)
