"""Reads the access log written by ``AccessLogger.java``.

The log has two generations of rows living side by side:

* legacy rows with three columns
  ``key,operation,timestamp``
* current rows with five
  ``key,operation,timestamp,hit,value_size``

Neither generation has a header. Mixing them in one file
is deliberate, so that logs collected before the ML work
started are not thrown away. This loader normalises both
into a single frame.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

COLUMNS = ["key", "operation", "timestamp", "hit", "value_size"]

# Operations the database is able to write. EVICT is
# written by the cache itself rather than requested by a
# client, which is why the label builder treats it
# differently, but it is not an unknown operation and
# must not be reported as one.
KNOWN_OPERATIONS = {"GET", "SET", "DELETE", "EXISTS", "EVICT"}


class AccessLogError(ValueError):
    """Raised when a log file cannot be interpreted."""


def load_access_log(
    path: str | Path,
    *,
    drop_malformed: bool = True,
) -> pd.DataFrame:
    """Load an access log into a normalised DataFrame.

    Returns a frame with columns ``key``, ``operation``,
    ``timestamp`` (int, epoch millis), ``hit``
    (nullable bool) and ``value_size`` (nullable int),
    sorted by timestamp.

    Legacy rows get ``hit=<NA>`` and ``value_size=<NA>``
    rather than a guessed value, so that downstream code
    can tell "unknown" apart from "miss".
    """
    path = Path(path)

    if not path.exists():
        raise AccessLogError(f"Access log not found: {path}")

    rows: list[dict] = []
    malformed = 0

    # csv.reader handles the quoting that AccessLogger
    # applies to keys containing commas or newlines.
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.reader(handle):
            if not raw or all(not field.strip() for field in raw):
                continue

            # A header row, if a future writer adds one.
            if raw[0].strip().lower() == "key":
                continue

            if len(raw) < 3:
                malformed += 1
                continue

            key, operation, timestamp = raw[0], raw[1].strip().upper(), raw[2]

            try:
                timestamp_ms = int(str(timestamp).strip())
            except (TypeError, ValueError):
                malformed += 1
                continue

            hit = _parse_optional_bool(raw[3]) if len(raw) > 3 else None

            value_size = _parse_optional_int(raw[4]) if len(raw) > 4 else None

            # AccessLogger writes -1 for "not applicable".
            if value_size is not None and value_size < 0:
                value_size = None

            rows.append(
                {
                    "key": key,
                    "operation": operation,
                    "timestamp": timestamp_ms,
                    "hit": hit,
                    "value_size": value_size,
                }
            )

    # drop_malformed is about tolerating bad lines, not
    # about accepting a log with nothing in it. A log
    # that yielded no rows at all is reported here,
    # where the file name is still in hand, rather than
    # surfacing later as a NaN in a feature vector.
    if not rows:
        detail = (
            f" ({malformed} malformed rows dropped)"
            if malformed
            else ""
        )

        raise AccessLogError(
            f"No usable rows in {path}{detail}"
        )

    if malformed and not drop_malformed:
        raise AccessLogError(
            f"{malformed} malformed rows in {path}"
        )

    frame = pd.DataFrame(rows, columns=COLUMNS)

    frame = frame.astype(
        {
            "timestamp": "int64",
            "hit": "boolean",
            "value_size": "Int64",
        }
    )

    # A stable sort keeps rows that share a millisecond in
    # the order they were written, which matters because
    # the replay is order sensitive.
    frame = frame.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    return frame


def _parse_optional_bool(value: str) -> bool | None:
    value = str(value).strip()

    if value in {"1", "true", "True"}:
        return True

    if value in {"0", "false", "False"}:
        return False

    return None


def _parse_optional_int(value: str) -> int | None:
    value = str(value).strip()

    if not value:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def summarise(frame: pd.DataFrame) -> dict:
    """Quick description of a loaded log, for the CLIs."""
    if frame.empty:
        return {
            "rows": 0,
            "distinct_keys": 0,
            "span_seconds": 0.0,
            "operations": {},
            "legacy_rows": 0,
        }

    span_ms = int(frame["timestamp"].max() - frame["timestamp"].min())

    return {
        "rows": int(len(frame)),
        "distinct_keys": int(frame["key"].nunique()),
        "span_seconds": round(span_ms / 1000.0, 3),
        "operations": {
            str(k): int(v)
            for k, v in frame["operation"].value_counts().items()
        },
        "legacy_rows": int(frame["hit"].isna().sum()),
        "unknown_operations": sorted(
            set(frame["operation"].unique()) - KNOWN_OPERATIONS
        ),
    }
