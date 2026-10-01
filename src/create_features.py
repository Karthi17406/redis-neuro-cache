"""Per key access counts from the log.

    python create_features.py

This is the original summary script, kept because its
output, ml_features.csv, is a useful thing to eyeball.
It now reads the log through ml.data_loader instead of
naming the columns itself, so it keeps working after the
logger grew hit and value_size columns, and it still
reads a log written by the older three column version.

For the dataset the model is actually trained on, which
needs point in time features and a forward looking
label, use:

    python -m ml.build_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_ACCESS_LOG
from ml.data_loader import AccessLogError, load_access_log

DEFAULT_OUTPUT = "ml_features.csv"


def summarise(log: pd.DataFrame) -> pd.DataFrame:
    """Count operations per key.

    EVICT is counted separately rather than folded into
    DELETE. Both remove a key, but only DELETE means a
    client asked for it to go, and telling them apart is
    the difference between measuring the workload and
    measuring the cache own behaviour.
    """
    operations = log["operation"]

    summary = (
        log.assign(
            is_get=(operations == "GET"),
            is_set=(operations == "SET"),
            is_delete=(operations == "DELETE"),
            is_exists=(operations == "EXISTS"),
            is_evict=(operations == "EVICT"),
        )
        .groupby("key")
        .agg(
            total_accesses=("key", "count"),
            get_count=("is_get", "sum"),
            set_count=("is_set", "sum"),
            delete_count=("is_delete", "sum"),
            exists_count=("is_exists", "sum"),
            evict_count=("is_evict", "sum"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )

    summary["lifespan_seconds"] = (
        summary["last_seen"] - summary["first_seen"]
    ) / 1000.0

    return summary.sort_values(
        "total_accesses", ascending=False
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise the MiniRedis access log"
    )
    parser.add_argument("--log", default=str(DEFAULT_ACCESS_LOG))
    parser.add_argument("--out", default=DEFAULT_OUTPUT)

    args = parser.parse_args(argv)

    log_path = Path(args.log)

    if not log_path.exists():
        print(f"No access log at {log_path}.")
        print("Run the server, or: java WorkloadGenerator")
        return 1

    try:
        log = load_access_log(log_path)

    except AccessLogError as error:
        print(error)
        return 1

    features = summarise(log)

    features.to_csv(args.out, index=False)

    print(features.head(20).to_string(index=False))

    print(
        f"\n{len(features)} keys from {len(log)} log rows "
        f"-> {args.out}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
