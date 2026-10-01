"""Build a labelled training set from an access log.

    python -m ml.build_dataset --log data/workload_log.csv

The label answers the question the cache actually has to
answer at eviction time: *will this key be asked for
again in the near future?* A key whose answer is no is
the one that should be dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_DATASET,
    DEFAULT_SNAPSHOTS,
    DEFAULT_WARMUP_FRACTION,
    DEFAULT_WORKLOAD_LOG,
    LABEL_NAME,
)
from .data_loader import load_access_log, summarise
from .features import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the MiniRedis eviction training set"
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_WORKLOAD_LOG),
        help="access log to replay",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_DATASET),
        help="where to write the dataset CSV",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=DEFAULT_SNAPSHOTS,
        help="number of points in time to sample",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=DEFAULT_WARMUP_FRACTION,
        help="fraction of the log replayed before sampling starts",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=None,
        help="reuse horizon; defaults to 5%% of the log span",
    )
    parser.add_argument(
        "--min-accesses",
        type=int,
        default=1,
        help="skip keys with fewer accesses than this",
    )

    args = parser.parse_args(argv)

    frame = load_access_log(args.log)

    print("Access log")
    for name, value in summarise(frame).items():
        print(f"  {name}: {value}")

    dataset, settings = build_dataset(
        frame,
        snapshots=args.snapshots,
        warmup_fraction=args.warmup_fraction,
        horizon_seconds=args.horizon_seconds,
        min_accesses=args.min_accesses,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)

    settings_path = out_path.with_suffix(".settings.json")
    settings_path.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    positives = int(dataset[LABEL_NAME].sum())
    total = int(len(dataset))

    print("\nDataset")
    print(f"  rows: {total}")
    print(f"  distinct keys: {dataset['key'].nunique()}")
    print(f"  snapshots: {dataset['snapshot_index'].nunique()}")
    print(f"  horizon: {settings['horizon_seconds']} s")
    print(
        f"  positives: {positives} "
        f"({positives / total:.1%})"
    )
    print(f"\nWritten to {out_path}")
    print(f"Settings written to {settings_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
