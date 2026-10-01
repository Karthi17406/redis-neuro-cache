"""Check that the dataset is worth training on.

A labelled dataset can look perfectly healthy and still
be useless, or worse, misleading. The two failure modes
that matter here are:

* **leakage** - a feature that quietly encodes the
  future. It produces wonderful offline scores and a
  model that collapses in production.
* **a degenerate label** - all one class, or a label
  that is really just one feature in disguise.

Each check below either passes, warns, or fails. The
exit code is non zero if anything fails, so this can sit
in front of training in a script.

    python -m ml.validate_labels
"""

from __future__ import annotations

import argparse
from bisect import bisect_right

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET, DEFAULT_WORKLOAD_LOG, FEATURE_NAMES, LABEL_NAME
from .data_loader import load_access_log
from .features import TrackerReplay, resolve_horizon_ms

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.results.append((status, name, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.results)

    def render(self) -> str:
        width = max(len(name) for _, name, _ in self.results)
        lines = []
        for status, name, detail in self.results:
            lines.append(f"  [{status}] {name.ljust(width)}  {detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------


def check_schema(dataset: pd.DataFrame, report: Report) -> None:
    missing = [
        column
        for column in FEATURE_NAMES + [LABEL_NAME]
        if column not in dataset.columns
    ]

    if missing:
        report.add(FAIL, "schema", f"missing columns: {missing}")
    else:
        report.add(
            PASS,
            "schema",
            f"{len(FEATURE_NAMES)} features + label present",
        )


def check_finite(dataset: pd.DataFrame, report: Report) -> None:
    features = dataset[FEATURE_NAMES]

    n_nan = int(features.isna().to_numpy().sum())

    values = features.to_numpy(dtype=float)
    n_inf = int(np.isinf(values).sum())

    if n_nan or n_inf:
        report.add(
            FAIL,
            "finite values",
            f"{n_nan} NaN and {n_inf} infinite feature values",
        )
    else:
        report.add(PASS, "finite values", "no NaN or infinite values")


def check_balance(dataset: pd.DataFrame, report: Report) -> None:
    labels = dataset[LABEL_NAME]

    unique = sorted(labels.unique().tolist())

    if set(unique) - {0, 1}:
        report.add(FAIL, "label domain", f"unexpected label values: {unique}")
        return

    if len(unique) < 2:
        report.add(
            FAIL,
            "label balance",
            f"label is constant at {unique[0]}, nothing to learn",
        )
        return

    rate = float(labels.mean())

    detail = f"{rate:.1%} positive ({int(labels.sum())}/{len(labels)})"

    # Below 5% or above 95% is learnable but needs care,
    # and the accuracy number stops meaning anything.
    if rate < 0.05 or rate > 0.95:
        report.add(WARN, "label balance", detail + " - severely imbalanced")
    else:
        report.add(PASS, "label balance", detail)


def check_per_snapshot_balance(
    dataset: pd.DataFrame, report: Report
) -> None:
    rates = dataset.groupby("snapshot_index")[LABEL_NAME].mean()

    constant = int(((rates == 0.0) | (rates == 1.0)).sum())

    detail = (
        f"per snapshot rate min {rates.min():.2f} "
        f"max {rates.max():.2f}, {constant} degenerate"
    )

    if constant > len(rates) * 0.5:
        report.add(FAIL, "snapshot balance", detail)
    elif constant:
        report.add(WARN, "snapshot balance", detail)
    else:
        report.add(PASS, "snapshot balance", detail)


def check_single_feature_separation(
    dataset: pd.DataFrame, report: Report
) -> None:
    """A lone feature scoring a perfect AUC means leakage.

    If one column can rank the label flawlessly, it is
    almost certainly derived from the future rather than
    the past.
    """
    from sklearn.metrics import roc_auc_score

    labels = dataset[LABEL_NAME].to_numpy()

    suspicious: list[str] = []
    strongest = ("", 0.5)

    for name in FEATURE_NAMES:
        column = dataset[name].to_numpy(dtype=float)

        if np.ptp(column) == 0:
            continue

        auc = roc_auc_score(labels, column)

        # Direction does not matter for leakage.
        auc = max(auc, 1.0 - auc)

        if auc > abs(strongest[1]):
            strongest = (name, auc)

        if auc >= 0.999:
            suspicious.append(f"{name} (AUC {auc:.4f})")

    if suspicious:
        report.add(
            FAIL,
            "leakage - single feature",
            f"near perfect separation by {', '.join(suspicious)}",
        )
    else:
        report.add(
            PASS,
            "leakage - single feature",
            f"strongest is {strongest[0]} at AUC {strongest[1]:.3f}",
        )


def check_constant_features(
    dataset: pd.DataFrame, report: Report
) -> None:
    constant = [
        name
        for name in FEATURE_NAMES
        if np.ptp(dataset[name].to_numpy(dtype=float)) == 0
    ]

    if constant:
        report.add(
            WARN,
            "constant features",
            f"carry no information: {constant}",
        )
    else:
        report.add(PASS, "constant features", "all features vary")


def check_causality(
    dataset: pd.DataFrame,
    log: pd.DataFrame,
    report: Report,
    *,
    samples: int = 4,
    rng_seed: int = 0,
) -> None:
    """Rebuild sampled rows from scratch and compare.

    This is the check that actually proves the features
    are causal. For a handful of snapshots the log is
    replayed independently, using only rows at or before
    the snapshot time, and the resulting feature vectors
    have to match the dataset exactly. If any future row
    had crept into the builder, the numbers would differ.
    """
    rng = np.random.default_rng(rng_seed)

    snapshot_ids = dataset["snapshot_index"].unique()

    chosen = rng.choice(
        snapshot_ids,
        size=min(samples, len(snapshot_ids)),
        replace=False,
    )

    timestamps = log["timestamp"].to_numpy(dtype=np.int64)
    keys = log["key"].to_numpy()
    operations = log["operation"].to_numpy()
    hits = log["hit"].to_numpy(na_value=None)
    sizes = log["value_size"].to_numpy(na_value=None)

    mismatches = 0
    compared = 0

    for snapshot_id in chosen:
        rows = dataset[dataset["snapshot_index"] == snapshot_id]

        if rows.empty:
            continue

        snapshot_ms = int(rows["snapshot_time"].iloc[0])

        replay = TrackerReplay()

        for index in range(len(log)):
            if int(timestamps[index]) > snapshot_ms:
                break
            replay.apply(
                str(keys[index]),
                str(operations[index]),
                int(timestamps[index]),
                hits[index],
                sizes[index],
            )

        for row in rows.itertuples(index=False):
            entry = replay.stats.get(row.key)

            if entry is None:
                mismatches += 1
                continue

            expected = entry.to_feature_vector(snapshot_ms)
            actual = [getattr(row, name) for name in FEATURE_NAMES]

            compared += 1

            if not np.allclose(expected, actual, rtol=1e-9, atol=1e-9):
                mismatches += 1

    if mismatches:
        report.add(
            FAIL,
            "leakage - causality",
            f"{mismatches}/{compared} rebuilt rows disagree",
        )
    else:
        report.add(
            PASS,
            "leakage - causality",
            f"{compared} rows rebuilt from past only rows match exactly",
        )


def check_label_matches_log(
    dataset: pd.DataFrame,
    log: pd.DataFrame,
    horizon_ms: int,
    report: Report,
    *,
    samples: int = 2000,
    rng_seed: int = 0,
) -> None:
    """Recompute sampled labels straight from the log."""
    rng = np.random.default_rng(rng_seed)

    key_times: dict[str, list[int]] = {}

    # Evictions are the cache acting on its own, not a
    # client coming back, so they are not reuse. This
    # has to match build_dataset exactly or the check
    # would report disagreements that are its own.
    requested = log["operation"].to_numpy() != "EVICT"

    for key, timestamp, is_request in zip(
        log["key"].to_numpy(),
        log["timestamp"].to_numpy(dtype=np.int64),
        requested,
        strict=True,
    ):
        if is_request:
            key_times.setdefault(str(key), []).append(int(timestamp))

    n = min(samples, len(dataset))

    indices = rng.choice(len(dataset), size=n, replace=False)

    sample = dataset.iloc[indices]

    mismatches = 0

    for row in sample.itertuples(index=False):
        times = key_times.get(row.key, [])
        snapshot_ms = int(row.snapshot_time)

        position = bisect_right(times, snapshot_ms)

        expected = int(
            position < len(times)
            and times[position] <= snapshot_ms + horizon_ms
        )

        if expected != int(getattr(row, LABEL_NAME)):
            mismatches += 1

    if mismatches:
        report.add(
            FAIL,
            "label correctness",
            f"{mismatches}/{n} sampled labels disagree with the log",
        )
    else:
        report.add(
            PASS,
            "label correctness",
            f"{n} sampled labels recomputed from the log match",
        )


def check_label_is_plausible(
    dataset: pd.DataFrame, report: Report
) -> None:
    """Hot keys should be reused more than cold ones.

    This is a sanity check on the workload rather than on
    the code, but a label that does not line up with the
    known key classes means something upstream is wrong.
    """
    prefixes = dataset["key"].str.split(":").str[0]

    if prefixes.nunique() < 2:
        report.add(
            WARN,
            "label plausibility",
            "log has no key class prefixes to compare",
        )
        return

    rates = dataset.groupby(prefixes)[LABEL_NAME].mean().sort_values()

    summary = ", ".join(
        f"{name} {rate:.2f}" for name, rate in rates.items()
    )

    if "hot" in rates.index and "cold" in rates.index:
        if rates["hot"] > rates["cold"]:
            report.add(PASS, "label plausibility", summary)
        else:
            report.add(
                FAIL,
                "label plausibility",
                f"hot keys are not reused more than cold ones: {summary}",
            )
    else:
        report.add(WARN, "label plausibility", summary)


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the MiniRedis eviction dataset"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--log", default=str(DEFAULT_WORKLOAD_LOG))
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=None,
        help="horizon the dataset was built with; inferred if omitted",
    )

    args = parser.parse_args(argv)

    dataset = pd.read_csv(args.dataset)
    log = load_access_log(args.log)

    horizon_ms = resolve_horizon_ms(log, args.horizon_seconds)

    report = Report()

    check_schema(dataset, report)
    check_finite(dataset, report)
    check_balance(dataset, report)
    check_per_snapshot_balance(dataset, report)
    check_constant_features(dataset, report)
    check_single_feature_separation(dataset, report)
    check_causality(dataset, log, report)
    check_label_matches_log(dataset, log, horizon_ms, report)
    check_label_is_plausible(dataset, report)

    print(f"Validating {args.dataset}")
    print(f"  {len(dataset)} rows, horizon {horizon_ms / 1000:.3f} s\n")
    print(report.render())

    if report.failed:
        print("\nVALIDATION FAILED")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
