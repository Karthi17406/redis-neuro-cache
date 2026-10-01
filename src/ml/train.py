"""Train and evaluate the key reuse model.

    python -m ml.train

Three things here are worth more than the model choice
itself.

**The split is chronological.** Rows from one snapshot
are heavily correlated, and the task is to predict the
future from the past, so a shuffled split would both
leak and flatter. Snapshots are cut into train,
validation and test blocks in time order. Selection
happens on validation, and the test block is touched
once, at the end.

**The baselines are the real ones.** Beating a coin flip
proves nothing; the question is whether the model beats
the eviction policies a cache would otherwise use. LRU
and LFU are scored here as rankers, on the same rows and
the same metric as the model.

**Ranking quality is what matters.** The eviction path
sorts keys by predicted reuse probability and drops the
lowest. ROC AUC and average precision describe that job.
Accuracy at a 0.5 threshold does not, so it is reported
but never used to choose.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    DEFAULT_DATASET,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    DEFAULT_TEST_FRACTION,
    FEATURE_NAMES,
    LABEL_NAME,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the snapshots into time ordered blocks.

    Splitting on snapshot rather than on row keeps every
    row of a snapshot on the same side of the boundary,
    so the model cannot see one key at a moment in time
    and then be tested on its neighbour at that same
    moment.
    """
    snapshots = np.sort(dataset["snapshot_index"].unique())

    n = len(snapshots)

    if n < 3:
        raise ValueError(
            f"Need at least 3 snapshots to split, found {n}"
        )

    n_test = max(1, int(round(n * test_fraction)))
    n_validation = max(1, int(round(n * validation_fraction)))

    if n_test + n_validation >= n:
        raise ValueError(
            "Validation and test fractions leave no training data"
        )

    train_ids = snapshots[: n - n_validation - n_test]
    validation_ids = snapshots[n - n_validation - n_test : n - n_test]
    test_ids = snapshots[n - n_test :]

    return (
        dataset[dataset["snapshot_index"].isin(train_ids)],
        dataset[dataset["snapshot_index"].isin(validation_ids)],
        dataset[dataset["snapshot_index"].isin(test_ids)],
    )


# ---------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------


def candidate_models() -> dict[str, Pipeline]:
    """The models to compare.

    Logistic regression is the designated baseline: it is
    cheap to score, its coefficients can be read, and it
    produces calibrated probabilities, which is what the
    eviction ranking wants. The trees are here to show
    what, if anything, the extra capacity buys.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------


def heuristic_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score each row the way a classic cache policy would.

    Higher means more likely to be reused, so that these
    can be fed to the same metrics as the model.
    """
    return {
        # LRU keeps whatever was touched most recently.
        "baseline_lru": -frame["recency_seconds"].to_numpy(dtype=float),
        # LFU keeps whatever is touched most often.
        "baseline_lfu": frame["accesses_per_minute"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------


def ranking_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Metrics that need only a ranking, not a threshold."""
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def full_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }

    metrics.update(ranking_metrics(y_true, probabilities))

    return metrics


# Eviction depths to report regret at.
#
# These have to straddle the number of keys that are
# genuinely dead at any moment, which on this workload is
# around 97 of ~202 resident keys. Below that count every
# policy scores a perfect zero, because there are more
# obvious victims than slots to fill, and the metric says
# nothing. The differences only show up once the cache is
# forced to evict keys that are actually in question.
EVICTION_K_VALUES = (50, 100, 150)


def eviction_regret(
    frame: pd.DataFrame,
    scores: np.ndarray,
    victims_per_snapshot: int = 5,
    *,
    rng_seed: int = RANDOM_STATE,
) -> float:
    """Share of evicted keys that were about to be reused.

    This is the metric the cache actually feels. At every
    snapshot the lowest scoring keys are dropped, and the
    fraction of those that turn out to be reused within
    the horizon is the mistake rate. Lower is better.

    Ties are broken at random rather than by row order.
    Without that, a policy that scores every key the same
    is judged on the order rows happen to sit in, which
    says nothing about the policy.
    """
    work = frame[["snapshot_index", LABEL_NAME]].copy()
    work["score"] = scores

    # Deterministic jitter, far below any real score gap.
    rng = np.random.default_rng(rng_seed)
    work["tiebreak"] = rng.random(len(work))

    mistakes = 0
    evicted = 0

    for _, group in work.groupby("snapshot_index", sort=False):
        k = min(victims_per_snapshot, len(group))

        victims = group.nsmallest(k, ["score", "tiebreak"])

        mistakes += int(victims[LABEL_NAME].sum())
        evicted += len(victims)

    return float(mistakes / evicted) if evicted else 0.0


def eviction_regret_curve(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, float]:
    """Regret at several eviction depths.

    A single depth is misleading. With five victims every
    policy looks perfect, because the obvious one shot
    keys absorb all of them. The differences only appear
    once the cache is under enough pressure to evict keys
    that are genuinely in question.
    """
    return {
        f"eviction_regret_at_{k}": eviction_regret(
            frame, scores, victims_per_snapshot=k
        )
        for k in EVICTION_K_VALUES
    }


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------


def train_and_select(
    dataset: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict:
    train, validation, test = chronological_split(
        dataset, test_fraction=test_fraction
    )

    x_train = train[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train[LABEL_NAME].to_numpy(dtype=int)

    x_validation = validation[FEATURE_NAMES].to_numpy(dtype=float)
    y_validation = validation[LABEL_NAME].to_numpy(dtype=int)

    x_test = test[FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test[LABEL_NAME].to_numpy(dtype=int)

    print(
        f"Split: {len(train)} train / {len(validation)} validation "
        f"/ {len(test)} test rows"
    )
    print(
        f"       {train['snapshot_index'].nunique()} / "
        f"{validation['snapshot_index'].nunique()} / "
        f"{test['snapshot_index'].nunique()} snapshots, in time order"
    )

    # -----------------------------------------------------------
    # Baselines, on the test block only
    # -----------------------------------------------------------

    baselines: dict[str, dict] = {}

    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    dummy.fit(x_train, y_train)

    dummy_probabilities = dummy.predict_proba(x_test)[:, 1]

    baselines["baseline_always_keep"] = {
        **full_metrics(y_test, dummy_probabilities),
        **eviction_regret_curve(test, dummy_probabilities),
    }

    for name, scores in heuristic_scores(test).items():
        baselines[name] = {
            **ranking_metrics(y_test, scores),
            **eviction_regret_curve(test, scores),
        }

    # -----------------------------------------------------------
    # Candidates, selected on validation
    # -----------------------------------------------------------

    validation_scores: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}

    for name, pipeline in candidate_models().items():
        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(x_validation)[:, 1]

        auc = float(roc_auc_score(y_validation, probabilities))

        validation_scores[name] = auc
        fitted[name] = pipeline

        print(f"  {name:<20} validation ROC AUC {auc:.4f}")

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = fitted[best_name]

    print(f"\nSelected on validation: {best_name}")

    # -----------------------------------------------------------
    # Final report, on the untouched test block
    # -----------------------------------------------------------

    test_report: dict[str, dict] = {}

    for name, pipeline in fitted.items():
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        test_report[name] = {
            **full_metrics(y_test, probabilities),
            **eviction_regret_curve(test, probabilities),
        }

    return {
        "selected_model": best_name,
        "selected_estimator": best_model,
        "validation_roc_auc": validation_scores,
        "baselines": baselines,
        "test": test_report,
        "split_sizes": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
    }


def feature_importance(pipeline: Pipeline) -> dict[str, float] | None:
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
    else:
        return None

    return {
        name: float(value)
        for name, value in sorted(
            zip(FEATURE_NAMES, values, strict=True),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
    }


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------


def print_comparison(result: dict) -> None:
    def row_for(name: str, metrics: dict) -> tuple[str, ...]:
        return (
            name,
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['average_precision']:.4f}",
            f"{metrics['f1']:.4f}" if "f1" in metrics else "-",
            *(
                f"{metrics[f'eviction_regret_at_{k}']:.4f}"
                for k in EVICTION_K_VALUES
            ),
        )

    rows = [
        row_for(name, metrics)
        for name, metrics in result["baselines"].items()
    ]

    rows += [
        row_for(
            name + (" *" if name == result["selected_model"] else ""),
            metrics,
        )
        for name, metrics in result["test"].items()
    ]

    header = (
        "model",
        "ROC AUC",
        "avg prec",
        "F1",
        *(f"regret@{k}" for k in EVICTION_K_VALUES),
    )

    widths = [
        max(len(header[i]), max(len(row[i]) for row in rows))
        for i in range(len(header))
    ]

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        )

    print("\nTest block results (chronologically last snapshots)")
    print("  " + render(header))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))

    for row in rows:
        print("  " + render(row))

    print(
        "\n  regret@k = share of the k lowest ranked keys per snapshot"
    )
    print(
        "             that were in fact reused. Lower is better."
    )


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the MiniRedis key reuse model"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS))
    parser.add_argument(
        "--test-fraction", type=float, default=DEFAULT_TEST_FRACTION
    )

    args = parser.parse_args(argv)

    import joblib

    dataset = pd.read_csv(args.dataset)

    missing = [
        column
        for column in FEATURE_NAMES + [LABEL_NAME]
        if column not in dataset.columns
    ]

    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    print(f"Training on {len(dataset)} rows from {args.dataset}\n")

    result = train_and_select(dataset, test_fraction=args.test_fraction)

    print_comparison(result)

    best_name = result["selected_model"]
    best_model = result["selected_estimator"]

    importance = feature_importance(best_model)

    if importance:
        print(f"\nFeature influence for {best_name}")
        for name, value in list(importance.items())[:6]:
            print(f"  {name:<28} {value:+.4f}")

    # -----------------------------------------------------------
    # Persist
    # -----------------------------------------------------------

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    settings_path = Path(args.dataset).with_suffix(".settings.json")

    dataset_settings = (
        json.loads(settings_path.read_text(encoding="utf-8"))
        if settings_path.exists()
        else {}
    )

    artifact = {
        "estimator": best_model,
        "feature_names": list(FEATURE_NAMES),
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_settings": dataset_settings,
        "test_metrics": result["test"][best_name],
        "python_version": platform.python_version(),
    }

    joblib.dump(artifact, model_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "trained_at": artifact["trained_at"],
                "dataset_settings": dataset_settings,
                "split_sizes": result["split_sizes"],
                "validation_roc_auc": result["validation_roc_auc"],
                "baselines": result["baselines"],
                "test": result["test"],
                "feature_importance": importance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nModel written to {model_path}")
    print(f"Metrics written to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
