"""Tests for the FastAPI model service.

The service is the only thing standing between a trained
model and a cache making real decisions, so what is
tested here is mostly what it refuses. Scoring a request
it does not fully understand would produce a confident
number with no meaning behind it, and the cache has no
way to tell that apart from a good one.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.config import FEATURE_NAMES
from ml.service import ModelBundle, app, bundle


# ---------------------------------------------------------------
# A stand in model
# ---------------------------------------------------------------


class StubEstimator:
    """Scores on one feature, so the ranking is knowable.

    Using a real trained model here would make these
    tests depend on a file that may not exist and on
    predictions that may change; what is under test is
    the service, not the estimator.
    """

    def __init__(self, feature_index: int) -> None:
        self.feature_index = feature_index

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        raw = np.asarray(matrix, dtype=float)[:, self.feature_index]

        # Squash into (0, 1) without assuming a range.
        positive = 1.0 / (1.0 + np.exp(-raw))

        return np.column_stack([1.0 - positive, positive])


@contextmanager
def serving(estimator, feature_names, metadata, error):
    """Run a client against a bundle we control.

    The bundle has to be set after entering the client,
    not before: starting the app runs the lifespan, which
    loads the real model file and would overwrite
    whatever the test had put there.
    """
    with TestClient(app) as test_client:
        saved = (
            bundle.estimator,
            bundle.feature_names,
            bundle.metadata,
            bundle.error,
        )

        bundle.estimator = estimator
        bundle.feature_names = list(feature_names)
        bundle.metadata = metadata
        bundle.error = error

        try:
            yield test_client

        finally:
            (
                bundle.estimator,
                bundle.feature_names,
                bundle.metadata,
                bundle.error,
            ) = saved


@pytest.fixture
def client():
    """A client with a loaded stub model."""
    with serving(
        StubEstimator(FEATURE_NAMES.index("total_accesses")),
        FEATURE_NAMES,
        {"model_name": "stub"},
        None,
    ) as test_client:
        yield test_client


@pytest.fixture
def empty_client():
    """A client with no model loaded at all."""
    with serving(None, [], {}, "no model for this test") as test_client:
        yield test_client


def features(**overrides: float) -> dict[str, float]:
    values = {name: 1.0 for name in FEATURE_NAMES}
    values.update(overrides)
    return values


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------


def test_health_is_200_even_without_a_model(empty_client):
    """The Java client cannot tell a connection error
    apart from a service that is up but untrained, so the
    service says which it is instead of failing."""
    response = empty_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()

    assert body["model_loaded"] is True
    assert body["model_name"] == "stub"
    assert body["feature_count"] == len(FEATURE_NAMES)


# ---------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------


def test_predict_without_a_model_is_503(empty_client):
    response = empty_client.post(
        "/predict",
        json={"candidates": [{"key": "a", "features": features()}]},
    )

    assert response.status_code == 503


def test_missing_features_are_rejected(client):
    """Zero filling would be the friendly thing to do and
    the wrong one: the model would score the key as if it
    had never been read."""
    incomplete = features()
    incomplete.pop(FEATURE_NAMES[0])

    response = client.post(
        "/predict",
        json={"candidates": [{"key": "a", "features": incomplete}]},
    )

    assert response.status_code == 422
    assert FEATURE_NAMES[0] in response.text


def test_unknown_features_are_rejected(client):
    response = client.post(
        "/predict",
        json={
            "candidates": [
                {"key": "a", "features": features(invented=1.0)}
            ]
        },
    )

    assert response.status_code == 422
    assert "invented" in response.text


def test_a_short_vector_is_rejected(client):
    response = client.post(
        "/predict",
        json={"candidates": [{"key": "a", "vector": [1.0, 2.0]}]},
    )

    assert response.status_code == 422


def test_neither_features_nor_vector_is_rejected(client):
    response = client.post(
        "/predict", json={"candidates": [{"key": "a"}]}
    )

    assert response.status_code == 422


def test_no_candidates_is_rejected(client):
    response = client.post("/predict", json={"candidates": []})

    assert response.status_code == 422


# ---------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------


def test_predict_returns_one_score_per_key(client):
    response = client.post(
        "/predict",
        json={
            "candidates": [
                {"key": "a", "features": features(total_accesses=1)},
                {"key": "b", "features": features(total_accesses=50)},
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert [p["key"] for p in body["predictions"]] == ["a", "b"]

    assert all(
        0.0 <= p["reuse_probability"] <= 1.0
        for p in body["predictions"]
    )


def test_a_vector_and_named_features_agree(client):
    """Both request forms exist, so they must produce the
    same score. Otherwise the choice of form silently
    changes the eviction decision."""
    values = features(total_accesses=7)

    by_name = client.post(
        "/predict",
        json={"candidates": [{"key": "a", "features": values}]},
    ).json()

    by_vector = client.post(
        "/predict",
        json={
            "candidates": [
                {
                    "key": "a",
                    "vector": [values[name] for name in FEATURE_NAMES],
                }
            ]
        },
    ).json()

    assert by_name["predictions"][0]["reuse_probability"] == (
        by_vector["predictions"][0]["reuse_probability"]
    )


def test_feature_order_does_not_matter(client):
    """Named features are sent as a JSON object, and
    nothing guarantees the order they arrive in."""
    values = features(total_accesses=9)

    forward = client.post(
        "/predict",
        json={"candidates": [{"key": "a", "features": values}]},
    ).json()

    reversed_values = dict(reversed(list(values.items())))

    backward = client.post(
        "/predict",
        json={
            "candidates": [{"key": "a", "features": reversed_values}]
        },
    ).json()

    assert forward["predictions"][0]["reuse_probability"] == (
        backward["predictions"][0]["reuse_probability"]
    )


# ---------------------------------------------------------------
# Evicting
# ---------------------------------------------------------------


def test_evict_returns_the_lowest_scoring_keys(client):
    response = client.post(
        "/evict",
        json={
            "count": 2,
            "candidates": [
                {"key": "hot", "features": features(total_accesses=100)},
                {"key": "warm", "features": features(total_accesses=10)},
                {"key": "cold", "features": features(total_accesses=1)},
                {
                    "key": "frozen",
                    "features": features(total_accesses=-50),
                },
            ],
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["victims"] == ["frozen", "cold"]

    # Every candidate is scored, not only the victims,
    # so the caller can log or override the decision.
    assert len(body["scores"]) == 4


def test_asking_for_more_victims_than_candidates(client):
    response = client.post(
        "/evict",
        json={
            "count": 10,
            "candidates": [{"key": "only", "features": features()}],
        },
    )

    assert response.status_code == 200
    assert response.json()["victims"] == ["only"]


def test_evict_without_a_model_is_503(empty_client):
    response = empty_client.post(
        "/evict",
        json={
            "count": 1,
            "candidates": [{"key": "a", "features": features()}],
        },
    )

    assert response.status_code == 503


def test_non_finite_features_are_rejected(client):
    """NaN reaches the service as JSON null, which fails
    validation, but an infinity can arrive as a huge
    float and would otherwise be scored."""
    # Sent as raw text, because no JSON encoder will
    # emit an infinity - but a decoder will happily read
    # one back out of a large enough literal.
    vector = ", ".join(["1e400"] * len(FEATURE_NAMES))

    response = client.post(
        "/predict",
        content=(
            '{"candidates": [{"key": "a", "vector": ['
            + vector
            + "]}]}"
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------


def test_a_missing_model_file_is_not_fatal(tmp_path):
    """The service must start without a model so the
    cache can fall back rather than fail to boot."""
    fresh = ModelBundle()

    fresh.load(tmp_path / "does-not-exist.joblib")

    assert fresh.ready is False
    assert "not found" in (fresh.error or "")
