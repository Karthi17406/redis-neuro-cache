"""FastAPI service exposing the key reuse model.

    python -m ml.service

The Java side calls this to decide what to evict. Two
design choices matter for that caller:

**Features may be sent by name or as a vector.** By name
is the safe form and the one MLClient.java uses, because
it cannot silently misalign if the feature list changes.
The ordered vector is accepted too, for callers that
already build one.

**The service never guesses.** A request naming a
feature the model was not trained on, or omitting one,
is rejected rather than defaulted to zero. A silently
zero filled feature produces a confident, wrong score,
which is worse for a cache than no score at all.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .config import DEFAULT_HOST, DEFAULT_METRICS, DEFAULT_MODEL, DEFAULT_PORT

logger = logging.getLogger("miniredis.ml")


# ---------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------


class ModelBundle:
    """Holds the loaded model, or explains why it is not."""

    def __init__(self) -> None:
        self.estimator: Any = None
        self.feature_names: list[str] = []
        self.metadata: dict = {}
        self.error: str | None = None
        self.path: Path | None = None

    @property
    def ready(self) -> bool:
        return self.estimator is not None

    def load(self, path: str | Path = DEFAULT_MODEL) -> None:
        import joblib

        path = Path(path)
        self.path = path

        if not path.exists():
            self.estimator = None
            self.error = (
                f"Model file not found at {path}. "
                "Run: python -m ml.train"
            )
            logger.warning(self.error)
            return

        try:
            artifact = joblib.load(path)

            self.estimator = artifact["estimator"]
            self.feature_names = list(artifact["feature_names"])
            self.metadata = {
                key: value
                for key, value in artifact.items()
                if key != "estimator"
            }
            self.error = None

            logger.info(
                "Loaded %s with %d features from %s",
                artifact.get("model_name", "model"),
                len(self.feature_names),
                path,
            )

        except Exception as exc:  # noqa: BLE001 - reported to caller
            self.estimator = None
            self.error = f"Failed to load model from {path}: {exc}"
            logger.exception(self.error)

    def vectorise(self, item: CandidateFeatures) -> list[float]:
        """Turn one request item into an ordered vector."""
        if item.vector is not None:
            if len(item.vector) != len(self.feature_names):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"vector has {len(item.vector)} values but the "
                        f"model expects {len(self.feature_names)}: "
                        f"{self.feature_names}"
                    ),
                )
            return [float(value) for value in item.vector]

        if not item.features:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="provide either 'features' or 'vector'",
            )

        missing = [
            name for name in self.feature_names if name not in item.features
        ]

        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"missing features: {missing}",
            )

        unknown = [
            name for name in item.features if name not in self.feature_names
        ]

        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"unknown features: {unknown}. "
                    f"expected: {self.feature_names}"
                ),
            )

        return [float(item.features[name]) for name in self.feature_names]

    def predict(self, vectors: list[list[float]]) -> np.ndarray:
        if not self.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=self.error or "model not loaded",
            )

        matrix = np.asarray(vectors, dtype=float)

        if not np.isfinite(matrix).all():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features contain NaN or infinite values",
            )

        return self.estimator.predict_proba(matrix)[:, 1]


bundle = ModelBundle()


# ---------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------


class CandidateFeatures(BaseModel):
    key: str = Field(default="", description="cache key, for the response")
    features: dict[str, float] | None = Field(
        default=None, description="feature name to value"
    )
    vector: list[float] | None = Field(
        default=None, description="features in MLFeatures.NAMES order"
    )


class PredictRequest(BaseModel):
    candidates: list[CandidateFeatures] = Field(min_length=1)


class Prediction(BaseModel):
    key: str
    reuse_probability: float


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[Prediction]


class EvictRequest(BaseModel):
    candidates: list[CandidateFeatures] = Field(min_length=1)
    count: int = Field(default=1, ge=1, description="victims to return")


class EvictResponse(BaseModel):
    model_name: str
    victims: list[str]
    scores: list[Prediction]


# ---------------------------------------------------------------
# Application
# ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once, when the service starts.

    A missing model is not fatal here. The service stays
    up and reports it through /health, so the Java client
    can fall back to LRU rather than crash on startup.
    """
    logging.basicConfig(level=logging.INFO)
    bundle.load()
    yield


app = FastAPI(
    title="MiniRedis eviction model",
    description=(
        "Predicts whether a cache key will be read again soon, "
        "so the least useful key can be evicted."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness plus whether a model is actually loaded.

    Returns 200 either way. The Java client reads the
    ``model_loaded`` flag and falls back to LRU when it
    is false, which is more useful than a connection
    error it cannot distinguish from the service being
    down entirely.
    """
    return {
        "status": "ok",
        "model_loaded": bundle.ready,
        "model_name": bundle.metadata.get("model_name"),
        "feature_count": len(bundle.feature_names),
        "error": bundle.error,
    }


@app.get("/model_info")
def model_info() -> dict:
    if not bundle.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=bundle.error or "model not loaded",
        )

    metrics_path = Path(DEFAULT_METRICS)

    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else None
    )

    return {
        "model_name": bundle.metadata.get("model_name"),
        "feature_names": bundle.feature_names,
        "trained_at": bundle.metadata.get("trained_at"),
        "dataset_settings": bundle.metadata.get("dataset_settings"),
        "test_metrics": bundle.metadata.get("test_metrics"),
        "model_path": str(bundle.path),
        "metrics": metrics,
    }


@app.post("/reload")
def reload_model() -> dict:
    """Pick up a newly trained model without a restart."""
    bundle.load()

    if not bundle.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=bundle.error or "model not loaded",
        )

    return {"status": "reloaded", "model_name": bundle.metadata.get("model_name")}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not bundle.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=bundle.error or "model not loaded",
        )

    vectors = [bundle.vectorise(item) for item in request.candidates]

    probabilities = bundle.predict(vectors)

    return PredictResponse(
        model_name=str(bundle.metadata.get("model_name")),
        predictions=[
            Prediction(
                key=item.key or f"#{index}",
                reuse_probability=float(probability),
            )
            for index, (item, probability) in enumerate(
                zip(request.candidates, probabilities, strict=True)
            )
        ],
    )


@app.post("/evict", response_model=EvictResponse)
def evict(request: EvictRequest) -> EvictResponse:
    """Rank candidates and name the ones to drop.

    The victims are the keys least likely to be read
    again. Returning the full score list as well lets the
    caller log or override the decision.
    """
    if not bundle.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=bundle.error or "model not loaded",
        )

    vectors = [bundle.vectorise(item) for item in request.candidates]

    probabilities = bundle.predict(vectors)

    scored = [
        Prediction(
            key=item.key or f"#{index}",
            reuse_probability=float(probability),
        )
        for index, (item, probability) in enumerate(
            zip(request.candidates, probabilities, strict=True)
        )
    ]

    ranked = sorted(scored, key=lambda p: p.reuse_probability)

    count = min(request.count, len(ranked))

    return EvictResponse(
        model_name=str(bundle.metadata.get("model_name")),
        victims=[p.key for p in ranked[:count]],
        scores=scored,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Serve the MiniRedis eviction model"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    args = parser.parse_args(argv)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
