"""Model-serving API.

Endpoints:
    GET  /health      liveness + whether a model is loaded
    GET  /model-info  training metadata (metrics, PR curve, feature importance)
    POST /predict     score a batch of transactions
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.schemas import ModelInfo, Prediction, PredictRequest, PredictResponse

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
STATE: dict = {}


def load_artifacts() -> None:
    STATE.clear()
    model_path, meta_path = MODEL_DIR / "model.joblib", MODEL_DIR / "metadata.json"
    if model_path.exists():
        STATE.update(joblib.load(model_path))
        STATE["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)


def _require_model() -> None:
    if "model" not in STATE:
        raise HTTPException(503, "No model loaded. Run the trainer service first.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/reload")
def reload() -> dict:
    """Pick up a newly trained model without restarting the container."""
    load_artifacts()
    return health()


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    _require_model()
    return ModelInfo(metadata=STATE["metadata"])


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    threshold: float | None = Query(None, ge=0, le=1, description="override decision threshold"),
) -> PredictResponse:
    _require_model()
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    X = STATE["features"].transform(df)
    probs = STATE["model"].predict_proba(X)[:, 1]
    thr = threshold if threshold is not None else STATE["metadata"].get("threshold", 0.5)
    ids = df["TransactionID"].tolist()
    return PredictResponse(
        threshold=thr,
        predictions=[
            Prediction(
                TransactionID=None if pd.isna(i) else int(i),
                fraud_probability=round(float(p), 6),
                flagged=bool(p >= thr),
            )
            for i, p in zip(ids, probs)
        ],
    )
