"""Model-serving API.

Endpoints:
    GET  /health                  liveness + whether a model is loaded
    POST /reload                  load a newly trained model without restarting
    GET  /model-info              training metadata (metrics, thresholds, SHAP importance)
    POST /predict                 score a batch of transactions (?explain=true for SHAP reasons)
    GET  /examples                sample of real validation transactions (review queue)
    GET  /examples/{id}           score + explain one example, with its true label
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.schemas import (
    ExampleDetail,
    ExampleSummary,
    ModelInfo,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from fraud_detection import routing
from fraud_detection.explain import top_reasons

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
STATE: dict = {}


def load_artifacts() -> None:
    STATE.clear()
    model_path = MODEL_DIR / "model.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    examples_path = MODEL_DIR / "examples.parquet"
    if model_path.exists():
        STATE.update(joblib.load(model_path))
        STATE["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if examples_path.exists():
            ex = pd.read_parquet(examples_path)
            STATE["examples"] = ex.set_index("TransactionID", drop=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Fraud Detection API", version="0.2.0", lifespan=lifespan)


def _require_model() -> None:
    if "model" not in STATE:
        raise HTTPException(503, "No model loaded. Run the trainer service first.")


def _score(df: pd.DataFrame, threshold: float | None, explain: bool) -> PredictResponse:
    fb, model = STATE["features"], STATE["model"]
    X = fb.transform(df)
    probs = routing.predict(model, fb, df, X)
    thr = threshold if threshold is not None else STATE["metadata"].get("threshold", 0.5)
    # Reasons use the scoring model's SHAP values, shown with the original (readable) values.
    reasons = top_reasons(routing.contributions(model, fb, df, X), X) if explain else [None] * len(df)
    ids = df["TransactionID"].tolist() if "TransactionID" in df else [None] * len(df)
    return PredictResponse(
        threshold=thr,
        predictions=[
            Prediction(
                TransactionID=None if i is None or pd.isna(i) else int(i),
                fraud_probability=round(float(p), 6),
                flagged=bool(p >= thr),
                reasons=r,
            )
            for i, p, r in zip(ids, probs, reasons)
        ],
    )


def _score_features(X: pd.DataFrame, seen, ids) -> PredictResponse:
    """Score precomputed feature rows (review-queue examples, as scored in training)."""
    model = STATE["model"]
    probs = routing.predict_with_seen(model, X, seen)
    thr = STATE["metadata"].get("threshold", 0.5)
    reasons = top_reasons(routing.contributions_with_seen(model, X, seen), X)
    return PredictResponse(threshold=thr, predictions=[
        Prediction(TransactionID=int(i), fraud_probability=round(float(p), 6),
                   flagged=bool(p >= thr), reasons=r)
        for i, p, r in zip(ids, probs, reasons)])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/reload")
def reload() -> dict:
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
    explain: bool = Query(False, description="include top SHAP reasons per transaction"),
) -> PredictResponse:
    _require_model()
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    return _score(df, threshold, explain)


@app.get("/examples", response_model=list[ExampleSummary])
def examples() -> list[ExampleSummary]:
    _require_model()
    ex = STATE.get("examples")
    if ex is None:
        return []
    return [
        ExampleSummary(
            TransactionID=int(r.TransactionID),
            amount=None if pd.isna(r.TransactionAmt) else float(r.TransactionAmt),
            product=None if pd.isna(r.ProductCD) else str(r.ProductCD),
            is_fraud=bool(r.isFraud),
            outcome=r.outcome,
            score=float(r.score),
        )
        for r in ex.itertuples(index=False)
    ]


@app.get("/examples/{transaction_id}", response_model=ExampleDetail)
def example_detail(transaction_id: int) -> ExampleDetail:
    _require_model()
    ex = STATE.get("examples")
    if ex is None or transaction_id not in ex.index:
        raise HTTPException(404, f"Example {transaction_id} not found")
    row = ex.loc[[transaction_id]]
    if "example_features" in STATE and transaction_id in STATE["example_features"].index:
        pred = _score_features(STATE["example_features"].loc[[transaction_id]],
                               STATE["example_seen"].loc[[transaction_id]].to_numpy(),
                               [transaction_id])
    else:  # models trained before in-context examples were saved
        pred = _score(row.drop(columns=["isFraud", "outcome", "score"]), None, explain=True)
    raw = row.iloc[0].drop(["isFraud", "outcome", "score"])
    shown = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "DeviceInfo"]
    return ExampleDetail(
        prediction=pred.predictions[0],
        threshold=pred.threshold,
        is_fraud=bool(row["isFraud"].iloc[0]),
        outcome=str(row["outcome"].iloc[0]),
        details={c: (None if pd.isna(raw[c]) else (raw[c].item() if isinstance(raw[c], np.generic) else raw[c]))
                 for c in shown if c in raw.index},
    )
