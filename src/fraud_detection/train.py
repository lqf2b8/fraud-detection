"""Train a baseline LightGBM fraud model with a time-based validation split.

Usage:
    python -m fraud_detection.train              # real IEEE-CIS data in data/raw
    python -m fraud_detection.train --synthetic  # synthetic data, no download needed
"""

import argparse
import json
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_detection import config
from fraud_detection.data import load_raw, make_synthetic
from fraud_detection.evaluate import evaluate, pr_curve_points
from fraud_detection.features import FeatureBuilder, make_uid


def time_split(df: pd.DataFrame, valid_frac: float = config.VALID_FRACTION):
    """Train on the past, validate on the future, the way the model will be used."""
    df = df.sort_values(config.TIME_COL)
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut], df.iloc[cut:]


FEATURE_SET = "v1-entity"


def segment_metrics(fb: FeatureBuilder, valid_df: pd.DataFrame, y_va, scores) -> dict:
    """Metrics for customers seen in training vs. new customers (the EDA found new ones riskier)."""
    uid = make_uid(valid_df)
    seen_idx = fb.history.get("uid", pd.DataFrame()).index
    segs = {"seen_customer": uid.isin(seen_idx).fillna(False).to_numpy(dtype=bool)}
    segs["new_customer"] = ~segs["seen_customer"]
    out = {}
    for name, mask in segs.items():
        y = np.asarray(y_va)[mask]
        if mask.sum() and 0 < y.sum() < len(y):
            m = evaluate(y, np.asarray(scores)[mask])
            out[name] = {"share": round(float(mask.mean()), 4),
                         **{k: m[k] for k in ("pr_auc", "roc_auc", "base_rate")}}
    return out


def train(df: pd.DataFrame, data_source: str) -> dict:
    train_df, valid_df = time_split(df)

    # fit_transform gives training rows past-only entity features; transform then scores the
    # validation period against the history accumulated in training (no look-ahead).
    fb = FeatureBuilder()
    X_tr, y_tr = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    scores = model.predict_proba(X_va)[:, 1]
    metrics = evaluate(y_va, scores)
    segments = segment_metrics(fb, valid_df, y_va, scores)

    importance = (
        pd.Series(model.booster_.feature_importance("gain"), index=fb.feature_names)
        .sort_values(ascending=False)
        .head(25)
    )

    metadata = {
        "model_type": "LightGBM",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_source": data_source,
        "n_train": len(train_df),
        "n_features": len(fb.feature_names),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "feature_set": FEATURE_SET,
        "metrics": metrics,
        "segments": segments,
        "threshold": metrics["best_f1_threshold"],
        "feature_importance": {k: round(float(v), 2) for k, v in importance.items()},
        "pr_curve": pr_curve_points(y_va, scores),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": fb}, config.MODEL_PATH)
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data")
    parser.add_argument("--nrows", type=int, default=None, help="limit rows for quick runs")
    args = parser.parse_args()

    if args.synthetic:
        df, source = make_synthetic(), "synthetic"
    else:
        df, source = load_raw(nrows=args.nrows), "ieee-cis"

    meta = train(df, source)
    print(json.dumps({"metrics": meta["metrics"], "segments": meta["segments"]}, indent=2))
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
