"""Train a LightGBM fraud model with a time-based validation split.

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
from fraud_detection.evaluate import (
    cost_optimal_point,
    evaluate,
    pr_curve_points,
    threshold_table,
)
from fraud_detection.explain import contributions, describe, global_importance
from fraud_detection.features import FeatureBuilder, make_uid


def time_split(df: pd.DataFrame, valid_frac: float = config.VALID_FRACTION):
    """Train on the past, validate on the future, the way the model will be used."""
    df = df.sort_values(config.TIME_COL)
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut], df.iloc[cut:]


FEATURE_SET = "v3.1-known-fraud"


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


def pick_examples(valid_df: pd.DataFrame, scores, threshold: float, per_group: int = 25) -> pd.DataFrame:
    """A small, honest sample of validation transactions for the dashboard's review queue:
    caught frauds, missed frauds, false alarms and correctly-cleared legit transactions."""
    y = valid_df[config.TARGET].to_numpy()
    flagged = np.asarray(scores) >= threshold
    outcome = np.select(
        [flagged & (y == 1), ~flagged & (y == 1), flagged & (y == 0)],
        ["caught fraud", "missed fraud", "false alarm"],
        default="cleared legit",
    )
    df = valid_df.assign(outcome=outcome, score=np.round(scores, 6))
    idx = [
        i
        for _, g in df.groupby("outcome")
        for i in g.sample(min(len(g), per_group), random_state=config.RANDOM_STATE).index
    ]
    return df.loc[idx].reset_index(drop=True)


def fit_model(X_all: pd.DataFrame, y_all: pd.Series, seed: int):
    """Choose the tree count by early stopping on the most recent slice of the training period,
    then refit on all training data. Validation is never used to make any modeling choice."""
    es_cut = int(len(X_all) * (1 - config.EARLY_STOP_FRACTION))
    X_tr, y_tr = X_all.iloc[:es_cut], y_all.iloc[:es_cut]
    X_es, y_es = X_all.iloc[es_cut:], y_all.iloc[es_cut:]
    params = {
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "random_state": seed,
        "verbose": -1,
    }
    selector = lgb.LGBMClassifier(n_estimators=2000, **params)
    selector.fit(
        X_tr,
        y_tr,
        eval_set=[(X_es, y_es)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iter = int(selector.best_iteration_ or selector.n_estimators)
    n_trees = max(1, round(best_iter / (1 - config.EARLY_STOP_FRACTION)))
    model = lgb.LGBMClassifier(n_estimators=n_trees, **params)
    model.fit(X_all, y_all)
    return model, best_iter, n_trees


def train(df: pd.DataFrame, data_source: str) -> dict:
    train_df, valid_df = time_split(df)

    # fit_transform gives training rows past-only entity features; transform then scores the
    # validation period against the history accumulated in training (no look-ahead).
    fb = FeatureBuilder()
    X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]

    model, best_iter, n_trees = fit_model(X_all, y_all, config.RANDOM_STATE)

    scores = model.predict_proba(X_va)[:, 1]
    metrics = evaluate(y_va, scores)
    segments = segment_metrics(fb, valid_df, y_va, scores)

    # Business layer: outcomes by review budget and the cost-optimal operating point.
    amounts = valid_df.get("TransactionAmt", pd.Series(0.0, index=valid_df.index))
    thresholds = threshold_table(y_va, scores, amounts)
    operating = cost_optimal_point(thresholds, config.REVIEW_COST_USD, config.FALSE_POSITIVE_COST_USD)

    # How well does "unusual" alone find fraud? Reported overall and for seen vs. new customers.
    anomaly_only = None
    if "anomaly_score" in X_va:
        a = X_va["anomaly_score"].to_numpy()
        anomaly_only = {
            "overall": {k: evaluate(y_va, a)[k] for k in ("pr_auc", "roc_auc")},
            **segment_metrics(fb, valid_df, y_va, a),
        }

    # Global SHAP importance on a sample of validation transactions.
    sample = X_va.sample(min(len(X_va), 5000), random_state=config.RANDOM_STATE)
    shap_importance = global_importance(contributions(model, sample))

    metadata = {
        "model_type": "LightGBM",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_source": data_source,
        "n_train": len(train_df),
        "n_features": len(fb.feature_names),
        "best_iteration": best_iter,
        "n_trees": n_trees,
        "feature_set": FEATURE_SET,
        "feature_groups": sorted(config.FEATURE_GROUPS),
        "seed": config.RANDOM_STATE,
        "label_delay_days": config.LABEL_DELAY_DAYS,
        "metrics": metrics,
        "segments": segments,
        "anomaly_only": anomaly_only,
        "operating_point": operating,
        "threshold": operating["threshold"],
        "threshold_table": thresholds,
        "shap_importance": shap_importance,
        "feature_descriptions": {f: describe(f) for f in shap_importance},
        "pr_curve": pr_curve_points(y_va, scores),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": fb}, config.MODEL_PATH)
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    pick_examples(valid_df, scores, operating["threshold"]).to_parquet(config.EXAMPLES_PATH)
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
    keys = ("feature_groups", "seed", "metrics", "segments", "anomaly_only", "operating_point")
    print(json.dumps({k: meta[k] for k in keys}, indent=2))
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
