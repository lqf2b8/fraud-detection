"""Train a LightGBM fraud model with a time-based validation split.

Usage:
    python -m fraud_detection.train              # real IEEE-CIS data in data/raw
    python -m fraud_detection.train --synthetic  # synthetic data, no download needed
"""

import argparse
import json
import warnings
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
from fraud_detection.explain import describe, global_importance
from fraud_detection.features import FeatureBuilder
from fraud_detection.preprocess import CatNumEncoder
from fraud_detection.routing import RoutedModel, predict
from fraud_detection.routing import contributions as route_contributions


def time_split(df: pd.DataFrame, valid_frac: float = config.VALID_FRACTION):
    """Train on the past, validate on the future, the way the model will be used."""
    df = df.sort_values(config.TIME_COL)
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut], df.iloc[cut:]


FEATURE_SET = "v5-routed"


def segment_masks(fb: FeatureBuilder, valid_df: pd.DataFrame, X_va: pd.DataFrame | None = None):
    """Two ways to slice validation:
    - seen vs. new: is the customer in the training history?
    - by what's knowable when the transaction arrives (needs the features):
        prior_known_fraud  customer has a fraud confirmed at least LABEL_DELAY_DAYS ago
        prior_known_legit  customer has confirmed history, all legit
        no_prior_info      no confirmed labels yet (new customers, or history too recent)
      plus excluding_known_fraud: everything but the first group. Vesta labels every later
      transaction of a fraud account as fraud, so prior_known_fraud is near-trivial; this is
      the honest measure of detection skill.
    """
    seen = fb.seen_mask(valid_df)
    segs = {"seen_customer": seen, "new_customer": ~seen}
    if X_va is not None and {"uid_known_fraud", "uid_known_n"} <= set(X_va.columns):
        kf = X_va["uid_known_fraud"].fillna(0).to_numpy() > 0
        kn = X_va["uid_known_n"].fillna(0).to_numpy() > 0
        segs.update({
            "prior_known_fraud": kf,
            "prior_known_legit": kn & ~kf,
            "no_prior_info": ~kn,
            "excluding_known_fraud": ~kf,
        })
    return segs


def segment_metrics(fb: FeatureBuilder, valid_df: pd.DataFrame, y_va, scores,
                    X_va: pd.DataFrame | None = None) -> dict:
    out = {}
    for name, mask in segment_masks(fb, valid_df, X_va).items():
        y = np.asarray(y_va)[mask]
        if mask.sum() and 0 < y.sum() < len(y):
            m = evaluate(y, np.asarray(scores)[mask])
            out[name] = {"share": round(float(mask.mean()), 4),
                         **{k: m[k] for k in ("pr_auc", "roc_auc", "base_rate")}}
        elif mask.sum():
            out[name] = {"share": round(float(mask.mean()), 4),
                         "base_rate": round(float(y.mean()), 4)}
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


def fit_model(X_all: pd.DataFrame, y_all: pd.Series, seed: int, sample_weight=None):
    """Choose the tree count by early stopping on the most recent slice of the training period,
    then refit on all training data. Validation is never used to make any modeling choice."""
    es_cut = int(len(X_all) * (1 - config.EARLY_STOP_FRACTION))
    X_tr, y_tr = X_all.iloc[:es_cut], y_all.iloc[:es_cut]
    X_es, y_es = X_all.iloc[es_cut:], y_all.iloc[es_cut:]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
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
    with warnings.catch_warnings():  # newer LightGBM deprecates eval_set; it still works
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        selector.fit(
            X_tr,
            y_tr,
            sample_weight=None if w is None else w[:es_cut],
            eval_set=[(X_es, y_es)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    best_iter = int(selector.best_iteration_ or selector.n_estimators)
    n_trees = max(1, round(best_iter / (1 - config.EARLY_STOP_FRACTION)))
    model = lgb.LGBMClassifier(n_estimators=n_trees, **params)
    model.fit(X_all, y_all, sample_weight=w)
    return model, best_iter, n_trees


def train(df: pd.DataFrame, data_source: str) -> dict:
    train_df, valid_df = time_split(df)

    # fit_transform gives training rows past-only entity features; transform then scores the
    # validation period against the history accumulated in training (no look-ahead).
    fb = FeatureBuilder()
    X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]

    model, best_iter, n_trees = fit_model(X_all, y_all, config.RANDOM_STATE)
    routing = None
    if config.ROUTE_NEW and config.ROUTE_NEW != "none":
        if config.ROUTE_NEW != "catnum":
            raise SystemExit(f"Unsupported ROUTE_NEW={config.ROUTE_NEW!r}; use 'catnum' or 'none'")
        encoder = CatNumEncoder().fit(X_all, list(fb.categorical))
        new_model, new_best, new_trees = fit_model(encoder.transform(X_all), y_all,
                                                   config.RANDOM_STATE)
        model = RoutedModel(model, new_model, encoder)
        routing = {"new_customers": "catnum", "base_trees": n_trees, "new_trees": new_trees,
                   "new_best_iteration": new_best}

    scores = predict(model, fb, valid_df, X_va)
    metrics = evaluate(y_va, scores)
    segments = segment_metrics(fb, valid_df, y_va, scores, X_va)

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
            **segment_metrics(fb, valid_df, y_va, a, X_va),
        }

    # Global SHAP importance on a sample of validation transactions.
    sample_idx = X_va.sample(min(len(X_va), 5000), random_state=config.RANDOM_STATE).index
    shap_importance = global_importance(
        route_contributions(model, fb, valid_df.loc[sample_idx], X_va.loc[sample_idx]))

    metadata = {
        "model_type": "LightGBM (routed: new customers -> catnum)" if routing else "LightGBM",
        "routing": routing,
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

    # Review-queue examples. Their feature rows are saved as scored here, in context: a customer's
    # earlier transactions in the period shape features like prior count and time since last
    # purchase, which re-scoring one transaction alone in the API can't reproduce.
    examples = pick_examples(valid_df, scores, operating["threshold"])
    by_id = pd.Index(valid_df[config.ID_COL])
    rows = by_id.get_indexer(examples[config.ID_COL])
    example_features = X_va.iloc[rows].set_axis(examples[config.ID_COL].to_numpy())
    example_seen = pd.Series(fb.seen_mask(valid_df)[rows], index=example_features.index)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": fb, "example_features": example_features,
                 "example_seen": example_seen}, config.MODEL_PATH)
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    examples.to_parquet(config.EXAMPLES_PATH)
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
    keys = ("feature_set", "routing", "seed", "metrics", "segments", "operating_point")
    print(json.dumps({k: meta[k] for k in keys}, indent=2))
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
