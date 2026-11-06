"""Compare feature sets fairly: same features, several seeds, mean ± standard deviation.

A single run can mislead. Seen-customer fraud is concentrated in a few customers (Vesta labels an
account's later transactions as fraud), so one run's PR AUC can swing by several points with the
random column and row sampling alone. This builds the features once, fits one model per seed, and
reports each seed plus the mean and spread. It never writes to models/, so the served model is safe.

    make experiment                               # baseline feature set, seeds 1-5
    make experiment FEATURE_GROUPS=velocity       # a candidate
    python -m fraud_detection.experiment --seeds 1 2 3 --nrows 200000
"""

import argparse
import json
import time

import numpy as np

from fraud_detection import config
from fraud_detection.data import load_raw, make_synthetic
from fraud_detection.evaluate import evaluate
from fraud_detection.features import FeatureBuilder
from fraud_detection.train import fit_model, segment_metrics, time_split

COLUMNS = [("overall", "pr_auc"), ("seen_customer", "pr_auc"), ("new_customer", "pr_auc"),
           ("overall", "precision_at_5pct")]


def _get(row: dict, group: str, metric: str) -> float:
    """A segment can be absent (e.g. no returning customers in synthetic data)."""
    return float(row.get(group, {}).get(metric, float("nan")))


def run(df, seeds: list[int]) -> dict:
    train_df, valid_df = time_split(df)
    t0 = time.time()
    fb = FeatureBuilder()
    X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]
    print(f"features: {X_all.shape[1]} columns, built in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for seed in seeds:
        t0 = time.time()
        model, _best_iter, n_trees = fit_model(X_all, y_all, seed)
        scores = model.predict_proba(X_va)[:, 1]
        overall = evaluate(y_va, scores)
        seg = segment_metrics(fb, valid_df, y_va, scores)
        row = {
            "seed": seed, "n_trees": n_trees,
            "overall": {k: overall[k] for k in ("pr_auc", "roc_auc", "precision_at_5pct")},
            **{k: {"pr_auc": v["pr_auc"]} for k, v in seg.items()},
        }
        rows.append(row)
        print(f"seed {seed:>3}  trees {n_trees:>5}  " + "  ".join(
            f"{g}.{m} {_get(row, g, m):.4f}" for g, m in COLUMNS) + f"  ({time.time() - t0:.0f}s)",
            flush=True)

    summary = {}
    for g, m in COLUMNS:
        vals = np.array([_get(r, g, m) for r in rows])
        summary[f"{g}.{m}"] = {"mean": round(float(vals.mean()), 4),
                               "sd": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}
    return {"feature_groups": sorted(config.FEATURE_GROUPS), "seeds": seeds, "summary": summary,
            "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    df = make_synthetic() if args.synthetic else load_raw(nrows=args.nrows)
    result = run(df, args.seeds)
    print("\nSUMMARY  feature_groups =", result["feature_groups"] or ["none"])
    for k, v in result["summary"].items():
        print(f"  {k:<28} {v['mean']:.4f} ± {v['sd']:.4f}")
    print(json.dumps({"feature_groups": result["feature_groups"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
