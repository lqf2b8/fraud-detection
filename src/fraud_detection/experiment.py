"""Compare feature sets fairly: same features, several seeds, mean ± standard deviation.

A single run can mislead. Seen-customer fraud is concentrated in a few customers (Vesta labels an
account's later transactions as fraud), so one run's PR AUC can swing by several points with the
random column and row sampling alone. This builds the features once, fits one model per seed, and
reports each seed plus the mean and spread. It never writes to models/, so the served model is safe.

    make experiment                               # baseline feature set, seeds 1-5
    make experiment FEATURE_GROUPS=velocity       # a candidate feature group
    make experiment ARGS=--route                  # separate model for new customers
    python -m fraud_detection.experiment --seeds 1 2 3 --nrows 200000
"""

import argparse
import json
import time

import numpy as np

from fraud_detection import config
from fraud_detection.data import load_raw, make_synthetic
from fraud_detection.evaluate import evaluate
from fraud_detection.features import FeatureBuilder, make_uid
from fraud_detection.train import fit_model, segment_metrics, time_split

COLUMNS = [("overall", "pr_auc"), ("seen_customer", "pr_auc"), ("new_customer", "pr_auc"),
           ("overall", "precision_at_5pct")]


def _get(row: dict, group: str, metric: str) -> float:
    """A segment can be absent (e.g. no returning customers in synthetic data)."""
    return float(row.get(group, {}).get(metric, float("nan")))


def run(df, seeds: list[int], route: bool = False) -> dict:
    train_df, valid_df = time_split(df)
    t0 = time.time()
    fb = FeatureBuilder()
    X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]
    print(f"features: {X_all.shape[1]} columns, built in {time.time() - t0:.0f}s", flush=True)
    if route:
        # New-customer model: trained only on each customer's first transaction (no prior history),
        # so label-propagated repeat fraud doesn't dominate its loss. Validation transactions whose
        # customer isn't in the training history are scored by it; the rest by the usual model.
        first_tx = X_all["uid_n_prev"].fillna(0).to_numpy() == 0
        seen_va = make_uid(valid_df).isin(fb.history["uid"].index).fillna(False).to_numpy(bool)
        print(f"routing: new-customer model trains on {first_tx.sum():,} first transactions "
              f"({y_all[first_tx].mean():.2%} fraud); scores {(~seen_va).sum():,} validation rows",
              flush=True)

    rows = []
    for seed in seeds:
        t0 = time.time()
        model, _best_iter, n_trees = fit_model(X_all, y_all, seed)
        scores = model.predict_proba(X_va)[:, 1]
        if route:
            new_model, _, _ = fit_model(X_all[first_tx], y_all[first_tx], seed)
            scores = np.where(seen_va, scores, new_model.predict_proba(X_va)[:, 1])
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
    return {"feature_groups": sorted(config.FEATURE_GROUPS), "route": route, "seeds": seeds,
            "summary": summary,
            "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--route", action="store_true",
                        help="separate model for new customers, trained on first transactions")
    args = parser.parse_args()
    df = make_synthetic() if args.synthetic else load_raw(nrows=args.nrows)
    result = run(df, args.seeds, args.route)
    print("\nSUMMARY  feature_groups =", result["feature_groups"] or ["none"],
          " routing =", "on" if args.route else "off")
    for k, v in result["summary"].items():
        print(f"  {k:<28} {v['mean']:.4f} ± {v['sd']:.4f}")
    print(json.dumps({"feature_groups": result["feature_groups"], "route": args.route,
                      "summary": result["summary"]}))


if __name__ == "__main__":
    main()
