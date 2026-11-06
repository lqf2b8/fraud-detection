"""Compare feature sets fairly: same features, several seeds, mean ± standard deviation.

A single run can mislead. Seen-customer fraud is concentrated in a few customers (Vesta labels an
account's later transactions as fraud), so one run's PR AUC can swing by several points with the
random column and row sampling alone. This builds the features once, fits one model per seed, and
reports each seed plus the mean and spread. It never writes to models/, so the served model is safe.

    make experiment                               # baseline feature set, seeds 1-5
    make experiment FEATURE_GROUPS=velocity       # a candidate feature group
    make experiment ARGS=--route                  # separate model for new customers
    make experiment ARGS="--half-life 0 30 90"    # recency weighting, several half-lives
    make experiment ARGS="--seeds 1 2 3 --variants catnum freq nan999 dnorm"   # preprocessing
    python -m fraud_detection.experiment --seeds 1 2 3 --nrows 200000
"""

import argparse
import json
import time

import numpy as np

from fraud_detection import config, preprocess
from fraud_detection.data import load_raw, make_synthetic
from fraud_detection.evaluate import evaluate
from fraud_detection.features import FeatureBuilder, make_uid
from fraud_detection.train import fit_model, segment_metrics, time_split

COLUMNS = [("overall", "pr_auc"), ("seen_customer", "pr_auc"), ("new_customer", "pr_auc"),
           ("overall", "precision_at_5pct")]


def _get(row: dict, group: str, metric: str) -> float:
    """A segment can be absent (e.g. no returning customers in synthetic data)."""
    return float(row.get(group, {}).get(metric, float("nan")))


def recency_weights(t: np.ndarray, half_life_days: float | None) -> np.ndarray | None:
    """Weight halves every half_life_days going back from the newest training transaction,
    rescaled to mean 1 so the effective learning rate is unchanged."""
    if not half_life_days:
        return None
    age_days = (t.max() - t) / 86_400
    w = 0.5 ** (age_days / half_life_days)
    return w / w.mean()


def run(df, seeds: list[int], route: bool = False, half_life: float | None = None,
        _cache: dict | None = None, variant: str = "baseline") -> dict:
    if _cache:
        train_df, valid_df, fb, X_all, y_all, X_va, y_va = _cache["data"]
    else:
        train_df, valid_df = time_split(df)
        t0 = time.time()
        fb = FeatureBuilder()
        X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
        X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]
        print(f"features: {X_all.shape[1]} columns, built in {time.time() - t0:.0f}s", flush=True)
        if _cache is not None:
            _cache["data"] = (train_df, valid_df, fb, X_all, y_all, X_va, y_va)
    if variant != "baseline":
        X_all, X_va = preprocess.apply(variant, X_all, X_va, list(fb.categorical),
                                       train_df[config.TIME_COL], valid_df[config.TIME_COL])
        print(f"variant {variant}: {X_all.shape[1]} columns", flush=True)
    weights = recency_weights(train_df[config.TIME_COL].to_numpy(float), half_life)
    if weights is not None:
        print(f"recency weighting: half-life {half_life:g} days; oldest weight "
              f"{weights.min():.2f}, newest {weights.max():.2f} (mean 1)", flush=True)
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
        model, _best_iter, n_trees = fit_model(X_all, y_all, seed, weights)
        scores = model.predict_proba(X_va)[:, 1]
        if route:
            new_model, _, _ = fit_model(X_all[first_tx], y_all[first_tx], seed,
                                        None if weights is None else weights[first_tx])
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
    return {"feature_groups": sorted(config.FEATURE_GROUPS), "route": route,
            "half_life_days": half_life, "variant": variant, "seeds": seeds,
            "summary": summary,
            "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--half-life", type=float, nargs="+", default=[0],
                        help="recency weighting half-life(s) in days; 0 = none. Several = one run each")
    parser.add_argument("--variants", nargs="+", default=["baseline"],
                        help="preprocessing variants, e.g. catnum freq catnum+freq+dnorm nan999")
    parser.add_argument("--route", action="store_true",
                        help="separate model for new customers, trained on first transactions")
    args = parser.parse_args()
    df = make_synthetic() if args.synthetic else load_raw(nrows=args.nrows)
    cache: dict = {}
    results = []
    for hl in args.half_life:
        for variant in args.variants:
            preprocess.parse(variant)  # fail fast on typos
            label = (f"half-life {hl:g}d" if hl else "unweighted") + f", variant {variant}"
            print(f"\n=== {label} ===", flush=True)
            r = run(df, args.seeds, args.route, hl or None, _cache=cache, variant=variant)
            results.append((label, r))
    for label, result in results:
        print(f"\nSUMMARY  feature_groups = {result['feature_groups'] or ['none']}  "
              f"routing = {'on' if args.route else 'off'}  {label}")
        for k, v in result["summary"].items():
            print(f"  {k:<28} {v['mean']:.4f} ± {v['sd']:.4f}")
    print(json.dumps([{"label": label, "summary": r["summary"]} for label, r in results]))

if __name__ == "__main__":
    main()
