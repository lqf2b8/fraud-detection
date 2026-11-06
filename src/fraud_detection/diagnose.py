"""Why does a feature group help or hurt? Three checks on the candidate group's columns.

1. Drift (PSI): does each new feature look different in validation than in the last month of
   training, for seen and new customers separately? PSI > 0.25 is a large shift.
2. Adversarial validation: can a small model tell late-training rows from validation rows using
   only the new columns? ROC AUC near 0.5 = indistinguishable; well above = they encode time.
3. Block permutation: shuffle all of the group's columns together within each segment of the
   validation set and re-score. If PR AUC goes UP when they're scrambled, the model is being
   misled by them, not just failing to use them.

    make diagnose FEATURE_GROUPS=velocity
"""

import argparse
import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_detection import config
from fraud_detection.data import load_raw, make_synthetic
from fraud_detection.evaluate import evaluate
from fraud_detection.features import FeatureBuilder, make_uid
from fraud_detection.train import fit_model, time_split

GROUP_PATTERNS = {
    "velocity": r"^vel_",
    "network": r"^net_",
    "anomaly": r"^anomaly_score$",
}
DAY = 86_400


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e, a = expected.dropna().to_numpy(float), actual.dropna().to_numpy(float)
    if len(e) < 50 or len(a) < 50:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [edges[0] - 1, edges[-1] + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    groups = sorted(config.FEATURE_GROUPS)
    if not groups:
        raise SystemExit("Pick a group to diagnose, e.g. FEATURE_GROUPS=velocity")
    pattern = "|".join(GROUP_PATTERNS[g] for g in groups)

    df = make_synthetic() if args.synthetic else load_raw(nrows=args.nrows)
    train_df, valid_df = time_split(df)
    fb = FeatureBuilder()
    X_all, y_all = fb.fit_transform(train_df), train_df[config.TARGET]
    X_va, y_va = fb.transform(valid_df), valid_df[config.TARGET]
    cols = [c for c in X_all.columns if re.search(pattern, c)]
    print(f"diagnosing {groups}: {len(cols)} columns", flush=True)

    seen_idx = fb.history.get("uid", pd.DataFrame()).index
    seen_va = make_uid(valid_df).isin(seen_idx).fillna(False).to_numpy(bool)
    # Late training: the last 30 days, and whether that customer had appeared before then.
    t_tr = train_df[config.TIME_COL].to_numpy()
    late = t_tr >= t_tr.max() - 30 * DAY
    uid_tr = make_uid(train_df)
    first_seen = uid_tr.groupby(uid_tr).transform(lambda s: s.index.min())
    seen_tr = (first_seen.to_numpy() != np.arange(len(uid_tr))) & uid_tr.notna().to_numpy()
    seen_tr = pd.Series(seen_tr, index=train_df.index).loc[X_all.index].to_numpy()

    # 1. Drift
    drift = []
    for c in cols:
        drift.append({
            "feature": c,
            "psi_seen": round(psi(X_all.loc[late & seen_tr, c], X_va.loc[seen_va, c]), 3),
            "psi_new": round(psi(X_all.loc[late & ~seen_tr, c], X_va.loc[~seen_va, c]), 3),
            "mean_train_seen": round(float(X_all.loc[late & seen_tr, c].mean()), 2),
            "mean_valid_seen": round(float(X_va.loc[seen_va, c].mean()), 2),
        })
    drift = sorted(drift, key=lambda d: -np.nan_to_num(d["psi_seen"]))
    print("\n1) DRIFT  late training -> validation (PSI > 0.25 = large shift)")
    print(pd.DataFrame(drift).to_string(index=False))

    # 2. Adversarial validation on the new columns only
    A = pd.concat([X_all.loc[late, cols], X_va[cols]], ignore_index=True)
    lab = np.r_[np.zeros(late.sum()), np.ones(len(X_va))]
    rng = np.random.default_rng(args.seed)
    mask = rng.random(len(A)) < 0.7
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, random_state=args.seed)
    clf.fit(A[mask], lab[mask])
    adv_auc = roc_auc_score(lab[~mask], clf.predict_proba(A[~mask])[:, 1])
    print(f"\n2) ADVERSARIAL VALIDATION  ROC AUC {adv_auc:.3f}  (0.5 = no time signal in these columns)")

    # 3. Block permutation within segments
    model, _, _ = fit_model(X_all, y_all, args.seed)
    base = model.predict_proba(X_va)[:, 1]
    X_perm = X_va.copy()
    for seg in (seen_va, ~seen_va):
        idx = np.flatnonzero(seg)
        X_perm.iloc[idx, [X_va.columns.get_loc(c) for c in cols]] = (
            X_va.iloc[rng.permutation(idx), [X_va.columns.get_loc(c) for c in cols]].to_numpy())
    perm = model.predict_proba(X_perm)[:, 1]
    print("\n3) BLOCK PERMUTATION  PR AUC with the group's columns intact vs. scrambled")
    out = {}
    for name, seg in (("overall", np.ones(len(y_va), bool)), ("seen_customer", seen_va),
                      ("new_customer", ~seen_va)):
        y = y_va.to_numpy()[seg]
        if 0 < y.sum() < len(y):
            a, b = evaluate(y, base[seg])["pr_auc"], evaluate(y, perm[seg])["pr_auc"]
            out[name] = {"intact": round(a, 4), "scrambled": round(b, 4), "delta": round(b - a, 4)}
            verdict = "MISLEADING (better scrambled)" if b > a + 0.002 else (
                "useful" if a > b + 0.002 else "unused")
            print(f"  {name:<14} intact {a:.4f}  scrambled {b:.4f}  -> {verdict}")

    print("\n" + json.dumps({"groups": groups, "adversarial_auc": round(adv_auc, 3),
                             "permutation": out, "top_drift": drift[:5]}))


if __name__ == "__main__":
    main()
