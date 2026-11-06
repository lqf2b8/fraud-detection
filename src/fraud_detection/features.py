"""Feature engineering.

`FeatureBuilder` is fit on training data and saved with the model, so the API applies
exactly the same transformations at inference time.

v1 adds **entity features**. A pseudo-customer ID (`uid`) is built from card + billing address +
email domain + account start day (day − D1). For each entity (uid, and card1 on its own),
features describe the entity's *past* behavior only: number of prior transactions, typical amount,
how unusual this amount is for them, and time since their last transaction.

Past-only is the important part:
- Training rows (`fit_transform`) only see earlier training rows.
- Inference rows (`transform`) see the history saved from training, plus earlier rows in the same
  batch. Validation therefore matches what a deployed model would see.

v2 adds **known-fraud history with a reporting delay**. For each entity (customer, card,
email domain + billing address, device), it counts how many of the entity's earlier transactions
were *already confirmed* fraud: labeled, and at least `LABEL_DELAY_DAYS` old. This mimics
chargeback lag. A transaction never sees its own label, or any label too recent to have been
reported yet. Entities like device and email + address can carry risk even for a brand-new
customer.

v3 adds **shared-attribute network features** (see network.py) and an unsupervised
**Isolation Forest anomaly score** (see anomaly.py). Both are aimed at new customers.

The saved history is frozen at training time. In production it would live in a feature store
updated as transactions and chargebacks arrive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection import config
from fraud_detection.anomaly import AnomalyScorer
from fraud_detection.counting import known_label_counts
from fraud_detection.network import build_events, network_features
from fraud_detection.velocity import VelocityTracker

# The raw timestamp is excluded: with a time-based split every validation timestamp lies beyond the
# training range, so the model can't learn anything from it that carries over. Hour and day of
# week are kept.
DROP_COLS = {config.TARGET, config.ID_COL, config.TIME_COL}
AMT = "TransactionAmt"
DAY = 86_400


def make_uid(df: pd.DataFrame) -> pd.Series:
    """card1 + addr1 + P_emaildomain + account start day. Missing when card1, D1 or time is missing."""
    need = ["card1", "D1", config.TIME_COL]
    if not all(c in df for c in need):
        return pd.Series(pd.NA, index=df.index, dtype="string")
    start_day = (df[config.TIME_COL] // DAY - df["D1"]).round()
    key = df["card1"].astype("Int64").astype("string")
    for part in [df.get("addr1"), df.get("P_emaildomain"), start_day]:
        part = pd.Series(pd.NA, index=df.index) if part is None else part
        key = key + "_" + part.astype("string").fillna("NA")
    return key.where(df["card1"].notna() & df["D1"].notna() & df[config.TIME_COL].notna())


def _entity_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    keys = {"uid": make_uid(df)}
    if "card1" in df:
        keys["card1"] = df["card1"].astype("Int64").astype("string")
    return keys


def _label_entity_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Entities whose known-fraud history is tracked. Shared attributes (email + address, device)
    let risk carry over to customers the model has never seen."""
    keys = _entity_keys(df)
    if "P_emaildomain" in df and "addr1" in df:
        keys["email_addr"] = (df["P_emaildomain"].astype("string") + "_"
                              + df["addr1"].astype("string")).where(
            df["P_emaildomain"].notna() & df["addr1"].notna())
    if "DeviceInfo" in df:
        keys["device"] = df["DeviceInfo"].astype("string")
    return keys


class FeatureBuilder:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.categorical: dict[str, list[str]] = {}
        # Per-entity history: index = entity key, columns = n, amt_sum, last_dt
        self.history: dict[str, pd.DataFrame] = {}
        # Labeled training transactions per entity (key, time, label) for known-fraud features
        self.label_history: dict[str, pd.DataFrame] = {}
        self.label_delay_days = config.LABEL_DELAY_DAYS
        self.network_events: dict = {}
        self.anomaly = None
        self.velocity = None
        self._fitting = False

    # ------------------------------------------------------------------ basic features
    @staticmethod
    def _basic(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if AMT in out:
            out["log_amt"] = np.log1p(out[AMT])
            out["amt_cents"] = (out[AMT] % 1).round(3)
        if config.TIME_COL in out:
            out["hour"] = (out[config.TIME_COL] // 3600) % 24
            out["dow"] = (out[config.TIME_COL] // DAY) % 7
        return out

    # ------------------------------------------------------------------ entity features
    def _entity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if config.TIME_COL not in df:
            return pd.DataFrame(index=df.index)
        dt = df[config.TIME_COL].astype("float64")
        amt = df[AMT].astype("float64") if AMT in df else pd.Series(np.nan, index=df.index)
        order = dt.sort_values(kind="stable").index  # causal order within the batch

        feats = {}
        for name, key in _entity_keys(df).items():
            k, d, a = key.loc[order], dt.loc[order], amt.loc[order].fillna(0.0)
            valid = k.notna()
            g_keys = k.fillna("__missing__")
            g = pd.DataFrame({"k": g_keys, "d": d, "a": a}).groupby("k", sort=False)

            hist = self.history.get(name)
            if hist is None or hist.empty:
                n0 = pd.Series(0.0, index=order)
                s0 = pd.Series(0.0, index=order)
                last0 = pd.Series(np.nan, index=order)
            else:
                n0 = g_keys.map(hist["n"]).fillna(0.0)
                s0 = g_keys.map(hist["amt_sum"]).fillna(0.0)
                last0 = g_keys.map(hist["last_dt"])

            n_prev = n0 + g.cumcount()
            sum_prev = s0 + g["a"].cumsum() - a
            last = g["d"].shift().fillna(last0)
            mean_prev = (sum_prev / n_prev).where(n_prev > 0)

            out = pd.DataFrame(
                {
                    f"{name}_n_prev": n_prev,
                    f"{name}_amt_mean_prev": mean_prev,
                    f"{name}_amt_ratio": amt.loc[order] / mean_prev,
                    f"{name}_secs_since_prev": d - last,
                },
                index=order,
            ).where(valid)
            feats[name] = out
        return pd.concat(feats.values(), axis=1).reindex(df.index) if feats else pd.DataFrame(index=df.index)

    def _label_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if config.TIME_COL not in df or not self.label_history:
            return pd.DataFrame(index=df.index)
        delay = self.label_delay_days * DAY
        out = {}
        for name, key in _label_entity_keys(df).items():
            hist = self.label_history.get(name)
            if hist is None:
                continue
            n, f = known_label_counts(hist["k"], hist["t"], hist["y"], key, df[config.TIME_COL], delay)
            out[f"{name}_known_n"] = n
            out[f"{name}_known_fraud"] = f
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{name}_known_fraud_rate"] = np.where(n > 0, f / n, np.nan)
        return pd.DataFrame(out, index=df.index)

    def _set_label_history(self, df: pd.DataFrame) -> None:
        self.label_history = {}
        if config.TARGET not in df or config.TIME_COL not in df:
            return
        for name, key in _label_entity_keys(df).items():
            h = pd.DataFrame({"k": key, "t": df[config.TIME_COL], "y": df[config.TARGET]})
            h = h.dropna(subset=["k"]).reset_index(drop=True)
            self.label_history[name] = h.astype({"k": "category", "y": "int8"})

    def _update_history(self, df: pd.DataFrame) -> None:
        if config.TIME_COL not in df:
            return
        amt = df[AMT].fillna(0.0) if AMT in df else pd.Series(0.0, index=df.index)
        for name, key in _entity_keys(df).items():
            agg = (
                pd.DataFrame({"k": key, "a": amt, "d": df[config.TIME_COL]})
                .dropna(subset=["k"])
                .groupby("k")
                .agg(n=("a", "size"), amt_sum=("a", "sum"), last_dt=("d", "max"))
                .astype("float64")
            )
            old = self.history.get(name)
            if old is not None and not old.empty:
                agg = pd.concat([old, agg]).groupby(level=0).agg(
                    n=("n", "sum"), amt_sum=("amt_sum", "sum"), last_dt=("last_dt", "max")
                )
            self.history[name] = agg

    # ------------------------------------------------------------------ public API
    def _extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if getattr(self, "network_events", None):
            parts.append(network_features(df, make_uid(df), self.network_events))
        if getattr(self, "anomaly", None) is not None:
            parts.append(self.anomaly.score(df).rename("anomaly_score").to_frame())
        if getattr(self, "velocity", None) is not None:
            # Training rows count each other (past-only); inference adds the saved training tail.
            history = None if getattr(self, "_fitting", False) else self.velocity.tail
            parts.append(self.velocity.features(df, history))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [self._basic(df), self._entity_features(df), self._label_features(df),
             self._extra_features(df)],
            axis=1,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data. Returns past-only features, then stores history for inference."""
        self.history = {}
        # Labels from the training data itself; the delay ensures a row only sees labels of
        # transactions at least LABEL_DELAY_DAYS older than itself (never its own).
        self._set_label_history(df)
        self.network_events, self.anomaly, self.velocity = {}, None, None
        if "velocity" in config.FEATURE_GROUPS:
            self.velocity = VelocityTracker().fit(df)
        if "network" in config.FEATURE_GROUPS and config.TARGET in df and config.TIME_COL in df:
            self.network_events = build_events(df, make_uid(df), self.label_delay_days)
        if "anomaly" in config.FEATURE_GROUPS:
            self.anomaly = AnomalyScorer().fit(df.drop(columns=[config.TARGET], errors="ignore"))
        self._fitting = True
        try:
            X = self._engineer(df)
        finally:
            self._fitting = False
        cols = [c for c in X.columns if c not in DROP_COLS]
        self.categorical = {
            c: sorted(X[c].dropna().astype(str).unique().tolist())
            for c in cols
            if X[c].dtype == object or str(X[c].dtype) in {"string", "category", "str"}
        }
        self.feature_names = cols
        self._update_history(df)
        return self._finalize(X)

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        self.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inference: uses the history saved from training (does not update it)."""
        return self._finalize(self._engineer(df))

    def _finalize(self, X: pd.DataFrame) -> pd.DataFrame:
        # Add any columns missing at inference time as NaN (LightGBM handles missing natively).
        X = X.reindex(columns=self.feature_names)
        X = X.astype({c: "float64" for c in self.feature_names if c not in self.categorical})
        for c, cats in self.categorical.items():
            X[c] = pd.Categorical(X[c].astype("object").where(X[c].notna(), None), categories=cats)
        return X
