"""Shared-attribute network features.

Customers (pseudo-customer IDs) and attributes (device, email domain + billing address, full card
fingerprint) form a bipartite graph: a customer is linked to every attribute value it has used.
Fraud rings show up as many customers sharing the same device or card, and some of those
customers already being confirmed fraud.

For a transaction by customer u using attribute value a at time t, we compute, using only
the past:
    customers        how many *other* customers used a before t
    fraud_customers  how many of those other customers were confirmed fraudsters before t
                     (a customer counts once its fraud label is LABEL_DELAY_DAYS old)
    fraud_share      fraud_customers / customers

This reaches brand-new customers. A first-time customer has no history of their own, but their
device or card may already be linked to known fraud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection import config
from fraud_detection.counting import known_label_counts

DAY = 86_400
SAME_TIME_EPS = 0.5  # seconds: events at exactly t are not "before" t


def network_attributes(df: pd.DataFrame) -> dict[str, pd.Series]:
    attrs = {}
    if "DeviceInfo" in df:
        attrs["device"] = df["DeviceInfo"].astype("string")
    if "P_emaildomain" in df and "addr1" in df:
        attrs["email_addr"] = (
            df["P_emaildomain"].astype("string") + "_" + df["addr1"].astype("string")
        ).where(df["P_emaildomain"].notna() & df["addr1"].notna())
    card_cols = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in df]
    if "card1" in card_cols and len(card_cols) > 1:
        key = df["card1"].astype("Int64").astype("string")
        for c in card_cols[1:]:
            key = key + "_" + df[c].astype("string").fillna("NA")
        attrs["card_full"] = key.where(df["card1"].notna())
    return attrs


def build_events(df: pd.DataFrame, uid: pd.Series, delay_days: float) -> dict[str, dict]:
    """Event tables from labeled history. For each attribute:
    cust: (attr, time the customer first used it), one row per (attr, customer)
    fraud: (attr, time a fraudster customer's link became *known*): max(first use, first fraud + delay)
    The *_self tables are keyed by attr|customer, so a customer can be excluded from its own counts.
    """
    t = df[config.TIME_COL]
    y = df[config.TARGET] if config.TARGET in df else pd.Series(0, index=df.index)
    fraud_t = t[(y == 1) & uid.notna()].groupby(uid[(y == 1) & uid.notna()]).min()
    events = {}
    for name, attr in network_attributes(df).items():
        ok = attr.notna() & uid.notna()
        pairs = (
            pd.DataFrame({"a": attr[ok], "u": uid[ok], "t": t[ok]})
            .groupby(["a", "u"], sort=False)["t"].min().reset_index()
        )
        known = pairs["u"].map(fraud_t) + delay_days * DAY
        f = pairs[known.notna()].assign(t=np.maximum(pairs["t"][known.notna()], known.dropna()))
        tables = {
            "cust": pd.DataFrame({"k": pairs["a"], "t": pairs["t"]}),
            "cust_self": pd.DataFrame({"k": pairs["a"] + "|" + pairs["u"], "t": pairs["t"]}),
            "fraud": pd.DataFrame({"k": f["a"], "t": f["t"]}),
            "fraud_self": pd.DataFrame({"k": f["a"] + "|" + f["u"], "t": f["t"]}),
        }
        events[name] = {k: _compact(v) for k, v in tables.items()}
    return events


def _compact(table: pd.DataFrame) -> pd.DataFrame:
    """Categorical keys and float32 times keep the saved model small."""
    return pd.DataFrame({"k": table["k"].astype("category"), "t": table["t"].astype("float64")})


def _count_before(table: pd.DataFrame, q_key: pd.Series, q_t: pd.Series) -> np.ndarray:
    n, _ = known_label_counts(table["k"], table["t"], np.zeros(len(table)), q_key, q_t, SAME_TIME_EPS)
    return n


def network_features(df: pd.DataFrame, uid: pd.Series, events: dict[str, dict]) -> pd.DataFrame:
    if config.TIME_COL not in df or not events:
        return pd.DataFrame(index=df.index)
    t = df[config.TIME_COL]
    out = {}
    for name, attr in network_attributes(df).items():
        ev = events.get(name)
        if ev is None:
            continue
        self_key = (attr + "|" + uid).where(attr.notna() & uid.notna())
        cust = _count_before(ev["cust"], attr, t) - np.nan_to_num(_count_before(ev["cust_self"], self_key, t))
        fraud = _count_before(ev["fraud"], attr, t) - np.nan_to_num(
            _count_before(ev["fraud_self"], self_key, t)
        )
        out[f"net_{name}_customers"] = cust
        out[f"net_{name}_fraud_customers"] = fraud
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"net_{name}_fraud_share"] = np.where(cust > 0, fraud / cust, np.nan)
    res = pd.DataFrame(out, index=df.index)
    fraud_cols = [c for c in res if c.endswith("_fraud_customers")]
    if fraud_cols:
        res["net_fraud_neighbors_total"] = res[fraud_cols].sum(axis=1, min_count=1)
        res["net_max_shared_customers"] = res[[c for c in res if c.endswith("_customers")
                                               and "fraud" not in c]].max(axis=1)
    return res
