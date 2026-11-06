"""SHAP explanations and plain-English feature names.

LightGBM computes exact TreeSHAP values natively (`pred_contrib=True`). For every transaction,
contributions are in log-odds and sum to the model's raw score:
    bias + sum(contributions) = logit(fraud probability)
Positive values push toward fraud, negative toward legit.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_EXACT = {
    "TransactionAmt": "Transaction amount ($)",
    "log_amt": "Transaction amount (log)",
    "amt_cents": "Cents part of amount",
    "hour": "Hour of day",
    "dow": "Day of week",
    "ProductCD": "Product code",
    "card4": "Card network",
    "card6": "Card type (debit/credit)",
    "addr1": "Billing region",
    "addr2": "Billing country",
    "dist1": "Distance (billing ↔ shipping)",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "DeviceType": "Device type",
    "DeviceInfo": "Device / OS",
}

_ENTITY = {"uid": "customer", "card1": "card", "email_addr": "email + address", "device": "device"}
_STAT = {
    "n_prev": "Prior transactions on this {e}",
    "amt_mean_prev": "Typical amount for this {e}",
    "amt_ratio": "Amount vs. this {e}'s typical amount",
    "secs_since_prev": "Time since this {e}'s last transaction",
    "known_n": "Prior transactions on this {e} with known outcome",
    "known_fraud": "Confirmed prior frauds on this {e}",
    "known_fraud_rate": "Confirmed fraud rate on this {e}",
}
_PREFIX = {
    "card": "Card attribute", "C": "Vesta count feature", "D": "Vesta time-delta feature",
    "M": "Vesta match flag", "V": "Vesta engineered feature", "id_": "Identity feature",
}


def describe(feature: str) -> str:
    if feature in _EXACT:
        return _EXACT[feature]
    for ent, ent_name in _ENTITY.items():
        if feature.startswith(ent + "_"):
            stat = feature[len(ent) + 1:]
            if stat in _STAT:
                return _STAT[stat].format(e=ent_name)
    m = re.fullmatch(r"(card|C|D|M|V|id_)(\d+)", feature)
    if m:
        return f"{_PREFIX[m.group(1)]} {m.group(1)}{m.group(2)}"
    return feature


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row SHAP contributions (log-odds). Last column 'bias' is the expected value."""
    raw = model.predict(X, pred_contrib=True)
    return pd.DataFrame(np.asarray(raw), columns=[*X.columns, "bias"], index=X.index)


def _fmt(v) -> str | float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        return round(float(v), 4)
    return str(v)


def top_reasons(contrib: pd.DataFrame, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
    """Top-k features by |contribution| for each row, with their values and descriptions."""
    c = contrib.drop(columns="bias")
    out = []
    for i in range(len(c)):
        row = c.iloc[i]
        idx = row.abs().nlargest(k).index
        out.append(
            [
                {
                    "feature": f,
                    "description": describe(f),
                    "value": _fmt(X.iloc[i][f]),
                    "contribution": round(float(row[f]), 4),
                }
                for f in idx
            ]
        )
    return out


def global_importance(contrib: pd.DataFrame, top: int = 25) -> dict[str, float]:
    """Mean |SHAP| per feature: how much each feature moves scores on average."""
    imp = contrib.drop(columns="bias").abs().mean().sort_values(ascending=False).head(top)
    return {k: round(float(v), 4) for k, v in imp.items()}
