"""Preprocessing variants from Chris Deotte's IEEE-CIS feature-engineering guide, adapted so the
training/serving contract holds (everything is frozen from training or computed past-only).

    catnum   categoricals label-encoded and treated as numbers instead of LightGBM categories;
             values seen in < 0.1% of training rows grouped into one code (rare-value relaxing)
    freq     frequency encoding: how common each card / region / domain / device value was in
             training (frozen map; unseen values = 0)
    nan999   missing numbers replaced by -999 so trees treat "missing" as a low value instead of
             re-deciding where NaNs go at every split
    dnorm    D columns (days since some event) minus their trailing 7-day mean across all
             transactions, removing the upward drift the raw values have over time

Variants combine with "+", e.g. "catnum+freq+dnorm".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from fraud_detection.counting import window_sums

RARE_SHARE = 0.001
FREQ_NUMERIC = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
WEEK = 7 * 86_400
KNOWN = {"catnum", "freq", "nan999", "dnorm"}


def parse(variant: str) -> set[str]:
    parts = {p.strip() for p in variant.split("+") if p.strip()} - {"baseline"}
    unknown = parts - KNOWN
    if unknown:
        raise SystemExit(f"Unknown variant part(s) {sorted(unknown)}; choose from {sorted(KNOWN)}")
    return parts


def _catnum(X_tr: pd.DataFrame, X_va: pd.DataFrame, cat_cols: list[str]):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        tr_codes = X_tr[c].cat.codes.to_numpy()          # -1 = missing
        va_codes = X_va[c].cat.codes.to_numpy()
        counts = np.bincount(tr_codes[tr_codes >= 0], minlength=len(X_tr[c].cat.categories))
        rare = counts < RARE_SHARE * len(X_tr)
        def relax(codes, rare=rare):
            out = codes.astype("float64")
            out[(codes >= 0) & rare[np.clip(codes, 0, None)]] = -2  # rare values share one code
            return out
        X_tr[c], X_va[c] = relax(tr_codes), relax(va_codes)
    return X_tr, X_va


def _freq(X_tr: pd.DataFrame, X_va: pd.DataFrame, cat_cols: list[str]):
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in [*cat_cols, *[c for c in FREQ_NUMERIC if c in X_tr]]:
        share = X_tr[c].astype("object").value_counts(normalize=True, dropna=True)
        X_tr[f"{c}_freq"] = X_tr[c].astype("object").map(share).astype("float64").fillna(0.0)
        X_va[f"{c}_freq"] = X_va[c].astype("object").map(share).astype("float64").fillna(0.0)
    return X_tr, X_va


def _nan999(X_tr: pd.DataFrame, X_va: pd.DataFrame):
    num = [c for c in X_tr.columns if str(X_tr[c].dtype) != "category"]
    X_tr, X_va = X_tr.copy(), X_va.copy()
    X_tr[num], X_va[num] = X_tr[num].fillna(-999.0), X_va[num].fillna(-999.0)
    return X_tr, X_va


def _dnorm(X_tr, X_va, t_tr: np.ndarray, t_va: np.ndarray):
    """Replace each D column with D minus the mean of that column over all transactions in the
    previous 7 days (strictly before t). Past-only; validation sees late training plus earlier
    validation, exactly as a live system would."""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    t_all = np.r_[t_tr, t_va]
    key = np.zeros(len(t_all), dtype="int64").astype(str)
    for c in [c for c in X_tr.columns if re.fullmatch(r"D\d+", c)]:
        d = np.r_[X_tr[c].to_numpy(float), X_va[c].to_numpy(float)]
        present = ~np.isnan(d)
        _, s = window_sums(key, t_all, np.nan_to_num(d), key, t_all, WEEK)
        _, n = window_sums(key, t_all, present.astype(float), key, t_all, WEEK)
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        detrended = d - mean
        X_tr[c], X_va[c] = detrended[: len(t_tr)], detrended[len(t_tr):]
    return X_tr, X_va


def apply(variant: str, X_tr, X_va, cat_cols: list[str], t_tr, t_va):
    parts = parse(variant)
    if "dnorm" in parts:
        X_tr, X_va = _dnorm(X_tr, X_va, np.asarray(t_tr, float), np.asarray(t_va, float))
    if "freq" in parts:
        X_tr, X_va = _freq(X_tr, X_va, cat_cols)
    if "catnum" in parts:
        X_tr, X_va = _catnum(X_tr, X_va, cat_cols)
    if "nan999" in parts:
        X_tr, X_va = _nan999(X_tr, X_va)
    return X_tr, X_va
