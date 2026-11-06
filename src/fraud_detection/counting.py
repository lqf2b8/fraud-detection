"""Vectorized as-of counting shared by label-history and network features."""

import numpy as np
import pandas as pd


def known_label_counts(hist_key, hist_t, hist_y, q_key, q_t, delay_secs: float):
    """For each query (key, time), count history rows with the same key whose time is at most
    time - delay, and how many of those were fraud. Vectorized with one sort + searchsorted."""
    hist_key = pd.Series(hist_key).astype("string").reset_index(drop=True)
    q_key = pd.Series(q_key).astype("string").reset_index(drop=True)
    codes, _ = pd.factorize(pd.concat([hist_key, q_key], ignore_index=True), use_na_sentinel=True)
    hk, qk = codes[: len(hist_key)], codes[len(hist_key):]
    h_ok = hk >= 0
    hk, ht, hy = hk[h_ok], np.asarray(hist_t, float)[h_ok], np.asarray(hist_y, float)[h_ok]

    span = max(ht.max(initial=0), np.nanmax(np.asarray(q_t, float), initial=0)) + delay_secs + 1
    order = np.lexsort((ht, hk))
    v = hk[order] * span + ht[order]
    cum_fraud = np.concatenate([[0.0], np.cumsum(hy[order])])

    q_ok = (qk >= 0) & ~np.isnan(np.asarray(q_t, float))
    qkk, qt = qk[q_ok].astype(float), np.asarray(q_t, float)[q_ok]
    start = np.searchsorted(v, qkk * span, side="left")
    end = np.searchsorted(v, qkk * span + (qt - delay_secs), side="right")
    end = np.maximum(end, start)  # (qt - delay) can fall before the group's first row

    n = np.full(len(q_key), np.nan)
    f = np.full(len(q_key), np.nan)
    n[q_ok] = end - start
    f[q_ok] = cum_fraud[end] - cum_fraud[start]
    return n, f
