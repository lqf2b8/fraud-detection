"""v5: route each transaction to the model whose encoding suits its customer.

    seen customers (in the training history) -> base model, categoricals as LightGBM categories.
        For a returning customer, the exact device or email domain is part of their identity.
    new customers                             -> catnum model: categoricals as numbers with rare
        values grouped. For a stranger, rare exact values are noise to memorize; grouping them
        generalizes better.

Both models train on all of the training data and differ only in categorical encoding.
5-seed experiment: overall PR AUC 0.658 -> 0.663, new customers 0.559 -> 0.566, seen unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.explain import contributions as _contributions
from fraud_detection.preprocess import CatNumEncoder


class RoutedModel:
    routed = True

    def __init__(self, base, new, encoder: CatNumEncoder) -> None:
        self.base, self.new, self.encoder = base, new, encoder

    def predict_proba(self, X: pd.DataFrame, seen) -> np.ndarray:
        seen = np.asarray(seen, dtype=bool)
        p = np.empty(len(X))
        if seen.any():
            p[seen] = self.base.predict_proba(X[seen])[:, 1]
        if (~seen).any():
            p[~seen] = self.new.predict_proba(self.encoder.transform(X[~seen]))[:, 1]
        return np.column_stack([1 - p, p])

    def contributions(self, X: pd.DataFrame, seen) -> pd.DataFrame:
        """SHAP contributions from whichever model scored each row (same columns either way)."""
        seen = np.asarray(seen, dtype=bool)
        parts = []
        if seen.any():
            parts.append(_contributions(self.base, X[seen]))
        if (~seen).any():
            enc = self.encoder.transform(X[~seen])
            parts.append(_contributions(self.new, enc).set_axis(enc.index))
        return pd.concat(parts).reindex(X.index)


def predict_with_seen(model, X: pd.DataFrame, seen) -> np.ndarray:
    if getattr(model, "routed", False):
        return model.predict_proba(X, seen)[:, 1]
    return model.predict_proba(X)[:, 1]


def contributions_with_seen(model, X: pd.DataFrame, seen) -> pd.DataFrame:
    if getattr(model, "routed", False):
        return model.contributions(X, seen)
    return _contributions(model, X)


def predict(model, fb, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability for a plain or routed model."""
    return predict_with_seen(model, X, fb.seen_mask(df))


def contributions(model, fb, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    return contributions_with_seen(model, X, fb.seen_mask(df))
