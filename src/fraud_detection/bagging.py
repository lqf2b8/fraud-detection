"""Seed bagging: the same LightGBM trained with several random seeds, averaged.

Members differ only in their random row and column sampling, so averaging cancels part of each
model's noise. Scores are averaged in log-odds space, not probability space, so SHAP values stay
exact: each member's SHAP values sum to its log-odds, so their average sums to the bag's log-odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def average_logits(prob_columns: list[np.ndarray]) -> np.ndarray:
    """Average several models' probabilities in log-odds space."""
    p = np.clip(np.column_stack(prob_columns), 1e-12, 1 - 1e-12)
    return _sigmoid(np.log(p / (1 - p)).mean(axis=1))


class BaggedModel:
    def __init__(self, members: list) -> None:
        self.members = members

    @property
    def n_trees(self) -> list[int]:
        return [m.n_estimators for m in self.members]

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X, raw_score=True) for m in self.members], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = _sigmoid(self.raw_score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame, pred_contrib: bool = False, **kwargs):
        if pred_contrib:  # same interface explain.contributions uses for a single LightGBM
            return np.mean([np.asarray(m.predict(X, pred_contrib=True)) for m in self.members],
                           axis=0)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
