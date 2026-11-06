"""Unsupervised anomaly score (Isolation Forest).

Fit on *all* training transactions without labels. Fitting only on legit transactions and then
scoring the training set would leak labels: legit rows would be in-sample and fraud rows
out-of-sample, so fraud would look more anomalous for the wrong reason.

The score is used two ways: as a feature for the supervised model, and reported on its own
(how well does "unusual" alone find fraud, especially for new customers with no history?).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraud_detection import config

MAX_FIT_ROWS = 200_000


def anomaly_columns(df: pd.DataFrame) -> list[str]:
    """Behavioral numeric columns: amount, counts (C), time deltas (D), distances, card/address codes."""
    base = ["TransactionAmt", "dist1", "dist2", "addr1", "card1", "card2", "card3", "card5"]
    pattern = re.compile(r"^(C|D)\d+$")
    return [c for c in base if c in df] + [c for c in df.columns if pattern.match(c)]


class AnomalyScorer:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.model: IsolationForest | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.columns).astype("float64")
        if "TransactionAmt" in X:
            X["TransactionAmt"] = np.log1p(X["TransactionAmt"])
        # Missing is informative in this data; a sentinel lets the trees isolate it.
        return X.fillna(-999.0).to_numpy()

    def fit(self, df: pd.DataFrame) -> AnomalyScorer:
        self.columns = anomaly_columns(df)
        if not self.columns:
            return self
        sample = df.sample(min(len(df), MAX_FIT_ROWS), random_state=config.RANDOM_STATE)
        self.model = IsolationForest(
            n_estimators=200, max_samples=min(256, len(sample)), random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ).fit(self._matrix(sample))
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Higher = more anomalous."""
        if self.model is None:
            return pd.Series(np.nan, index=df.index)
        return pd.Series(-self.model.score_samples(self._matrix(df)), index=df.index)
