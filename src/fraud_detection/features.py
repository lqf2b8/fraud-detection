"""Feature engineering.

`FeatureBuilder` is fit on training data and saved with the model, so the API applies
exactly the same transformations at inference time. Start simple; extend `_engineer`
with entity/UID aggregates, velocity features, graph features, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection import config

DROP_COLS = {config.TARGET, config.ID_COL}


class FeatureBuilder:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.categorical: dict[str, list[str]] = {}

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "TransactionAmt" in out:
            out["log_amt"] = np.log1p(out["TransactionAmt"])
            out["amt_cents"] = (out["TransactionAmt"] % 1).round(3)
        if config.TIME_COL in out:
            out["hour"] = (out[config.TIME_COL] // 3600) % 24
            out["dow"] = (out[config.TIME_COL] // 86_400) % 7
        return out

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        X = self._engineer(df)
        cols = [c for c in X.columns if c not in DROP_COLS]
        self.categorical = {
            c: sorted(X[c].dropna().astype(str).unique().tolist())
            for c in cols
            if X[c].dtype == object or str(X[c].dtype) in {"string", "category", "str"}
        }
        self.feature_names = cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._engineer(df)
        # Add any columns missing at inference time as NaN (LightGBM handles missing natively).
        for c in self.feature_names:
            if c not in X:
                X[c] = np.nan
        X = X[self.feature_names]
        X = X.astype({c: "float64" for c in self.feature_names if c not in self.categorical})
        for c, cats in self.categorical.items():
            X[c] = pd.Categorical(X[c].astype("object").where(X[c].notna(), None), categories=cats)
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
