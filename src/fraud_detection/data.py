"""Data loading.

`load_raw` reads the IEEE-CIS competition files (download with scripts/download_data.sh).
`make_synthetic` generates a small dataset with the same core schema so the whole stack
(training, API, dashboard, CI) runs without the 1.3 GB Kaggle download.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection import config


def load_raw(raw_dir: Path = config.RAW_DATA_DIR, nrows: int | None = None) -> pd.DataFrame:
    """Load and join train_transaction + train_identity on TransactionID."""
    trans = pd.read_csv(raw_dir / "train_transaction.csv", nrows=nrows)
    ident_path = raw_dir / "train_identity.csv"
    if ident_path.exists():
        ident = pd.read_csv(ident_path)
        trans = trans.merge(ident, on=config.ID_COL, how="left")
    return trans


def make_synthetic(n: int = 20_000, fraud_rate: float = 0.035, seed: int = 0) -> pd.DataFrame:
    """Small synthetic stand-in that mirrors a subset of the IEEE-CIS columns."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    amt = np.where(
        is_fraud,
        rng.lognormal(mean=4.6, sigma=1.1, size=n),
        rng.lognormal(mean=4.0, sigma=0.9, size=n),
    ).round(2)

    df = pd.DataFrame(
        {
            config.ID_COL: np.arange(2_987_000, 2_987_000 + n),
            config.TIME_COL: np.sort(rng.integers(86_400, 86_400 * 180, size=n)),
            "TransactionAmt": amt,
            "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n, p=[0.74, 0.12, 0.06, 0.06, 0.02]),
            "card1": rng.integers(1000, 18_000, size=n),
            "card4": rng.choice(["visa", "mastercard", "american express", "discover"], size=n),
            "card6": rng.choice(["debit", "credit"], size=n, p=[0.75, 0.25]),
            "addr1": rng.integers(100, 540, size=n).astype(float),
            "P_emaildomain": rng.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], size=n
            ),
            "DeviceType": rng.choice(["desktop", "mobile", None], size=n, p=[0.15, 0.1, 0.75]),
            "C1": rng.poisson(np.where(is_fraud, 6, 2)).astype(float),
            "D1": np.where(is_fraud, rng.integers(0, 30, n), rng.integers(0, 600, n)).astype(float),
            config.TARGET: is_fraud.astype(int),
        }
    )
    # Inject realistic missingness.
    df.loc[rng.random(n) < 0.1, "addr1"] = np.nan
    df.loc[rng.random(n) < 0.2, "D1"] = np.nan
    return df
