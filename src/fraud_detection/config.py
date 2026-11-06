"""Central configuration. Every path can be overridden with an environment variable."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Fraction of the most recent transactions held out for validation (time-based split).
VALID_FRACTION = float(os.getenv("VALID_FRACTION", "0.2"))
RANDOM_STATE = 42

# Fraud labels (chargebacks) arrive late. Label-based features may only use labels at least this
# many days old, so the model never relies on information a live system wouldn't have yet.
LABEL_DELAY_DAYS = float(os.getenv("LABEL_DELAY_DAYS", "30"))
