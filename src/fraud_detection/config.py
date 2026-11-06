"""Central configuration. Every path can be overridden with an environment variable."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
EXAMPLES_PATH = MODEL_DIR / "examples.parquet"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Fraction of the most recent transactions held out for validation (time-based split).
VALID_FRACTION = float(os.getenv("VALID_FRACTION", "0.2"))
RANDOM_STATE = int(os.getenv("SEED", "42"))

# Optional feature groups, off by default: the v3 ablation found that neither improves the model
# (see README). Enable for experiments, e.g. FEATURE_GROUPS=velocity  or  FEATURE_GROUPS=network,anomaly
FEATURE_GROUPS = {
    g.strip() for g in os.getenv("FEATURE_GROUPS", "none").split(",") if g.strip()
} - {"none"}

# Fraud labels (chargebacks) arrive late. Label-based features may only use labels at least this
# many days old, so the model never relies on information a live system wouldn't have yet.
LABEL_DELAY_DAYS = float(os.getenv("LABEL_DELAY_DAYS", "30"))

# Business assumptions for threshold selection (adjustable in the dashboard).
# Cost at a threshold = REVIEW_COST_USD × flagged + FALSE_POSITIVE_COST_USD × false alarms
#                      + $ of fraud missed.
REVIEW_COST_USD = float(os.getenv("REVIEW_COST_USD", "10"))
# Customer friction per false alarm (delayed or declined good customer). 0 = review cost only.
FALSE_POSITIVE_COST_USD = float(os.getenv("FALSE_POSITIVE_COST_USD", "0"))

# Fraction of the training period held out for early stopping, so validation stays untouched.
EARLY_STOP_FRACTION = float(os.getenv("EARLY_STOP_FRACTION", "0.1"))
