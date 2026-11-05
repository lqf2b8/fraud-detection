#!/usr/bin/env bash
# Downloads the IEEE-CIS Fraud Detection data into data/raw.
# Prereqs:
#   1. pip install kaggle
#   2. Kaggle credentials: copy .env.example to .env and fill in KAGGLE_USERNAME / KAGGLE_KEY
#      (~/.kaggle/kaggle.json also works if .env isn't present)
#   3. Join the competition and accept its rules:
#      https://www.kaggle.com/competitions/ieee-fraud-detection/rules
# The raw data is git-ignored: competition rules don't allow redistributing it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load credentials from .env if present (exported so the kaggle CLI sees them).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]] && [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo "No Kaggle credentials found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip -o data/raw/ieee-fraud-detection.zip -d data/raw
rm data/raw/ieee-fraud-detection.zip
ls -lh data/raw
