#!/usr/bin/env bash
# Downloads the IEEE-CIS Fraud Detection data into data/raw.
# Prereqs:
#   1. pip install kaggle and put your API token at ~/.kaggle/kaggle.json
#   2. Join the competition and accept its rules:
#      https://www.kaggle.com/competitions/ieee-fraud-detection/rules
# The raw data is git-ignored: competition rules don't allow redistributing it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip -o data/raw/ieee-fraud-detection.zip -d data/raw
rm data/raw/ieee-fraud-detection.zip
ls -lh data/raw
