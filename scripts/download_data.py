"""Download the IEEE-CIS Fraud Detection data into data/raw (works on Windows, macOS, Linux).

Prereqs:
  1. pip install kaggle
  2. Kaggle credentials: copy .env.example to .env and fill in KAGGLE_USERNAME / KAGGLE_KEY
     (~/.kaggle/kaggle.json also works if .env isn't present)
  3. Join the competition and accept its rules:
     https://www.kaggle.com/competitions/ieee-fraud-detection/rules

The raw data is git-ignored: competition rules don't allow redistributing it.
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
COMPETITION = "ieee-fraud-detection"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def main() -> None:
    load_dotenv(ROOT / ".env")

    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_json):
        sys.exit("No Kaggle credentials found. Copy .env.example to .env and fill it in.")

    # Import after credentials are in the environment: the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} to {RAW_DIR} (~1.3 GB)...")
    try:
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as e:  # noqa: BLE001 - most common cause: rules not accepted yet (403)
        sys.exit(
            f"Download failed: {e}\nIf this is a 403, join the competition and accept the rules at "
            f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )

    archive = RAW_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink()

    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
