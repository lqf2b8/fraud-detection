"""Download the IEEE-CIS Fraud Detection data into data/raw (works on Windows, macOS, Linux).

Prereqs:
  1. pip install kaggle
  2. Kaggle credentials: copy .env.example to .env and fill in either KAGGLE_API_TOKEN (new
     tokens) or KAGGLE_USERNAME + KAGGLE_KEY (legacy). ~/.kaggle/kaggle.json also works.
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

    kaggle_dir = Path.home() / ".kaggle"
    has_creds = any(
        [
            os.getenv("KAGGLE_API_TOKEN"),
            os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"),
            (kaggle_dir / "kaggle.json").exists(),
            (kaggle_dir / "access_token").exists(),
        ]
    )
    if not has_creds:
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
        msg = f"Download failed: {e}\n"
        if "401" in str(e):
            msg += (
                "401 = credentials rejected. A token from 'Generate New Token' goes in "
                "KAGGLE_API_TOKEN, not KAGGLE_KEY. Check .env for typos/extra spaces, or "
                "generate a fresh token."
            )
        elif "403" in str(e):
            msg += (
                "403 = no access. Join the competition and accept the rules at "
                f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
            )
        sys.exit(msg)

    archive = RAW_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink()

    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
