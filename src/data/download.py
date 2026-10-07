"""Download the IBM AML dataset (HI-Small) from Kaggle.

This module wraps the Kaggle CLI to fetch the dataset into data/raw/.
Idempotent: skips download if files already present.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
EXPECTED_MARKER = "HI-Small_Trans.csv"


def _check_kaggle_cli() -> None:
    try:
        subprocess.run(
            ["kaggle", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("Kaggle CLI not available. Run: uv sync --extra dev")
        raise SystemExit(1) from exc


def _already_downloaded() -> bool:
    marker = RAW_DIR / EXPECTED_MARKER
    return marker.exists()


def _extract_zip() -> None:
    """Extract only HI-Small_* files (the rest is 40+ GB of unused data)."""
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        logger.warning("No .zip found to extract.")
        return

    import fnmatch

    for archive in zips:
        logger.info(f"Extracting HI-Small files from {archive.name} ...")
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.namelist():
                if fnmatch.fnmatch(member, "HI-Small*"):
                    logger.info(f"  -> {member}")
                    zf.extract(member, RAW_DIR, pwd=None)
        archive.unlink()
        logger.info(f"Removed archive {archive.name}")

def download() -> Path:
    """Download the dataset and return the raw directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if _already_downloaded():
        logger.info(f"Dataset already present at {RAW_DIR}. Skipping download.")
        return RAW_DIR

    _check_kaggle_cli()

    logger.info(f"Downloading dataset: {DATASET_SLUG}")
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET_SLUG,
                "-p",
                str(RAW_DIR),
                "--force",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle download failed. Check ~/.kaggle/ credentials.")
        raise SystemExit(1) from exc

    _extract_zip()
    logger.success(f"Dataset ready in {RAW_DIR}")
    return RAW_DIR


def main() -> None:
    download()


if __name__ == "__main__":
    sys.exit(main())
