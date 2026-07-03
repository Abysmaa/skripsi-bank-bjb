from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bjb_reviews_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "bjb_reviews_clean.csv"
FINAL_DATA_PATH = DATA_DIR / "final" / "bjb_sentiment_result.csv"
INDOBERT_MODEL_DIR = PROJECT_ROOT / "models" / "indobert"
LDA_MODEL_DIR = PROJECT_ROOT / "models" / "lda"


def configure_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log(message: str) -> None:
    logging.getLogger("bank_bjb").info(message)


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    if not file_path.exists():
        log(f"CSV not found: {file_path}")
        return []

    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def write_csv_rows(
    file_path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    ensure_parent_dir(file_path)

    with file_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log(f"CSV written: {file_path}")
