from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .labeling import ANNOTATION_BATCH_DIR
    from .utils import configure_logging, log
except ImportError:
    from labeling import ANNOTATION_BATCH_DIR
    from utils import configure_logging, log


VALID_LABELS = {"negatif", "netral", "positif"}


def normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in VALID_LABELS else ""


def summarize_batch(batch_path: Path) -> dict[str, object]:
    df = pd.read_csv(batch_path, encoding="utf-8-sig").fillna("")
    if "manual_label" not in df.columns:
        df["manual_label"] = ""

    df["manual_label_normalized"] = df["manual_label"].apply(normalize_label)
    filled_mask = df["manual_label_normalized"] != ""
    filled_count = int(filled_mask.sum())
    total_count = int(len(df))
    pending_count = total_count - filled_count

    summary = {
        "batch_name": batch_path.name,
        "total": total_count,
        "filled": filled_count,
        "pending": pending_count,
        "progress_pct": (filled_count / total_count * 100) if total_count else 0.0,
        "distribution": df.loc[filled_mask, "manual_label_normalized"].value_counts().sort_index(),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cek progres pengisian manual_label pada file batch anotasi."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=ANNOTATION_BATCH_DIR,
        help=f"Folder batch anotasi. Default: {ANNOTATION_BATCH_DIR}",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.batch_dir.exists():
        raise FileNotFoundError(f"Folder batch tidak ditemukan: {args.batch_dir}")

    batch_files = sorted(
        batch_file
        for batch_file in args.batch_dir.glob("*.csv")
        if not batch_file.stem.endswith("_backup")
    )
    if not batch_files:
        log(f"Tidak ada file batch di: {args.batch_dir}")
        return

    total_rows = 0
    total_filled = 0

    for batch_file in batch_files:
        summary = summarize_batch(batch_file)
        total_rows += int(summary["total"])
        total_filled += int(summary["filled"])

        log(
            f"{summary['batch_name']}: total={summary['total']}, "
            f"filled={summary['filled']}, pending={summary['pending']}, "
            f"progress={summary['progress_pct']:.2f}%"
        )
        distribution = summary["distribution"]
        if len(distribution) > 0:
            log(f"Distribusi manual_label {summary['batch_name']}:\n{distribution.to_string()}")

    total_pending = total_rows - total_filled
    total_progress = (total_filled / total_rows * 100) if total_rows else 0.0
    log(
        f"Total semua batch: total={total_rows}, filled={total_filled}, "
        f"pending={total_pending}, progress={total_progress:.2f}%"
    )


if __name__ == "__main__":
    main()
