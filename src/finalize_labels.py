from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .labeling import ANNOTATION_BATCH_DIR, LABELED_DATA_PATH
    from .utils import DATA_DIR, configure_logging, ensure_parent_dir, log
except ImportError:
    from labeling import ANNOTATION_BATCH_DIR, LABELED_DATA_PATH
    from utils import DATA_DIR, configure_logging, ensure_parent_dir, log


FINAL_LABEL_PATH = DATA_DIR / "processed" / "bjb_reviews_labeled_final.csv"
VALID_LABELS = {"negatif", "netral", "positif"}
FINAL_COLUMNS = ["review_id", "clean_content", "final_label", "score", "at", "is_security"]


def normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in VALID_LABELS else ""


def load_annotation_batches(batch_dir: Path) -> pd.DataFrame:
    batch_files = sorted(batch_dir.glob("*.csv"))
    if not batch_files:
        return pd.DataFrame(columns=["review_id", "manual_label"])

    batch_frames = []
    for batch_file in batch_files:
        batch_df = pd.read_csv(batch_file, encoding="utf-8-sig").fillna("")
        if "review_id" not in batch_df.columns:
            continue
        if "manual_label" not in batch_df.columns:
            batch_df["manual_label"] = ""
        batch_frames.append(batch_df[["review_id", "manual_label"]].copy())

    if not batch_frames:
        return pd.DataFrame(columns=["review_id", "manual_label"])

    merged_batches = pd.concat(batch_frames, ignore_index=True)
    merged_batches["manual_label"] = merged_batches["manual_label"].apply(normalize_label)
    merged_batches = merged_batches.drop_duplicates(subset=["review_id"], keep="last")
    return merged_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalisasi label dengan menggabungkan manual_label dari batch anotasi."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=LABELED_DATA_PATH,
        help=f"Path CSV input labeled. Default: {LABELED_DATA_PATH}",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=ANNOTATION_BATCH_DIR,
        help=f"Folder batch anotasi. Default: {ANNOTATION_BATCH_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FINAL_LABEL_PATH,
        help=f"Path CSV output final. Default: {FINAL_LABEL_PATH}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Parameter dokumentatif untuk ukuran batch anotasi manual.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    labeled_df = pd.read_csv(args.input, encoding="utf-8-sig").fillna("")
    batch_df = load_annotation_batches(args.batch_dir)

    labeled_df["semi_label"] = labeled_df.get("semi_label", "").apply(normalize_label)
    merged_df = labeled_df.merge(
        batch_df,
        on="review_id",
        how="left",
        suffixes=("", "_batch"),
    )

    if "manual_label_batch" in merged_df.columns:
        merged_df["manual_label"] = merged_df["manual_label_batch"]
        merged_df = merged_df.drop(columns=["manual_label_batch"])
    elif "manual_label" not in merged_df.columns:
        merged_df["manual_label"] = ""

    merged_df["manual_label"] = merged_df["manual_label"].apply(normalize_label)
    merged_df["final_label"] = merged_df["manual_label"]
    empty_final = merged_df["final_label"].str.strip() == ""
    merged_df.loc[empty_final, "final_label"] = merged_df.loc[empty_final, "semi_label"]
    merged_df = merged_df[merged_df["final_label"].str.strip() != ""].copy()

    for column in FINAL_COLUMNS:
        if column not in merged_df.columns:
            merged_df[column] = ""

    final_df = merged_df[FINAL_COLUMNS].drop_duplicates(subset=["review_id"], keep="first")

    ensure_parent_dir(args.output)
    final_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    log(f"Batch size parameter: {args.batch_size}")
    log(f"Final labeled dataset written: {args.output}")
    log(
        "Distribusi final_label:\n"
        f"{final_df['final_label'].value_counts().sort_index().to_string()}"
    )
    log(f"Total baris final: {len(final_df)}")


if __name__ == "__main__":
    main()
