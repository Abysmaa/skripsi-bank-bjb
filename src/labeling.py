from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .utils import DATA_DIR, PROCESSED_DATA_PATH, configure_logging, ensure_parent_dir, log
except ImportError:
    from utils import DATA_DIR, PROCESSED_DATA_PATH, configure_logging, ensure_parent_dir, log


LABELED_DATA_PATH = DATA_DIR / "processed" / "bjb_reviews_labeled.csv"
ANNOTATION_BATCH_DIR = DATA_DIR / "processed" / "annotation_batches"
ANNOTATION_BATCH_PATH = ANNOTATION_BATCH_DIR / "batch_001.csv"
BATCH_COLUMNS = ["review_id", "clean_content", "score", "semi_label", "manual_label"]


def sample_annotation_rows(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    neutral_df = df[df["score_numeric"] == 3].copy()
    low_score_df = df[df["score_numeric"].isin([1, 2])].copy()
    high_score_df = df[df["score_numeric"].isin([4, 5])].copy()

    low_sample = low_score_df.sample(frac=0.1, random_state=random_state) if not low_score_df.empty else low_score_df
    high_sample = high_score_df.sample(frac=0.1, random_state=random_state) if not high_score_df.empty else high_score_df

    annotation_df = pd.concat([neutral_df, low_sample, high_sample], ignore_index=True)
    annotation_df = annotation_df.drop_duplicates(subset=["review_id"], keep="first")
    annotation_df = annotation_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return annotation_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Siapkan semi-label dan batch anotasi manual untuk ulasan Bank BJB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help=f"Path CSV input processed. Default: {PROCESSED_DATA_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LABELED_DATA_PATH,
        help=f"Path CSV output labeled. Default: {LABELED_DATA_PATH}",
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=ANNOTATION_BATCH_PATH,
        help=f"Path CSV output batch anotasi. Default: {ANNOTATION_BATCH_PATH}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed untuk sampling acak anotasi manual.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig").fillna("")
    df["semi_label"] = df["inferred_label"].astype(str)
    df["manual_label"] = ""
    df["score_numeric"] = pd.to_numeric(df["score"], errors="coerce")

    annotation_df = sample_annotation_rows(df, random_state=args.random_state)

    ensure_parent_dir(args.output)
    df.drop(columns=["score_numeric"]).to_csv(args.output, index=False, encoding="utf-8-sig")

    ensure_parent_dir(args.batch_output)
    for existing_batch in args.batch_output.parent.glob("*.csv"):
        existing_batch.unlink()
    annotation_df[BATCH_COLUMNS].to_csv(args.batch_output, index=False, encoding="utf-8-sig")

    log(f"Labeled dataset written: {args.output}")
    log(f"Annotation batch written: {args.batch_output}")
    log(f"Jumlah total data: {len(df)}")
    log(f"Jumlah sampel anotasi: {len(annotation_df)}")


if __name__ == "__main__":
    main()
