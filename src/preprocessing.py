from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from .utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, configure_logging, ensure_parent_dir, log
except ImportError:
    from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, configure_logging, ensure_parent_dir, log


OUTPUT_COLUMNS = [
    "review_id",
    "user_name",
    "score",
    "content",
    "clean_content",
    "at",
    "app_version",
    "inferred_label",
    "is_security",
]

SECURITY_PATTERNS = [
    "login",
    "password",
    "otp",
    "pin",
    "verifikasi",
    "autentikasi",
    "hack",
    "bobol",
    "keamanan",
    "aman",
    "tidak aman",
    "gagal masuk",
    "akun",
    "sidik jari",
    "fingerprint",
    "face id",
    "enkripsi",
    "keamanan data",
    "token",
    "sandi",
    "lupa password",
    "ganti password",
    "2fa",
]


def clean_text(text: object) -> str:
    cleaned = str(text or "").lower()
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"#\w+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def preprocess_text(text: object) -> str:
    return clean_text(text)


def infer_label_from_score(score: object) -> str:
    try:
        score_value = int(float(score))
    except (TypeError, ValueError):
        return ""

    if score_value in (1, 2):
        return "negatif"
    if score_value == 3:
        return "netral"
    if score_value in (4, 5):
        return "positif"
    return ""


def contains_security_keyword(text: object) -> bool:
    normalized_text = f" {str(text or '').strip()} "
    return any(f" {keyword} " in normalized_text for keyword in SECURITY_PATTERNS)


def preprocess_dataframe(df: pd.DataFrame, infer_label: bool) -> pd.DataFrame:
    working_df = df.copy()

    for column in ["review_id", "user_name", "score", "content", "at", "app_version"]:
        if column not in working_df.columns:
            working_df[column] = ""

    working_df["content"] = working_df["content"].fillna("").astype(str)
    working_df["clean_content"] = working_df["content"].apply(clean_text)
    working_df = working_df[working_df["clean_content"].str.strip() != ""].copy()

    if infer_label:
        working_df["inferred_label"] = working_df["score"].apply(infer_label_from_score)
    else:
        working_df["inferred_label"] = ""

    working_df["is_security"] = working_df["clean_content"].apply(contains_security_keyword)
    working_df = working_df[OUTPUT_COLUMNS].copy()
    working_df = working_df.drop_duplicates(subset=["review_id"], keep="first")
    return working_df.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess ulasan Bank BJB dan tandai review terkait aspek keamanan."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help=f"Path CSV input. Default: {RAW_DATA_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help=f"Path CSV output. Default: {PROCESSED_DATA_PATH}",
    )
    parser.add_argument(
        "--infer-label",
        action="store_true",
        help="Infer label otomatis dari score ke kolom inferred_label.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    processed_df = preprocess_dataframe(df, infer_label=args.infer_label)

    ensure_parent_dir(args.output)
    processed_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    log(f"Preprocessed data written: {args.output}")
    log(f"Total baris: {len(processed_df)}")
    log(
        "Distribusi inferred_label:\n"
        f"{processed_df['inferred_label'].value_counts(dropna=False).sort_index().to_string()}"
    )
    log(f"Jumlah ulasan is_security=True: {int(processed_df['is_security'].sum())}")


if __name__ == "__main__":
    main()
