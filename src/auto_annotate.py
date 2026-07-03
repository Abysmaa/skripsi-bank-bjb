from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

try:
    from .utils import PROJECT_ROOT, configure_logging, log
except ImportError:
    from utils import PROJECT_ROOT, configure_logging, log


BATCH_PATH = PROJECT_ROOT / "data" / "processed" / "annotation_batches" / "batch_001.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "processed" / "annotation_batches" / "batch_001_backup.csv"
VALID_LABELS = {"negatif", "netral", "positif"}

STRONG_NEGATIVE_PHRASES = [
    "gagal",
    "error",
    "bobol",
    "hack",
    "hilang",
    "crash",
    "force close",
    "otp tidak masuk",
    "transaksi gagal",
    "dana hilang",
    "tidak bisa login",
    "tidak bisa transfer",
    "belum berhasil",
]

NEGATIVE_PHRASES = [
    "gagal",
    "error",
    "tidak bisa",
    "tidak berfungsi",
    "lambat",
    "lemot",
    "crash",
    "force close",
    "tidak aman",
    "bobol",
    "hack",
    "kecewa",
    "buruk",
    "jelek",
    "susah",
    "ribet",
    "lama",
    "tidak masuk",
    "tidak bisa login",
    "otp tidak masuk",
    "sering error",
    "aplikasi bermasalah",
    "tidak berguna",
    "mengecewakan",
    "tidak responsif",
    "blank",
    "loading lama",
    "transaksi gagal",
    "dana hilang",
    "tidak bisa transfer",
    "saldo tidak update",
    "tidak bisa buka",
    "belum berhasil",
    "kurang bagus",
]

POSITIVE_PHRASES = [
    "bagus",
    "baik",
    "mudah",
    "lancar",
    "cepat",
    "aman",
    "nyaman",
    "recommended",
    "membantu",
    "mantap",
    "keren",
    "canggih",
    "memuaskan",
    "senang",
    "suka",
    "praktis",
    "simple",
    "fitur lengkap",
    "lengkap",
    "tidak ada masalah",
    "berhasil",
    "sangat membantu",
    "luar biasa",
    "top",
    "terbaik",
    "sempurna",
    "memudahkan",
]


def contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None


def count_matches(text: str, phrases: list[str]) -> int:
    return sum(1 for phrase in phrases if contains_phrase(text, phrase))


def normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in VALID_LABELS else ""


def detect_text_label(clean_text: object) -> str:
    text = str(clean_text or "").strip().lower()
    if not text:
        return "netral"

    word_count = len(text.split())
    strong_negative_hits = count_matches(text, STRONG_NEGATIVE_PHRASES)
    negative_hits = count_matches(text, NEGATIVE_PHRASES)
    positive_hits = count_matches(text, POSITIVE_PHRASES)

    if re.search(r"\b(belum|tidak|ga|gak|nggak)\s+berhasil\b", text):
        strong_negative_hits += 1
        negative_hits += 1

    if re.search(r"\b(kurang|tidak|ga|gak|nggak)\s+bagus\b", text):
        negative_hits += 1

    if strong_negative_hits > 0:
        return "negatif"

    if negative_hits > 0 and positive_hits == 0:
        return "negatif"

    if positive_hits > 0 and negative_hits == 0:
        return "positif"

    if negative_hits > 0 and positive_hits > 0:
        if negative_hits > positive_hits:
            return "negatif"
        if positive_hits > negative_hits:
            return "positif"
        return "netral"

    if word_count < 5:
        return "netral"

    return "netral"


def auto_annotate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working_df = df.copy().fillna("")
    if "manual_label" not in working_df.columns:
        working_df["manual_label"] = ""

    corrections: list[dict[str, str]] = []
    manual_labels: list[str] = []

    for _, row in working_df.iterrows():
        detected_label = detect_text_label(row.get("clean_content", ""))
        semi_label = normalize_label(row.get("semi_label", ""))
        manual_label = ""

        if detected_label != semi_label:
            manual_label = detected_label
            corrections.append(
                {
                    "review_id": str(row.get("review_id", "")),
                    "semi_label": semi_label,
                    "manual_label": manual_label,
                    "clean_content": str(row.get("clean_content", "")),
                }
            )

        manual_labels.append(manual_label)

    working_df["manual_label"] = manual_labels
    corrections_df = pd.DataFrame(corrections)
    return working_df, corrections_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isi manual_label batch anotasi secara otomatis berdasarkan aturan kata kunci."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BATCH_PATH,
        help=f"Path file batch anotasi. Default: {BATCH_PATH}",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=BACKUP_PATH,
        help=f"Path file backup sebelum overwrite. Default: {BACKUP_PATH}",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.backup)
    log(f"Backup batch created: {args.backup}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    updated_df, corrections_df = auto_annotate(df)
    updated_df.to_csv(args.input, index=False, encoding="utf-8-sig")

    total_rows = len(updated_df)
    corrected_count = len(corrections_df)
    empty_count = int((updated_df["manual_label"].fillna("").astype(str).str.strip() == "").sum())

    log(f"Auto annotation written: {args.input}")
    log(f"Total baris diproses: {total_rows}")
    log(f"Jumlah manual_label diisi: {corrected_count}")
    log(f"Jumlah manual_label kosong: {empty_count}")

    if corrected_count > 0:
        distribution = (
            corrections_df["manual_label"]
            .value_counts()
            .sort_index()
            .to_string()
        )
        log(f"Distribusi manual_label terisi:\n{distribution}")

        preview_df = corrections_df.head(5).copy()
        preview_df["text_preview"] = preview_df["clean_content"].apply(lambda text: str(text)[:120])
        log("Contoh 5 baris yang dikoreksi:")
        for _, row in preview_df.iterrows():
            log(
                f"{row['review_id']} | {row['semi_label']} -> {row['manual_label']} | "
                f"{row['text_preview']}"
            )
    else:
        log("Tidak ada koreksi otomatis. Semua semi_label dianggap sudah sesuai.")


if __name__ == "__main__":
    main()
