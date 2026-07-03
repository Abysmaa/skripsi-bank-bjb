from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from google_play_scraper import Sort, reviews
except ImportError:  # pragma: no cover - dependency is optional at edit time
    Sort = None
    reviews = None

try:
    from .utils import RAW_DATA_PATH, ensure_parent_dir, log
except ImportError:
    from utils import RAW_DATA_PATH, ensure_parent_dir, log


APP_ID = "com.bjbsmb"
LANG = "id"
COUNTRY = "id"
DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 12, 31)
REQUEST_DELAY_SECONDS = 1.5
CHECKPOINT_PATH = RAW_DATA_PATH.with_name("bjb_reviews_raw_checkpoint.json")
OUTPUT_COLUMNS = [
    "review_id",
    "user_name",
    "score",
    "content",
    "at",
    "app_version",
    "thumbs_up_count",
    "reply_content",
    "replied_at",
]


@dataclass(slots=True)
class RawReview:
    review_id: str
    user_name: str
    score: int
    content: str
    at: str
    app_version: str
    thumbs_up_count: int
    reply_content: str
    replied_at: str


def format_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value or "")


def parse_review_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value

    if not value:
        raise ValueError("Nilai `at` kosong.")

    text = str(value).strip()
    text = text.replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def parse_review(raw_review: dict[str, Any]) -> tuple[RawReview, datetime]:
    review_datetime = parse_review_datetime(raw_review.get("at"))
    parsed_review = RawReview(
        review_id=str(raw_review.get("reviewId", "")),
        user_name=str(raw_review.get("userName", "")),
        score=int(raw_review.get("score", 0) or 0),
        content=str(raw_review.get("content", "") or "").strip(),
        at=format_datetime(review_datetime),
        app_version=str(
            raw_review.get("reviewCreatedVersion")
            or raw_review.get("appVersion")
            or ""
        ),
        thumbs_up_count=int(raw_review.get("thumbsUpCount", 0) or 0),
        reply_content=str(raw_review.get("replyContent", "") or "").strip(),
        replied_at=format_datetime(raw_review.get("repliedAt")),
    )
    return parsed_review, review_datetime


def validate_dependencies() -> None:
    if reviews is None or Sort is None:
        raise ImportError(
            "Package `google-play-scraper` belum terpasang. "
            "Install dependency dari requirements.txt terlebih dahulu."
        )


def serialize_continuation_token(token: Any) -> dict[str, object] | None:
    if token is None:
        return None

    return {
        "token": getattr(token, "token", None),
        "lang": getattr(token, "lang", LANG),
        "country": getattr(token, "country", COUNTRY),
        "sort": getattr(token, "sort", Sort.NEWEST.value),
        "count": getattr(token, "count", 200),
        "filter_score_with": getattr(token, "filter_score_with", None),
        "filter_device_with": getattr(token, "filter_device_with", None),
    }


def deserialize_continuation_token(payload: dict[str, object] | None) -> Any:
    if not payload:
        return None

    reviews_globals = getattr(reviews, "__globals__", {})
    continuation_class = reviews_globals.get("_ContinuationToken")
    if continuation_class is None:
        raise RuntimeError("Tidak dapat memulihkan continuation token dari package.")

    return continuation_class(
        payload.get("token"),
        payload.get("lang", LANG),
        payload.get("country", COUNTRY),
        payload.get("sort", Sort.NEWEST.value),
        payload.get("count", 200),
        payload.get("filter_score_with"),
        payload.get("filter_device_with"),
    )


def load_checkpoint(checkpoint_path: Path) -> tuple[Any, dict[str, dict[str, object]], int]:
    if not checkpoint_path.exists():
        return None, {}, 0

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    saved_rows = checkpoint.get("collected_reviews", [])
    collected_map = {
        str(row["review_id"]): {column: row.get(column, "") for column in OUTPUT_COLUMNS}
        for row in saved_rows
        if row.get("review_id")
    }
    continuation_token = deserialize_continuation_token(
        checkpoint.get("continuation_token")
    )
    scanned_count = int(checkpoint.get("scanned_count", 0) or 0)
    return continuation_token, collected_map, scanned_count


def save_checkpoint(
    checkpoint_path: Path,
    continuation_token: str | None,
    collected_rows: list[dict[str, object]],
    scanned_count: int,
) -> None:
    ensure_parent_dir(checkpoint_path)
    payload = {
        "app_id": APP_ID,
        "lang": LANG,
        "country": COUNTRY,
        "sort": "NEWEST",
        "date_start": DATE_START.isoformat(),
        "date_end": DATE_END.isoformat(),
        "scanned_count": scanned_count,
        "matched_count": len(collected_rows),
        "continuation_token": serialize_continuation_token(continuation_token),
        "collected_reviews": collected_rows,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    checkpoint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def remove_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def fetch_reviews_in_date_range(
    app_id: str,
    lang: str,
    country: str,
    checkpoint_path: Path,
    sleep_seconds: float,
) -> list[dict[str, object]]:
    validate_dependencies()

    continuation_token, collected_map, scanned_count = load_checkpoint(checkpoint_path)
    if collected_map:
        log(
            "Checkpoint ditemukan. "
            f"Melanjutkan scraping dengan {len(collected_map)} review tersimpan."
        )

    progress_bar = tqdm(
        total=None,
        desc="Scraping Google Play reviews",
        unit="review",
        initial=scanned_count,
    )
    stop_scraping = False

    try:
        while True:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token,
            )

            if not result:
                log("Tidak ada batch ulasan baru dari Google Play.")
                break

            progress_bar.update(len(result))
            batch_oldest_date: str | None = None

            for raw_review in result:
                parsed_review, review_datetime = parse_review(raw_review)
                review_date = review_datetime.date()
                batch_oldest_date = review_date.isoformat()

                if review_date < DATE_START:
                    stop_scraping = True
                    break

                if review_date > DATE_END:
                    continue

                collected_map[parsed_review.review_id] = asdict(parsed_review)

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                continuation_token=continuation_token,
                collected_rows=list(collected_map.values()),
                scanned_count=progress_bar.n,
            )
            progress_bar.set_postfix(
                in_range=len(collected_map),
                oldest=batch_oldest_date or "-",
            )

            if stop_scraping:
                log(
                    "Scraping dihentikan karena sudah mencapai ulasan yang lebih tua "
                    f"dari {DATE_START.isoformat()}."
                )
                break

            if continuation_token is None:
                log("Continuation token habis. Semua batch yang tersedia sudah diambil.")
                break

            time.sleep(sleep_seconds)
    finally:
        progress_bar.close()

    deduped_rows = list(collected_map.values())
    log(f"Total review dalam rentang tanggal: {len(deduped_rows)}")
    return deduped_rows


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = ""
    return aligned[OUTPUT_COLUMNS]


def merge_and_save_reviews(
    existing_file_path: Path,
    new_rows: list[dict[str, object]],
) -> tuple[int, int, int]:
    new_df = pd.DataFrame(new_rows)
    new_df = align_columns(new_df)

    if existing_file_path.exists():
        existing_df = pd.read_csv(existing_file_path, dtype=str).fillna("")
        existing_df = align_columns(existing_df)
    else:
        existing_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["review_id"], keep="first")

    combined_df["at_sort"] = pd.to_datetime(combined_df["at"], errors="coerce")
    combined_df = combined_df.sort_values("at_sort", ascending=False, na_position="last")
    combined_df = combined_df.drop(columns=["at_sort"])
    combined_df.to_csv(existing_file_path, index=False, encoding="utf-8")

    return len(new_df), combined_before_dedup, len(combined_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape ulasan Google Play Store aplikasi Bank BJB pada rentang "
            "2024-01-01 sampai 2025-12-31, lalu merge ke CSV mentah yang ada."
        )
    )
    parser.add_argument(
        "--app-id",
        default=APP_ID,
        help=f"App ID Google Play. Default: {APP_ID}",
    )
    parser.add_argument(
        "--lang",
        default=LANG,
        help=f"Kode bahasa ulasan. Default: {LANG}",
    )
    parser.add_argument(
        "--country",
        default=COUNTRY,
        help=f"Kode negara Google Play. Default: {COUNTRY}",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_PATH),
        help="Path file checkpoint JSON untuk resume scraping.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help=f"Jeda antar request. Default: {REQUEST_DELAY_SECONDS} detik.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    new_rows = fetch_reviews_in_date_range(
        app_id=args.app_id,
        lang=args.lang,
        country=args.country,
        checkpoint_path=checkpoint_path,
        sleep_seconds=args.sleep_seconds,
    )

    new_df = pd.DataFrame(new_rows)
    if not new_df.empty:
        new_df = align_columns(new_df)
        before_dedup = len(new_df)
        new_df = new_df.drop_duplicates(subset=["review_id"], keep="first")
        new_rows = new_df.to_dict(orient="records")
        log(
            "Dedup hasil scraping selesai. "
            f"Sebelum: {before_dedup}, sesudah: {len(new_rows)}."
        )
    else:
        log("Tidak ada review 2024-2025 yang ditemukan pada scraping ini.")

    new_count, merged_before_dedup, merged_after_dedup = merge_and_save_reviews(
        existing_file_path=RAW_DATA_PATH,
        new_rows=new_rows,
    )
    remove_checkpoint(checkpoint_path)

    log(f"Review baru dalam rentang 2024-2025: {new_count}")
    log(f"Total baris gabungan sebelum dedup: {merged_before_dedup}")
    log(f"Total baris gabungan setelah dedup: {merged_after_dedup}")
    log(f"CSV final tersimpan di: {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
