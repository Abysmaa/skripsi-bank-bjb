from __future__ import annotations

import argparse
from math import exp

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - dependency is optional at edit time
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from .utils import (
        FINAL_DATA_PATH,
        INDOBERT_MODEL_DIR,
        PROCESSED_DATA_PATH,
        log,
        read_csv_rows,
        write_csv_rows,
    )
except ImportError:
    from utils import (
        FINAL_DATA_PATH,
        INDOBERT_MODEL_DIR,
        PROCESSED_DATA_PATH,
        log,
        read_csv_rows,
        write_csv_rows,
    )


SENTIMENT_FIELDS = [
    "review_id",
    "score",
    "content",
    "content_clean",
    "true_label",
    "predicted_label",
    "confidence",
]


def prepare_prediction_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(PROCESSED_DATA_PATH)
    normalized_rows: list[dict[str, str]] = []

    for row in rows:
        normalized_row = dict(row)
        if not str(normalized_row.get("content_clean", "") or "").strip():
            normalized_row["content_clean"] = str(
                normalized_row.get("clean_content", "") or ""
            )
        if not str(normalized_row.get("label", "") or "").strip():
            normalized_row["label"] = str(
                normalized_row.get("final_label")
                or normalized_row.get("inferred_label")
                or ""
            )
        normalized_rows.append(normalized_row)

    prediction_rows = [
        row
        for row in normalized_rows
        if str(row.get("content_clean", "") or "").strip()
    ]
    log(f"Prediction rows available: {len(prediction_rows)}")
    return prediction_rows


def require_transformers_dependencies() -> None:
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ImportError(
            "Dependency `transformers` dan `torch` belum tersedia. "
            "Install dependency dari requirements.txt terlebih dahulu."
        )


def load_model_components(model_dir=INDOBERT_MODEL_DIR):
    require_transformers_dependencies()

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Folder model tidak ditemukan: {model_dir}. "
            "Latih atau simpan model IndoBERT terlebih dahulu."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    log(f"Loaded model from: {model_dir}")
    log(f"Inference device: {device}")
    return tokenizer, model, device


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def resolve_id2label(model) -> dict[int, str]:
    config_labels = getattr(model.config, "id2label", {}) or {}
    id2label: dict[int, str] = {}

    for key, value in config_labels.items():
        try:
            numeric_key = int(key)
        except (TypeError, ValueError):
            numeric_key = key
        id2label[numeric_key] = str(value).lower()

    if not id2label:
        label_count = int(getattr(model.config, "num_labels", 0) or 0)
        id2label = {index: f"label_{index}" for index in range(label_count)}

    return id2label


def predict_sentiments(
    rows: list[dict[str, str]],
    batch_size: int = 16,
) -> list[dict[str, str]]:
    if not rows:
        return []

    tokenizer, model, device = load_model_components()
    id2label = resolve_id2label(model)
    results: list[dict[str, str]] = []

    for start_index in range(0, len(rows), batch_size):
        batch_rows = rows[start_index : start_index + batch_size]
        batch_texts = [row.get("content_clean", "") for row in batch_rows]
        encoded_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded_batch = {
            key: value.to(device)
            for key, value in encoded_batch.items()
        }

        with torch.no_grad():
            outputs = model(**encoded_batch)

        batch_logits = outputs.logits.detach().cpu().tolist()

        for row, logits in zip(batch_rows, batch_logits):
            probabilities = softmax([float(value) for value in logits])
            predicted_id = max(
                range(len(probabilities)),
                key=probabilities.__getitem__,
            )
            results.append(
                {
                    "review_id": row.get("review_id", ""),
                    "score": row.get("score", ""),
                    "content": row.get("content", ""),
                    "content_clean": row.get("content_clean", ""),
                    "true_label": row.get("label", ""),
                    "predicted_label": id2label.get(predicted_id, str(predicted_id)),
                    "confidence": f"{probabilities[predicted_id]:.4f}",
                }
            )

    log(f"Prediction completed: {len(results)} rows")
    return results


def export_sentiment_results(rows: list[dict[str, str]]) -> None:
    write_csv_rows(FINAL_DATA_PATH, SENTIMENT_FIELDS, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jalankan inferensi sentimen IndoBERT dan simpan hasil ke data/final."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Ukuran batch untuk inferensi.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = prepare_prediction_rows()
    predicted_rows = predict_sentiments(rows, batch_size=args.batch_size)
    export_sentiment_results(predicted_rows)
    log(f"Sentiment results saved: {FINAL_DATA_PATH}")


if __name__ == "__main__":
    main()
