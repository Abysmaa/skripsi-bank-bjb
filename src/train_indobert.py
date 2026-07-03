from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from .preprocessing import infer_label_from_score
    from .utils import INDOBERT_MODEL_DIR, PROCESSED_DATA_PATH, log
except ImportError:
    from preprocessing import infer_label_from_score
    from utils import INDOBERT_MODEL_DIR, PROCESSED_DATA_PATH, log


MODEL_NAME = "indobenchmark/indobert-base-p1"
LABEL_ALIASES = {
    "positif": "positive",
    "positive": "positive",
    "negatif": "negative",
    "negative": "negative",
    "netral": "neutral",
    "neutral": "neutral",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tuning IndoBERT untuk klasifikasi sentimen review Bank BJB."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Jumlah epoch training.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate optimizer.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Batch size training per device.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size evaluasi per device.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Panjang token maksimal.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Gunakan klasifikasi biner dengan membuang label neutral.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proporsi data validation dari sisa data train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk split dataset.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Aktifkan mixed precision fp16 jika GPU mendukung.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_PATH),
        help="Path CSV input untuk training.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Nama kolom label yang digunakan untuk training.",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=["none", "undersample", "oversample"],
        default="none",
        help="Strategi balancing dataset sebelum split.",
    )
    return parser


def load_training_dataframe(
    input_path: Path,
    label_column: str,
    binary: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(input_path)
    if "content_clean" not in df.columns and "clean_content" in df.columns:
        df["content_clean"] = df["clean_content"]
    if "content" not in df.columns and "clean_content" in df.columns:
        df["content"] = df["clean_content"]

    df["content_clean"] = df["content_clean"].fillna("").astype(str)

    if label_column not in df.columns:
        df["label"] = df["score"].apply(infer_label_from_score)
    else:
        df["label"] = (
            df[label_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(LABEL_ALIASES)
        )
        missing_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
        df.loc[missing_mask, "label"] = df.loc[missing_mask, "score"].apply(
            infer_label_from_score
        )

    df = df[df["content_clean"].str.strip() != ""].copy()
    valid_labels = ["negative", "neutral", "positive"]
    df = df[df["label"].isin(valid_labels)].copy()

    if binary:
        df = df[df["label"] != "neutral"].copy()
        label_list = ["negative", "positive"]
    else:
        label_list = valid_labels

    df = df.drop_duplicates(subset=["content_clean", "label"]).reset_index(drop=True)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    df["labels"] = df["label"].map(label2id)
    return df, label_list


def balance_dataframe(
    df: pd.DataFrame,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    if strategy == "none":
        return df

    grouped = {label: group.copy() for label, group in df.groupby("label")}
    class_sizes = {label: len(group) for label, group in grouped.items()}

    if strategy == "undersample":
        target_size = min(class_sizes.values())
        balanced_parts = [
            group.sample(n=target_size, random_state=seed, replace=False)
            for group in grouped.values()
        ]
    else:
        target_size = max(class_sizes.values())
        balanced_parts = [
            group.sample(n=target_size, random_state=seed, replace=True)
            if len(group) < target_size
            else group.sample(n=target_size, random_state=seed, replace=False)
            for group in grouped.values()
        ]

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df["label"].value_counts().min() < 3:
        raise ValueError(
            "Setiap kelas minimal harus memiliki 3 sampel untuk split train/val/test."
        )

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_df["label"],
    )
    return train_df, val_df, test_df


def build_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[["content_clean", "labels"]].reset_index(drop=True),
                preserve_index=False,
            ),
            "validation": Dataset.from_pandas(
                val_df[["content_clean", "labels"]].reset_index(drop=True),
                preserve_index=False,
            ),
            "test": Dataset.from_pandas(
                test_df[["content_clean", "labels"]].reset_index(drop=True),
                preserve_index=False,
            ),
        }
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["content_clean"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    return tokenized_dataset.remove_columns(["content_clean"])


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def save_artifacts(
    output_dir: Path,
    train_result,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    class_report: dict[str, dict[str, float]],
    confusion: list[list[int]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
) -> None:
    summary = {
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "dataset_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "classification_report": class_report,
        "confusion_matrix": confusion,
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    test_predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data tidak ditemukan: {input_path}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Training device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    df, label_list = load_training_dataframe(
        input_path=input_path,
        label_column=args.label_column,
        binary=args.binary,
    )
    original_distribution = df["label"].value_counts().sort_index()
    df = balance_dataframe(df, strategy=args.balance_strategy, seed=args.seed)
    balanced_distribution = df["label"].value_counts().sort_index()
    train_df, val_df, test_df = split_dataframe(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    log(
        "Dataset sizes "
        f"train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}"
    )
    log(f"Original label distribution:\n{original_distribution}")
    log(f"Balanced label distribution:\n{balanced_distribution}")

    dataset = build_dataset_dict(train_df, val_df, test_df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    checkpoint_dir = INDOBERT_MODEL_DIR / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=5,
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    validation_metrics = trainer.evaluate(tokenized_dataset["validation"])

    test_output = trainer.predict(tokenized_dataset["test"])
    test_predictions = np.argmax(test_output.predictions, axis=-1)
    test_labels = np.array(test_output.label_ids)

    test_metrics = {
        "test_accuracy": accuracy_score(test_labels, test_predictions),
        "test_f1_weighted": f1_score(
            test_labels,
            test_predictions,
            average="weighted",
        ),
    }

    class_report = classification_report(
        test_labels,
        test_predictions,
        target_names=label_list,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        test_labels,
        test_predictions,
        labels=list(range(len(label_list))),
    ).tolist()

    test_prediction_columns = [
        column
        for column in ["review_id", "score", "content", "content_clean", "label"]
        if column in test_df.columns
    ]
    test_predictions_df = test_df[test_prediction_columns].reset_index(drop=True)
    test_predictions_df["predicted_label"] = [
        id2label[prediction] for prediction in test_predictions
    ]

    trainer.save_model(str(INDOBERT_MODEL_DIR))
    tokenizer.save_pretrained(str(INDOBERT_MODEL_DIR))
    save_artifacts(
        output_dir=INDOBERT_MODEL_DIR,
        train_result=train_result,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        class_report=class_report,
        confusion=confusion,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        test_predictions_df=test_predictions_df,
    )

    log(f"Validation metrics: {validation_metrics}")
    log(f"Test metrics: {test_metrics}")
    log(f"Model and artifacts saved to: {INDOBERT_MODEL_DIR}")


if __name__ == "__main__":
    main()
