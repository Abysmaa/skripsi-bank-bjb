from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

try:
    from .utils import FINAL_DATA_PATH, LDA_MODEL_DIR, log
except ImportError:
    from utils import FINAL_DATA_PATH, LDA_MODEL_DIR, log


VIS_DIR = FINAL_DATA_PATH.parent / "visualizations"


def ensure_visualization_dir() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_final_dataframe() -> pd.DataFrame:
    if not FINAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Hasil final tidak ditemukan: {FINAL_DATA_PATH}")
    df = pd.read_csv(FINAL_DATA_PATH)
    df["content_clean"] = df["content_clean"].fillna("").astype(str)
    return df


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["predicted_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", color=["#c0392b", "#27ae60"], ax=ax)
    ax.set_title("Distribusi Sentimen Prediksi")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah Review")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "sentiment_distribution.png", dpi=200)
    plt.close(fig)
    counts.rename_axis("predicted_label").reset_index(name="count").to_csv(
        VIS_DIR / "sentiment_distribution.csv",
        index=False,
    )


def make_wordcloud(text: str, output_path: Path, title: str) -> None:
    if not text.strip():
        return

    cloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_wordclouds(df: pd.DataFrame) -> None:
    negative_text = " ".join(
        df.loc[df["predicted_label"] == "negative", "content_clean"].tolist()
    )
    positive_text = " ".join(
        df.loc[df["predicted_label"] == "positive", "content_clean"].tolist()
    )
    make_wordcloud(
        negative_text,
        VIS_DIR / "wordcloud_negative.png",
        "Word Cloud Review Negatif",
    )
    make_wordcloud(
        positive_text,
        VIS_DIR / "wordcloud_positive.png",
        "Word Cloud Review Positif",
    )


def plot_label_comparison(df: pd.DataFrame) -> None:
    if "true_label" not in df.columns:
        return

    comparison = (
        df.groupby(["true_label", "predicted_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    comparison.to_csv(VIS_DIR / "label_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Perbandingan True Label vs Predicted Label")
    ax.set_xlabel("True Label")
    ax.set_ylabel("Jumlah Review")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "label_comparison.png", dpi=200)
    plt.close(fig)


def plot_lda_topics() -> None:
    topic_terms_path = LDA_MODEL_DIR / "lda_topic_terms.csv"
    if not topic_terms_path.exists():
        return

    topic_terms_df = pd.read_csv(topic_terms_path)
    for topic_id, topic_df in topic_terms_df.groupby("topic_id"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sorted_df = topic_df.sort_values("weight", ascending=True)
        ax.barh(sorted_df["term"], sorted_df["weight"], color="#2980b9")
        ax.set_title(f"Top Terms Topic {topic_id}")
        ax.set_xlabel("Weight")
        fig.tight_layout()
        fig.savefig(VIS_DIR / f"lda_topic_{topic_id}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    ensure_visualization_dir()
    df = load_final_dataframe()
    plot_sentiment_distribution(df)
    plot_wordclouds(df)
    plot_label_comparison(df)
    plot_lda_topics()
    log(f"Visualization artifacts saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
