from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora
from gensim.models import LdaModel

try:
    from .utils import FINAL_DATA_PATH, LDA_MODEL_DIR, PROCESSED_DATA_PATH, configure_logging, log
except ImportError:
    from utils import FINAL_DATA_PATH, LDA_MODEL_DIR, PROCESSED_DATA_PATH, configure_logging, log


SECURITY_KEYWORDS = {
    "akun",
    "aktivasi",
    "blokir",
    "fingerprint",
    "keaman",
    "kode",
    "login",
    "otp",
    "password",
    "pin",
    "qr",
    "rekening",
    "token",
    "transfer",
    "transaksi",
    "verifikasi",
    "wajah",
}

CUSTOM_STOPWORDS = {
    "ada",
    "aja",
    "akan",
    "akhir",
    "aman",
    "atau",
    "banget",
    "bank",
    "belum",
    "bikin",
    "bisa",
    "bjb",
    "buat",
    "coba",
    "dah",
    "dan",
    "dari",
    "dengan",
    "digi",
    "di",
    "dong",
    "emang",
    "gak",
    "hal",
    "harus",
    "ini",
    "itu",
    "jadi",
    "justru",
    "juga",
    "kalau",
    "kalo",
    "kan",
    "karena",
    "ke",
    "kenapa",
    "ketika",
    "lagi",
    "lain",
    "lebih",
    "malah",
    "masih",
    "memang",
    "minta",
    "mohon",
    "mau",
    "nih",
    "nya",
    "padahal",
    "pada",
    "pake",
    "pakai",
    "please",
    "saat",
    "sama",
    "sangat",
    "saya",
    "sebelum",
    "secara",
    "sekali",
    "selalu",
    "setelah",
    "seperti",
    "sih",
    "sudah",
    "susahnya",
    "tapi",
    "tidak",
    "tolong",
    "terus",
    "udah",
    "untuk",
    "waktu",
    "yang",
    "ya",
    "yg",
    "aplikasi",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Latih LDA untuk topik review keamanan Bank BJB."
    )
    parser.add_argument("--topics", type=int, default=4, help="Jumlah topik LDA.")
    parser.add_argument("--passes", type=int, default=20, help="Jumlah passes LDA.")
    parser.add_argument(
        "--words-per-topic",
        type=int,
        default=10,
        help="Jumlah kata per topik yang disimpan.",
    )
    return parser


def load_source_dataframe() -> pd.DataFrame:
    if FINAL_DATA_PATH.exists():
        df = pd.read_csv(FINAL_DATA_PATH)
    elif PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        raise FileNotFoundError("Dataset untuk LDA tidak ditemukan.")

    if "content_clean" not in df.columns and "clean_content" in df.columns:
        df["content_clean"] = df["clean_content"]

    df["content_clean"] = df["content_clean"].fillna("").astype(str)
    if "predicted_label" not in df.columns:
        df["predicted_label"] = df.get("label", "").fillna("").astype(str)
    return df


def select_security_documents(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["content_clean"].str.strip() != ""].copy()
    df["contains_security_keyword"] = df["content_clean"].apply(
        lambda text: bool(set(text.split()) & SECURITY_KEYWORDS)
    )

    filtered_df = df[
        (df["predicted_label"].astype(str).str.lower() == "negative")
        & (df["contains_security_keyword"])
    ].copy()

    if len(filtered_df) < 30:
        filtered_df = df[df["predicted_label"].astype(str).str.lower() == "negative"].copy()

    if len(filtered_df) < 30:
        filtered_df = df.copy()

    return filtered_df.reset_index(drop=True)


def prepare_tokens(df: pd.DataFrame) -> tuple[list[list[str]], pd.DataFrame]:
    token_lists = []
    kept_rows = []

    for _, row in df.iterrows():
        tokens = [
            token
            for token in row["content_clean"].split()
            if len(token) >= 4 and token not in CUSTOM_STOPWORDS
        ]
        if not tokens:
            continue
        token_lists.append(tokens)
        kept_rows.append(row)

    prepared_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    return token_lists, prepared_df


def train_lda(
    tokens: list[list[str]],
    num_topics: int,
    passes: int,
) -> tuple[LdaModel, corpora.Dictionary, list[list[tuple[int, int]]]]:
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=3, no_above=0.85)
    corpus = [dictionary.doc2bow(token_list) for token_list in tokens]
    corpus = [document for document in corpus if document]

    if not corpus:
        raise ValueError("Corpus kosong setelah filtering. Tidak bisa menjalankan LDA.")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha="auto",
        eta="auto",
    )
    return lda_model, dictionary, corpus


def load_previous_topic_terms() -> pd.DataFrame:
    topic_terms_path = LDA_MODEL_DIR / "lda_topic_terms.csv"
    if not topic_terms_path.exists():
        return pd.DataFrame(columns=["topic_id", "term", "weight"])
    return pd.read_csv(topic_terms_path)


def save_lda_outputs(
    lda_model: LdaModel,
    dictionary: corpora.Dictionary,
    corpus: list[list[tuple[int, int]]],
    source_df: pd.DataFrame,
    words_per_topic: int,
) -> None:
    LDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lda_model.save(str(LDA_MODEL_DIR / "lda_model.gensim"))
    dictionary.save(str(LDA_MODEL_DIR / "lda_dictionary.gensim"))

    topic_rows: list[dict[str, object]] = []
    for topic_id in range(lda_model.num_topics):
        for term, weight in lda_model.show_topic(topic_id, topn=words_per_topic):
            topic_rows.append(
                {
                    "topic_id": topic_id,
                    "term": term,
                    "weight": round(weight, 6),
                }
            )

    pd.DataFrame(topic_rows).to_csv(
        LDA_MODEL_DIR / "lda_topic_terms.csv",
        index=False,
    )

    doc_topic_rows: list[dict[str, object]] = []
    for index, bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dominant_topic, dominant_score = max(topic_distribution, key=lambda item: item[1])
        source_row = source_df.iloc[index]
        doc_topic_rows.append(
            {
                "review_id": source_row.get("review_id", ""),
                "predicted_label": source_row.get("predicted_label", ""),
                "content": source_row.get("content", ""),
                "content_clean": source_row.get("content_clean", ""),
                "dominant_topic": dominant_topic,
                "dominant_score": round(float(dominant_score), 6),
            }
        )

    pd.DataFrame(doc_topic_rows).to_csv(
        LDA_MODEL_DIR / "lda_document_topics.csv",
        index=False,
    )

    # Hindari multiprocessing joblib yang sering gagal di sandbox Windows.
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary, n_jobs=1)
    pyldavis_path = LDA_MODEL_DIR / "lda_visualization.html"
    pyLDAvis.save_html(lda_vis, str(pyldavis_path))

    summary = {
        "num_topics": lda_model.num_topics,
        "num_documents": len(source_df),
        "dictionary_size": len(dictionary),
        "output_files": [
            "lda_model.gensim",
            "lda_dictionary.gensim",
            "lda_topic_terms.csv",
            "lda_document_topics.csv",
            "lda_visualization.html",
        ],
    }
    (LDA_MODEL_DIR / "lda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def format_topic_terms(topic_terms_df: pd.DataFrame) -> dict[int, str]:
    if topic_terms_df.empty:
        return {}

    formatted: dict[int, str] = {}
    for topic_id, topic_df in topic_terms_df.groupby("topic_id"):
        sorted_df = topic_df.sort_values("weight", ascending=False)
        formatted[int(topic_id)] = ", ".join(sorted_df["term"].head(10).tolist())
    return formatted


def log_topic_comparison(previous_terms: pd.DataFrame, latest_terms: pd.DataFrame) -> None:
    previous_map = format_topic_terms(previous_terms)
    latest_map = format_topic_terms(latest_terms)

    log("Perbandingan top terms sebelum vs sesudah stopword baru:")
    for topic_id in range(max(len(previous_map), len(latest_map), 4)):
        old_terms = previous_map.get(topic_id, "-")
        new_terms = latest_map.get(topic_id, "-")
        log(f"Topik {topic_id} | sebelum: {old_terms}")
        log(f"Topik {topic_id} | sesudah: {new_terms}")


def log_topic_distribution() -> None:
    document_topics_path = LDA_MODEL_DIR / "lda_document_topics.csv"
    if not document_topics_path.exists():
        return

    docs_df = pd.read_csv(document_topics_path)
    distribution = docs_df["dominant_topic"].value_counts().sort_index()
    log(f"Sebaran dokumen per topik terbaru:\n{distribution.to_string()}")


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    previous_topic_terms = load_previous_topic_terms()
    source_df = load_source_dataframe()
    security_df = select_security_documents(source_df)
    tokens, prepared_df = prepare_tokens(security_df)
    lda_model, dictionary, corpus = train_lda(
        tokens=tokens,
        num_topics=args.topics,
        passes=args.passes,
    )
    save_lda_outputs(
        lda_model=lda_model,
        dictionary=dictionary,
        corpus=corpus,
        source_df=prepared_df,
        words_per_topic=args.words_per_topic,
    )
    latest_topic_terms = pd.read_csv(LDA_MODEL_DIR / "lda_topic_terms.csv")

    log(f"LDA documents used: {len(prepared_df)}")
    log(f"LDA artifacts saved to: {LDA_MODEL_DIR}")
    log_topic_comparison(previous_topic_terms, latest_topic_terms)
    log_topic_distribution()


if __name__ == "__main__":
    main()
