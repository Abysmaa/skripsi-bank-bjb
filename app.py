from __future__ import annotations

import json
import sys
from math import exp
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import preprocess_text  # noqa: E402

DEPENDENCY_IMPORT_ERROR: Exception | None = None
try:
    import torch  # noqa: E402
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
except ImportError as error:  # pragma: no cover
    DEPENDENCY_IMPORT_ERROR = error
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


st.set_page_config(
    page_title="Analisis Sentimen Digi bank bjb",
    layout="wide",
)


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_RESULT_PATH = DATA_DIR / "final" / "bjb_sentiment_result.csv"
VIS_DIR = DATA_DIR / "final" / "visualizations"
TRAINING_SUMMARY_PATH = MODELS_DIR / "indobert" / "training_summary.json"
TEST_PREDICTIONS_PATH = MODELS_DIR / "indobert" / "test_predictions.csv"
LDA_SUMMARY_PATH = MODELS_DIR / "lda" / "lda_summary.json"
LDA_TOPIC_TERMS_PATH = MODELS_DIR / "lda" / "lda_topic_terms.csv"
LDA_DOCUMENT_TOPICS_PATH = MODELS_DIR / "lda" / "lda_document_topics.csv"
LDA_VISUALIZATION_PATH = MODELS_DIR / "lda" / "lda_visualization.html"
MODEL_DIR = MODELS_DIR / "indobert"
CONCLUSION_MD_PATH = PROJECT_ROOT / "BAB5_KESIMPULAN_DAN_SARAN.md"

TOPIC_LABELS = {
    0: "Aktivasi, Rekening, dan Koneksi",
    1: "Aktivasi Pasca-Update dan Jaringan",
    2: "Login, Password, dan Akses",
    3: "Transaksi, Saldo, dan Transfer",
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 28%),
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.08), transparent 22%),
                    linear-gradient(180deg, #f7fafc 0%, #eef5f3 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.2rem;
                max-width: 1280px;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(15, 23, 42, 0.08);
                padding: 1rem 1rem 0.8rem 1rem;
                border-radius: 18px;
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
            }
            .hero-panel {
                background: linear-gradient(135deg, #0f172a 0%, #16394f 48%, #0f766e 100%);
                color: #f8fafc;
                border-radius: 28px;
                padding: 1.6rem 1.8rem;
                margin-bottom: 1.1rem;
                box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
            }
            .hero-kicker {
                display: inline-block;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 999px;
                padding: 0.28rem 0.7rem;
                font-size: 0.76rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .hero-title {
                font-size: 2.25rem;
                line-height: 1.1;
                margin: 0.9rem 0 0.65rem 0;
                font-weight: 700;
            }
            .hero-subtitle {
                font-size: 1rem;
                line-height: 1.65;
                color: rgba(248, 250, 252, 0.88);
                max-width: 58rem;
                margin-bottom: 0.9rem;
            }
            .info-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 20px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
                height: 100%;
            }
            .info-card h4 {
                margin: 0 0 0.5rem 0;
                color: #0f172a;
                font-size: 1rem;
            }
            .info-card p {
                margin: 0;
                color: #334155;
                font-size: 0.95rem;
                line-height: 1.6;
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0.35rem 0 0.65rem 0;
            }
            .sidebar-note {
                background: rgba(255,255,255,0.76);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                margin-top: 0.75rem;
            }
            .sidebar-note h4 {
                margin: 0 0 0.45rem 0;
                color: #0f172a;
                font-size: 0.95rem;
            }
            .sidebar-note p, .sidebar-note li {
                color: #334155;
                font-size: 0.88rem;
                line-height: 1.5;
            }
            .topic-chip {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(15, 118, 110, 0.1);
                color: #115e59;
                border: 1px solid rgba(15, 118, 110, 0.16);
                font-size: 0.82rem;
                margin-bottom: 0.55rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


@st.cache_data
def load_final_results() -> pd.DataFrame:
    df = pd.read_csv(FINAL_RESULT_PATH)
    for column in ["content", "content_clean", "predicted_label", "true_label"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


@st.cache_data
def load_training_summary() -> dict[str, object]:
    return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_lda_summary() -> dict[str, object]:
    return json.loads(LDA_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_topic_terms() -> pd.DataFrame:
    return pd.read_csv(LDA_TOPIC_TERMS_PATH)


@st.cache_data
def load_document_topics() -> pd.DataFrame:
    return pd.read_csv(LDA_DOCUMENT_TOPICS_PATH)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH)


@st.cache_data
def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@st.cache_resource
def load_model():
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        details = f" Detail import: {DEPENDENCY_IMPORT_ERROR}" if DEPENDENCY_IMPORT_ERROR else ""
        raise ImportError(
            "Dependency transformers/torch belum tersedia pada lingkungan Python yang menjalankan Streamlit. "
            f"Python aktif: {sys.executable}. "
            "Pasang dependency project dengan `pip install -r requirements.txt`."
            f"{details}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_text(text: str) -> tuple[str, float, str]:
    cleaned_text = preprocess_text(text)
    if not cleaned_text.strip():
        return "", 0.0, cleaned_text

    tokenizer, model, device = load_model()
    encoded = tokenizer(
        [cleaned_text],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits.detach().cpu().tolist()[0]
    probabilities = softmax([float(value) for value in logits])
    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)

    id2label_config = getattr(model.config, "id2label", {}) or {}
    id2label = {int(key): str(value).lower() for key, value in id2label_config.items()}
    label = id2label.get(predicted_id, str(predicted_id))
    confidence = float(probabilities[predicted_id])
    return label, confidence, cleaned_text


def render_sidebar(df: pd.DataFrame, training_summary: dict[str, object], lda_summary: dict[str, object]) -> str:
    st.sidebar.title("Mode Sidang")
    st.sidebar.caption("Dashboard dapat dipakai dalam mode demo singkat atau eksplorasi lengkap.")
    view_mode = st.sidebar.radio(
        "Pilih Mode Dashboard",
        ["Demo 5 Menit", "Eksplorasi Lengkap"],
        index=0,
    )

    label_counts = df["predicted_label"].value_counts()
    st.sidebar.metric("Review Dianalisis", f"{len(df):,}")
    st.sidebar.metric("Sentimen Positif", f"{int(label_counts.get('positive', 0)):,}")
    st.sidebar.metric("Sentimen Negatif", f"{int(label_counts.get('negative', 0)):,}")
    st.sidebar.metric("Akurasi Test", f"{training_summary['test_metrics']['test_accuracy']:.4f}")

    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            <h4>Inti Temuan</h4>
            <ul>
                <li>Sentimen positif masih dominan.</li>
                <li>Keluhan utama terkonsentrasi pada login, aktivasi, transaksi, dan rekening.</li>
                <li>LDA memetakan 4 topik dominan dari review negatif.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <h4>Artefak Aktif</h4>
            <p>Model IndoBERT: <code>{MODEL_DIR.name}</code></p>
            <p>Dokumen LDA: <strong>{lda_summary['num_documents']}</strong></p>
            <p>Topik LDA: <strong>{lda_summary['num_topics']}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(df: pd.DataFrame, training_summary: dict[str, object]) -> None:
    label_counts = df["predicted_label"].value_counts()
    positive = int(label_counts.get("positive", 0))
    negative = int(label_counts.get("negative", 0))
    accuracy = training_summary["test_metrics"]["test_accuracy"]

    st.markdown(
        f"""
        <div class="hero-panel">
            <span class="hero-kicker">Dashboard Skripsi | IndoBERT + LDA</span>
            <div class="hero-title">Analisis Sentimen Ulasan Digi bank bjb</div>
            <div class="hero-subtitle">
                Dashboard ini merangkum hasil akhir penelitian: distribusi sentimen pengguna,
                evaluasi model IndoBERT, pemetaan topik keluhan dengan LDA, dan prediksi review
                baru untuk demonstrasi saat sidang.
            </div>
            <div class="hero-subtitle">
                <strong>{positive:,}</strong> review diprediksi positif,
                <strong>{negative:,}</strong> negatif,
                dengan akurasi uji model sebesar <strong>{accuracy:.4f}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card_columns(df: pd.DataFrame) -> None:
    total_reviews = len(df)
    label_counts = df["predicted_label"].value_counts()
    negative_count = int(label_counts.get("negative", 0))
    positive_count = int(label_counts.get("positive", 0))
    avg_confidence = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Review", f"{total_reviews:,}")
    col2.metric("Prediksi Positif", f"{positive_count:,}")
    col3.metric("Prediksi Negatif", f"{negative_count:,}")
    col4.metric("Rata-rata Confidence", f"{avg_confidence:.2%}")


def render_info_cards(df: pd.DataFrame, training_summary: dict[str, object], lda_summary: dict[str, object]) -> None:
    label_counts = df["predicted_label"].value_counts()
    total = len(df)
    positive_pct = (label_counts.get("positive", 0) / total * 100) if total else 0.0
    negative_pct = (label_counts.get("negative", 0) / total * 100) if total else 0.0

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""
        <div class="info-card">
            <h4>Interpretasi Sentimen</h4>
            <p>
                Sentimen positif mendominasi sebesar <strong>{positive_pct:.2f}%</strong>,
                tetapi porsi negatif masih signifikan di <strong>{negative_pct:.2f}%</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="info-card">
            <h4>Kualitas Model</h4>
            <p>
                IndoBERT mencapai akurasi test <strong>{training_summary['test_metrics']['test_accuracy']:.4f}</strong>
                dan F1 weighted <strong>{training_summary['test_metrics']['test_f1_weighted']:.4f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="info-card">
            <h4>Pemetaan Topik</h4>
            <p>
                LDA memproses <strong>{lda_summary['num_documents']}</strong> dokumen negatif
                menjadi <strong>{lda_summary['num_topics']}</strong> topik utama yang mudah diinterpretasi.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(df: pd.DataFrame, training_summary: dict[str, object], lda_summary: dict[str, object]) -> None:
    st.markdown('<div class="section-title">Ringkasan Eksekutif</div>', unsafe_allow_html=True)
    metric_card_columns(df)
    render_info_cards(df, training_summary, lda_summary)

    counts = (
        df["predicted_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("**Distribusi Sentimen**")
        st.bar_chart(counts.set_index("label")["count"])
        st.dataframe(
            counts.assign(percentage=counts["percentage"].map(lambda value: f"{value:.2f}%")),
            hide_index=True,
            width="stretch",
        )

    with col2:
        st.markdown("**Ringkasan Model**")
        test_metrics = training_summary["test_metrics"]
        validation_metrics = training_summary["validation_metrics"]
        st.write(f"- Akurasi uji: `{test_metrics['test_accuracy']:.4f}`")
        st.write(f"- F1 tertimbang uji: `{test_metrics['test_f1_weighted']:.4f}`")
        st.write(f"- Akurasi validasi: `{validation_metrics['eval_accuracy']:.4f}`")
        st.write(f"- Dokumen LDA: `{lda_summary['num_documents']}`")
        st.write(f"- Jumlah topik: `{lda_summary['num_topics']}`")
        st.write(f"- Ukuran kamus LDA: `{lda_summary['dictionary_size']}`")
        st.write("- Fokus isu negatif: `login`, `aktivasi`, `transaksi`, `rekening`")

    image_col1, image_col2 = st.columns(2)
    with image_col1:
        negative_wordcloud = VIS_DIR / "wordcloud_negative.png"
        if negative_wordcloud.exists():
            st.image(str(negative_wordcloud), caption="Wordcloud Review Negatif", width="stretch")
    with image_col2:
        positive_wordcloud = VIS_DIR / "wordcloud_positive.png"
        if positive_wordcloud.exists():
            st.image(str(positive_wordcloud), caption="Wordcloud Review Positif", width="stretch")


def render_dataset(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Eksplorasi Dataset Hasil</div>', unsafe_allow_html=True)

    labels = ["Semua"] + sorted(df["predicted_label"].dropna().unique().tolist())
    filter_col1, filter_col2, filter_col3 = st.columns([0.9, 1.1, 1.6])
    with filter_col1:
        selected_label = st.selectbox("Filter label", labels)
    with filter_col2:
        min_confidence = st.slider("Confidence minimum", 0.0, 1.0, 0.0, 0.01)
    with filter_col3:
        search_query = st.text_input("Cari kata dalam review")

    filtered_df = df.copy()
    if selected_label != "Semua":
        filtered_df = filtered_df[filtered_df["predicted_label"] == selected_label]
    if "confidence" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["confidence"] >= min_confidence]
    if search_query.strip():
        mask = filtered_df["content"].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    st.caption(f"Menampilkan {len(filtered_df):,} review")
    show_columns = [
        column
        for column in ["review_id", "score", "predicted_label", "confidence", "content", "content_clean"]
        if column in filtered_df.columns
    ]
    st.dataframe(filtered_df[show_columns], width="stretch", hide_index=True)


def render_model_metrics(training_summary: dict[str, object], test_predictions: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Evaluasi IndoBERT</div>', unsafe_allow_html=True)

    test_metrics = training_summary["test_metrics"]
    class_report = training_summary["classification_report"]
    confusion_matrix = training_summary["confusion_matrix"]

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Akurasi Test", f"{test_metrics['test_accuracy']:.4f}")
    metric2.metric("F1 Weighted Test", f"{test_metrics['test_f1_weighted']:.4f}")
    metric3.metric("Jumlah Data Test", str(training_summary["dataset_sizes"]["test"]))

    st.progress(float(test_metrics["test_accuracy"]))
    st.caption("Progress bar di atas merepresentasikan akurasi test untuk membantu pembacaan cepat saat presentasi.")

    report_rows = []
    for label in ["negative", "positive", "neutral"]:
        if label in class_report:
            report_rows.append(
                {
                    "label": label,
                    "precision": class_report[label]["precision"],
                    "recall": class_report[label]["recall"],
                    "f1_score": class_report[label]["f1-score"],
                    "support": class_report[label]["support"],
                }
            )
    if report_rows:
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report_rows), hide_index=True, width="stretch")

    st.markdown("**Confusion Matrix**")
    labels = ["negative", "positive"][: len(confusion_matrix)]
    confusion_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    st.dataframe(confusion_df, width="stretch")

    st.markdown("**Contoh Prediksi Test**")
    st.dataframe(test_predictions.head(20), width="stretch", hide_index=True)


def render_topics(topic_terms: pd.DataFrame, document_topics: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Topic Modeling LDA</div>', unsafe_allow_html=True)

    topic_counts = (
        document_topics["dominant_topic"]
        .value_counts()
        .sort_index()
        .rename_axis("topic_id")
        .reset_index(name="count")
    )
    topic_counts["topic_name"] = topic_counts["topic_id"].map(
        lambda topic_id: TOPIC_LABELS.get(topic_id, f"Topik {topic_id}")
    )

    st.markdown("**Sebaran Dokumen per Topik**")
    chart_df = topic_counts.set_index("topic_name")[["count"]]
    st.bar_chart(chart_df)

    topic_options = sorted(topic_terms["topic_id"].unique().tolist())
    topic_display = {
        topic_id: f"Topik {topic_id} | {TOPIC_LABELS.get(topic_id, 'Topik')}"
        for topic_id in topic_options
    }
    selected_topic = st.selectbox(
        "Pilih topik",
        topic_options,
        format_func=lambda topic_id: topic_display[topic_id],
    )

    left_col, right_col = st.columns([1, 1.2])
    with left_col:
        st.markdown(
            f'<div class="topic-chip">{topic_display[selected_topic]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Kata Dominan**")
        selected_terms = topic_terms[topic_terms["topic_id"] == selected_topic].sort_values(
            "weight", ascending=False
        )
        st.dataframe(selected_terms, hide_index=True, width="stretch")

        topic_image = VIS_DIR / f"lda_topic_{selected_topic}.png"
        if topic_image.exists():
            st.image(str(topic_image), caption=f"Top Terms Topic {selected_topic}", width="stretch")

    with right_col:
        st.markdown("**Contoh Review dalam Topik Ini**")
        selected_docs = document_topics[document_topics["dominant_topic"] == selected_topic][
            ["review_id", "dominant_score", "content", "content_clean"]
        ].head(15)
        st.dataframe(selected_docs, hide_index=True, width="stretch")

    if LDA_VISUALIZATION_PATH.exists():
        with st.expander("Buka HTML pyLDAvis"):
            html_content = LDA_VISUALIZATION_PATH.read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=700, scrolling=True)


def render_recommendations_page(document_topics: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Saran untuk Bank BJB</div>', unsafe_allow_html=True)
    st.write(
        "Halaman ini merangkum objek keluhan yang paling sering muncul pada komentar negatif, "
        "lalu menerjemahkannya menjadi saran tindakan yang bisa diprioritaskan oleh Bank BJB."
    )

    recommendation_map = {
        0: {
            "objek": "Aktivasi, pembukaan rekening, dan koneksi layanan",
            "masalah": "Komentar negatif pada topik ini banyak berkaitan dengan aktivasi yang gagal, proses buka rekening yang tersendat, koneksi yang terputus, serta layanan yang terasa tidak stabil saat proses awal penggunaan.",
            "saran": [
                "Sederhanakan alur aktivasi dan pembukaan rekening agar pengguna tidak terjebak pada proses berulang.",
                "Perkuat stabilitas koneksi dan fallback ketika layanan terputus di tengah proses registrasi atau aktivasi.",
                "Berikan pesan error yang lebih spesifik pada kasus koneksi, pulsa OTP, atau rekening yang gagal diproses.",
            ],
        },
        1: {
            "objek": "Aktivasi pasca-update, jaringan, dan nomor terdaftar",
            "masalah": "Topik ini menunjukkan keluhan setelah update aplikasi, khususnya aktivasi ulang, jaringan yang menghambat akses, serta masalah pada nomor atau identitas akun yang membuat pengguna sulit masuk kembali.",
            "saran": [
                "Uji skenario migrasi pengguna lama sebelum update dirilis agar tidak memicu aktivasi ulang yang bermasalah.",
                "Pastikan proses verifikasi nomor dan identitas akun tetap stabil pada berbagai kondisi jaringan.",
                "Tambahkan panduan pemulihan akun yang jelas untuk pengguna yang gagal masuk setelah pembaruan aplikasi.",
            ],
        },
        2: {
            "objek": "Login, password, dan hambatan akses akun",
            "masalah": "Komentar pada topik ini menonjolkan kesalahan password, login yang sulit, proses akses yang ribet, dan hambatan saat pengguna mencoba masuk kembali ke akun mereka.",
            "saran": [
                "Perbaiki validasi password, PIN, dan kredensial agar pengguna tidak sering mengalami false rejection.",
                "Sederhanakan alur login ulang dan reset akses agar tidak terasa ribet bagi pengguna lama.",
                "Sediakan bantuan cepat untuk kasus akun tidak bisa diakses meskipun data login pengguna dianggap benar.",
            ],
        },
        3: {
            "objek": "Transaksi, saldo, dan transfer",
            "masalah": "Topik ini memperlihatkan kekhawatiran pengguna terhadap transaksi yang lambat atau gagal, saldo yang tidak segera berubah, transfer bermasalah, dan indikator layanan yang tidak meyakinkan.",
            "saran": [
                "Prioritaskan reliabilitas transaksi inti seperti transfer dan pembaruan saldo secara real-time.",
                "Percepat penanganan transaksi gagal agar dana dan status rekening segera sinkron kembali.",
                "Tampilkan indikator status layanan yang lebih jelas agar pengguna tahu apakah kendala berasal dari sistem, jaringan, atau proses transfer.",
            ],
        },
    }

    topic_counts = document_topics["dominant_topic"].value_counts().sort_index()
    summary_rows = []
    for topic_id, total_comments in topic_counts.items():
        summary_rows.append(
            {
                "topic_id": topic_id,
                "objek_keluhan": recommendation_map.get(topic_id, {}).get("objek", TOPIC_LABELS.get(topic_id, f"Topik {topic_id}")),
                "jumlah_komentar_negatif": int(total_comments),
            }
        )
    st.markdown("**Ringkasan objek keluhan negatif**")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    for topic_id in sorted(topic_counts.index.tolist()):
        config = recommendation_map.get(topic_id, {})
        object_name = config.get("objek", TOPIC_LABELS.get(topic_id, f"Topik {topic_id}"))
        issue_text = config.get("masalah", "Komentar negatif pada topik ini perlu ditindaklanjuti lebih lanjut.")
        suggestions = config.get("saran", [])
        examples = (
            document_topics[document_topics["dominant_topic"] == topic_id]["content"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        st.markdown(
            f'<div class="topic-chip">Topik {topic_id} | {object_name}</div>',
            unsafe_allow_html=True,
        )
        left_col, right_col = st.columns([1.1, 1.2])
        with left_col:
            st.markdown("**Hasil dari komentar negatif**")
            st.write(issue_text)
            if examples:
                st.markdown("**Contoh komentar**")
                for example in examples:
                    st.markdown(f"- {example}")
        with right_col:
            st.markdown("**Saran untuk Bank BJB**")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")


def render_conclusion_page(df: pd.DataFrame, training_summary: dict[str, object], topic_terms: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Kesimpulan Sidang</div>', unsafe_allow_html=True)
    label_counts = df["predicted_label"].value_counts()
    total = len(df)
    positive = int(label_counts.get("positive", 0))
    negative = int(label_counts.get("negative", 0))
    accuracy = training_summary["test_metrics"]["test_accuracy"]
    positive_pct = (positive / total * 100) if total else 0.0
    negative_pct = (negative / total * 100) if total else 0.0

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Pesan Inti Penelitian</h4>
                <p>
                    Dari <strong>{total:,}</strong> review, sentimen positif masih dominan sebesar
                    <strong>{positive_pct:.2f}%</strong>, sedangkan sentimen negatif sebesar
                    <strong>{negative_pct:.2f}%</strong>. Model IndoBERT mencapai akurasi
                    <strong>{accuracy:.4f}</strong>, sehingga cukup andal untuk memetakan persepsi pengguna.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>Temuan Utama untuk Disampaikan</h4>
                <p>
                    Keluhan pengguna paling sering terkonsentrasi pada login, aktivasi ulang,
                    kestabilan transaksi, pembukaan rekening, dan kekhawatiran terhadap dana.
                    LDA membantu mengubah keluhan-keluhan itu menjadi kelompok topik yang mudah dijelaskan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Poin Kesimpulan Cepat**")
    st.markdown(
        f"""
        1. Sentimen positif masih dominan dengan {positive:,} review, tetapi keluhan negatif tetap signifikan pada {negative:,} review.
        2. IndoBERT memberi performa tinggi dengan akurasi test {accuracy:.4f} dan mendukung analisis sentimen secara otomatis.
        3. LDA memetakan keluhan negatif menjadi empat isu besar: aktivasi dan koneksi layanan, dampak update pada akses akun, login dan password, serta transaksi dan saldo.
        4. Titik lemah aplikasi bukan hanya di fitur tambahan, tetapi di proses inti yang menentukan kepercayaan pengguna.
        """,
    )

    st.markdown("**Topik Dominan**")
    topic_preview = (
        topic_terms.sort_values(["topic_id", "weight"], ascending=[True, False])
        .groupby("topic_id")
        .head(5)
        .groupby("topic_id")["term"]
        .apply(lambda values: ", ".join(values))
        .reset_index()
    )
    topic_preview["nama_topik"] = topic_preview["topic_id"].map(
        lambda topic_id: TOPIC_LABELS.get(topic_id, f"Topik {topic_id}")
    )
    st.dataframe(
        topic_preview[["topic_id", "nama_topik", "term"]],
        hide_index=True,
        width="stretch",
    )

    if CONCLUSION_MD_PATH.exists():
        with st.expander("Lihat draft Bab 5 di dashboard"):
            st.markdown(load_text_file(CONCLUSION_MD_PATH))


def render_live_prediction(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Prediksi Review Baru</div>', unsafe_allow_html=True)
    default_text = "Aplikasi bagus dan membantu transaksi saya setiap hari."
    user_text = st.text_area("Masukkan review", value=default_text, height=160)

    if st.button("Prediksi Sentimen", type="primary"):
        if not user_text.strip():
            st.warning("Masukkan review terlebih dahulu.")
        else:
            try:
                label, confidence, cleaned_text = predict_text(user_text)
            except Exception as error:  # pragma: no cover
                st.error(f"Gagal memuat model atau memprediksi teks: {error}")
            else:
                st.success(f"Prediksi: {label} ({confidence:.2%})")
                st.write("**Hasil preprocessing**")
                st.code(cleaned_text or "-", language="text")

                examples = df[df["predicted_label"] == label][["content", "confidence"]].copy()
                if not examples.empty:
                    st.write("**Contoh review serupa dari dataset**")
                    st.dataframe(
                        examples.sort_values("confidence", ascending=False).head(5),
                        hide_index=True,
                        width="stretch",
                    )


def main() -> None:
    inject_custom_css()

    required_paths = [
        FINAL_RESULT_PATH,
        TRAINING_SUMMARY_PATH,
        LDA_SUMMARY_PATH,
        LDA_TOPIC_TERMS_PATH,
        LDA_DOCUMENT_TOPICS_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        st.error("Beberapa artefak belum tersedia:")
        for path in missing_paths:
            st.write(f"- `{path}`")
        st.stop()

    df = load_final_results()
    training_summary = load_training_summary()
    lda_summary = load_lda_summary()
    topic_terms = load_topic_terms()
    document_topics = load_document_topics()
    test_predictions = load_test_predictions()

    view_mode = render_sidebar(df, training_summary, lda_summary)
    render_hero(df, training_summary)

    if view_mode == "Demo 5 Menit":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Topik LDA", "Saran", "Kesimpulan Sidang", "Demo Prediksi"]
        )

        with tab1:
            render_overview(df, training_summary, lda_summary)
        with tab2:
            render_topics(topic_terms, document_topics)
        with tab3:
            render_recommendations_page(document_topics)
        with tab4:
            render_conclusion_page(df, training_summary, topic_terms)
        with tab5:
            render_live_prediction(df)
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Overview", "Eksplorasi Data", "Evaluasi Model", "Topik LDA", "Saran", "Kesimpulan Sidang", "Demo Prediksi"]
        )

        with tab1:
            render_overview(df, training_summary, lda_summary)
        with tab2:
            render_dataset(df)
        with tab3:
            render_model_metrics(training_summary, test_predictions)
        with tab4:
            render_topics(topic_terms, document_topics)
        with tab5:
            render_recommendations_page(document_topics)
        with tab6:
            render_conclusion_page(df, training_summary, topic_terms)
        with tab7:
            render_live_prediction(df)


if __name__ == "__main__":
    main()
