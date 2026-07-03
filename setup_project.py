from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_TITLE = (
    "Analisis Sentimen dan Topic Modeling Aspek Keamanan Aplikasi "
    "Mobile Banking Bank BJB menggunakan IndoBERT dan LDA"
)

README_CONTENT = f"""# {PROJECT_TITLE}

Project ini digunakan untuk penelitian skripsi data science yang berfokus pada
analisis sentimen ulasan aplikasi mobile banking Bank BJB menggunakan IndoBERT
serta topic modeling dengan metode LDA untuk mengidentifikasi aspek keamanan
yang sering dibahas oleh pengguna.

## Tujuan Project

- Mengumpulkan ulasan aplikasi dari Google Play Store
- Membersihkan dan menyiapkan data teks untuk analisis
- Membuat semi-label dan menyiapkan anotasi manual
- Melatih model analisis sentimen berbasis IndoBERT
- Mengekstraksi topik utama terkait aspek keamanan dengan LDA
- Menyajikan hasil analisis dalam bentuk visualisasi dan laporan

## Struktur Folder

```text
skripsi-bank-bjb/
|-- data/
|   |-- raw/
|   |-- processed/
|   |   `-- annotation_batches/
|   `-- final/
|       `-- visualizations/
|-- notebooks/
|-- models/
|   |-- indobert/
|   `-- lda/
|-- src/
|-- requirements.txt
|-- LABELING_GUIDE.md
|-- README.md
`-- setup_project.py
```

## Alur Kerja

1. Jalankan `setup_project.py` untuk memastikan struktur project lengkap.
2. Lakukan scraping ulasan dan simpan hasilnya ke `data/raw/`.
3. Bersihkan teks dan siapkan dataset di `data/processed/`.
4. Buat semi-label dan isi `manual_label` untuk review yang perlu anotasi.
5. Jalankan finalisasi label untuk menghasilkan dataset training final.
6. Latih model IndoBERT dan lakukan inferensi sentimen.
7. Lakukan topic modeling LDA pada teks yang sudah diproses.
8. Simpan hasil akhir analisis ke `data/final/`.

## Modul `src`

- `scraper.py`: pipeline scraping dan penyimpanan data mentah
- `preprocessing.py`: fungsi pembersihan teks dan pembuatan data bersih
- `labeling.py`: membuat semi-label dan template review manual
- `finalize_labels.py`: menggabungkan manual label dan semi-label menjadi dataset final
- `train_indobert.py`: training IndoBERT dengan split train/validation/test dan ekspor metrik
- `sentiment.py`: inferensi sentimen dengan model IndoBERT yang sudah dilatih
- `lda_model.py`: topic modeling LDA untuk review keamanan
- `visualization.py`: membuat grafik distribusi sentimen, word cloud, dan top terms LDA
- `utils.py`: utilitas path, logging, dan helper CSV

## Menjalankan Setup

```bash
python setup_project.py
```

Jika environment Windows belum mengenali `python`, gunakan interpreter Python yang
tersedia di VS Code atau tambahkan Python ke `PATH` terlebih dahulu.

## Virtual Environment

Project ini sudah disiapkan untuk memakai virtual environment lokal `.venv`.

Aktivasi di PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

Interpreter VS Code juga diarahkan ke `.venv\\Scripts\\python.exe`.

## Contoh Pipeline Terminal

```powershell
python src\\scraper.py --app-id com.bjbsmb --count 1000 --lang id --country id --sort newest
python src\\preprocessing.py --infer-label
python src\\labeling.py
python src\\finalize_labels.py --batch-size 100
python src\\train_indobert.py --input data\\processed\\bjb_reviews_labeled_final.csv --label-column final_label --binary --balance-strategy undersample --epochs 3 --train-batch-size 8 --eval-batch-size 8
python src\\sentiment.py --batch-size 16
python src\\lda_model.py --topics 4 --passes 20 --words-per-topic 10
python src\\visualization.py
```

Artifact training utama akan disimpan ke `models/indobert/`, termasuk:

- `config.json`
- `model.safetensors`
- `training_summary.json`
- `test_predictions.csv`

Artifact LDA akan disimpan ke `models/lda/`, termasuk:

- `lda_topic_terms.csv`
- `lda_document_topics.csv`
- `lda_visualization.html`

Artifact visualisasi akan disimpan ke `data/final/visualizations/`.
"""

REQUIREMENTS_CONTENT = """pandas
numpy
scikit-learn
google-play-scraper
transformers
torch
datasets
accelerate
nltk
sastrawi
gensim
pyLDAvis
matplotlib
wordcloud
"""

LABELING_GUIDE_CONTENT = """# Panduan Labeling Sentimen

File ini digunakan sebagai acuan saat mengisi kolom `manual_label` pada
`data/processed/bjb_reviews_manual_review.csv`.

## Label yang digunakan

- `negative`
- `neutral`
- `positive`

## Aturan Ringkas

### `negative`

Gunakan label `negative` jika review berisi keluhan, kegagalan fitur, hambatan
akses, masalah transaksi, atau kritik terhadap aplikasi.

Contoh indikasi:

- gagal login
- OTP tidak masuk
- aplikasi error, lemot, atau force close
- aktivasi gagal
- transaksi gagal
- verifikasi bermasalah
- keluhan keamanan akun, PIN, password, atau blokir

### `positive`

Gunakan label `positive` jika review menyatakan kepuasan, kemudahan penggunaan,
aplikasi membantu, transaksi lancar, atau pujian terhadap fitur.

Contoh indikasi:

- aplikasi sangat membantu
- mudah digunakan
- transaksi lancar
- bagus, aman, cepat

### `neutral`

Gunakan label `neutral` jika review bersifat informatif, ambigu, pertanyaan
tanpa emosi jelas, atau mencampur positif dan negatif tanpa dominasi kuat.

Contoh indikasi:

- hanya bertanya arti kode error
- review terlalu singkat dan tidak cukup jelas
- ada keluhan kecil tapi nada keseluruhan tidak jelas

## Aturan Prioritas

Jika satu review memuat unsur positif dan negatif sekaligus:

- pilih `negative` bila inti review adalah keluhan atau hambatan
- pilih `positive` bila inti review adalah kepuasan
- pilih `neutral` bila keduanya seimbang dan tidak jelas dominannya

## Cara Kerja Disarankan

1. Buka `data/processed/bjb_reviews_manual_review.csv`
2. Isi kolom `manual_label`
3. Simpan file
4. Jalankan:

```powershell
python src\\finalize_labels.py
python src\\train_indobert.py --input data\\processed\\bjb_reviews_labeled_final.csv --label-column final_label --binary --balance-strategy undersample --epochs 3 --train-batch-size 8 --eval-batch-size 8
```
"""

NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}


def make_markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def make_code_cell(code: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def build_notebook(cells: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "cells": cells,
            "metadata": NOTEBOOK_METADATA,
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=2,
    ) + "\n"


SCRAPING_NOTEBOOK_CONTENT = build_notebook(
    [
        make_markdown_cell(
            """
            # 01 Scraping Ulasan Bank BJB

            Notebook ini digunakan untuk mengambil ulasan aplikasi mobile banking
            Bank BJB dari Google Play Store dan menyimpannya ke folder `data/raw/`.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import sys

            CANDIDATES = [Path.cwd().resolve(), Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                (path for path in CANDIDATES if (path / "src").exists()),
                Path.cwd().resolve(),
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import pandas as pd

            from src.scraper import fetch_reviews, save_raw_reviews
            from src.utils import RAW_DATA_PATH

            RAW_DATA_PATH
            """
        ),
        make_markdown_cell(
            """
            ## Langkah Kerja

            1. Tentukan `app_id` aplikasi Bank BJB.
            2. Ambil ulasan menggunakan `google-play-scraper`.
            3. Simpan hasil scraping ke file CSV mentah.
            """
        ),
        make_code_cell(
            """
            APP_ID = "com.bjbsmb"
            COUNT = 1000
            LANG = "id"
            COUNTRY = "id"
            SORT = "newest"

            reviews = fetch_reviews(
                app_id=APP_ID,
                count=COUNT,
                lang=LANG,
                country=COUNTRY,
                sort=SORT,
            )
            df_raw = pd.DataFrame([review.__dict__ for review in reviews])
            df_raw.head()
            """
        ),
        make_code_cell(
            """
            save_raw_reviews(reviews)
            RAW_DATA_PATH
            """
        ),
    ]
)

PREPROCESSING_NOTEBOOK_CONTENT = build_notebook(
    [
        make_markdown_cell(
            """
            # 02 Preprocessing Data

            Notebook ini digunakan untuk membersihkan teks ulasan, melakukan
            normalisasi, dan menyiapkan dataset untuk analisis sentimen dan LDA.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import sys

            CANDIDATES = [Path.cwd().resolve(), Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                (path for path in CANDIDATES if (path / "src").exists()),
                Path.cwd().resolve(),
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import pandas as pd

            from src.preprocessing import preprocess_reviews
            from src.utils import PROCESSED_DATA_PATH, RAW_DATA_PATH

            RAW_DATA_PATH, PROCESSED_DATA_PATH
            """
        ),
        make_markdown_cell(
            """
            ## Langkah Preprocessing

            - Hapus duplikasi dan nilai kosong
            - Bersihkan URL, simbol, dan karakter non-teks
            - Lakukan case folding, tokenisasi, stopword removal, dan stemming
            - Siapkan kolom label jika anotasi dilakukan manual
            """
        ),
        make_code_cell(
            """
            processed_rows = preprocess_reviews(
                infer_label=True,
                drop_neutral=False,
                use_stemming=True,
            )
            df_processed = pd.DataFrame(processed_rows)
            df_processed.head()
            """
        ),
        make_code_cell(
            """
            df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
            PROCESSED_DATA_PATH
            """
        ),
    ]
)

INDOBERT_NOTEBOOK_CONTENT = build_notebook(
    [
        make_markdown_cell(
            """
            # 03 Training IndoBERT

            Notebook ini disiapkan untuk pelatihan model klasifikasi sentimen
            menggunakan IndoBERT pada data ulasan yang sudah dibersihkan.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import json
            import subprocess
            import sys

            CANDIDATES = [Path.cwd().resolve(), Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                (path for path in CANDIDATES if (path / "src").exists()),
                Path.cwd().resolve(),
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bjb_reviews_labeled_final.csv"
            MODEL_DIR = PROJECT_ROOT / "models" / "indobert"

            DATA_PATH, MODEL_DIR
            """
        ),
        make_markdown_cell(
            """
            ## Tahapan Training

            1. Muat dataset yang sudah memiliki label sentimen
            2. Bagi data menjadi train, validation, dan test
            3. Tokenisasi menggunakan tokenizer IndoBERT
            4. Lakukan fine-tuning dan evaluasi model
            """
        ),
        make_code_cell(
            """
            train_command = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_indobert.py"),
                "--input",
                str(DATA_PATH),
                "--label-column",
                "final_label",
                "--binary",
                "--balance-strategy",
                "undersample",
                "--epochs",
                "3",
                "--train-batch-size",
                "8",
                "--eval-batch-size",
                "8",
            ]
            train_command
            """
        ),
        make_code_cell(
            """
            subprocess.run(train_command, check=True)
            """
        ),
        make_code_cell(
            """
            summary_path = MODEL_DIR / "training_summary.json"
            json.loads(summary_path.read_text(encoding="utf-8"))
            """
        ),
        make_code_cell(
            """
            predict_command = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "sentiment.py"),
                "--batch-size",
                "16",
            ]
            subprocess.run(predict_command, check=True)
            """
        ),
    ]
)

LDA_NOTEBOOK_CONTENT = build_notebook(
    [
        make_markdown_cell(
            """
            # 04 LDA Topic Modeling

            Notebook ini digunakan untuk mengeksplorasi topik utama dari ulasan
            yang relevan dengan aspek keamanan aplikasi mobile banking Bank BJB.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import subprocess
            import sys

            CANDIDATES = [Path.cwd().resolve(), Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                (path for path in CANDIDATES if (path / "src").exists()),
                Path.cwd().resolve(),
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import pandas as pd

            DATA_PATH = PROJECT_ROOT / "data" / "final" / "bjb_sentiment_result.csv"
            MODEL_DIR = PROJECT_ROOT / "models" / "lda"

            DATA_PATH, MODEL_DIR
            """
        ),
        make_markdown_cell(
            """
            ## Tahapan Topic Modeling

            - Seleksi dokumen yang relevan
            - Tokenisasi dan pembuatan dictionary serta corpus
            - Pelatihan model LDA
            - Evaluasi coherence score dan interpretasi topik
            """
        ),
        make_code_cell(
            """
            lda_command = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "lda_model.py"),
                "--topics",
                "4",
                "--passes",
                "20",
                "--words-per-topic",
                "10",
            ]
            subprocess.run(lda_command, check=True)
            """
        ),
        make_code_cell(
            """
            pd.read_csv(MODEL_DIR / "lda_topic_terms.csv").head(20)
            """
        ),
    ]
)

VISUALIZATION_NOTEBOOK_CONTENT = build_notebook(
    [
        make_markdown_cell(
            """
            # 05 Visualisasi Hasil

            Notebook ini digunakan untuk membuat visualisasi distribusi sentimen,
            kata dominan, serta ringkasan topik dari hasil analisis.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import subprocess
            import sys

            CANDIDATES = [Path.cwd().resolve(), Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                (path for path in CANDIDATES if (path / "src").exists()),
                Path.cwd().resolve(),
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import pandas as pd

            FINAL_PATH = PROJECT_ROOT / "data" / "final" / "bjb_sentiment_result.csv"
            VIS_DIR = FINAL_PATH.parent / "visualizations"

            FINAL_PATH, VIS_DIR
            """
        ),
        make_markdown_cell(
            """
            ## Ide Visualisasi

            - Distribusi label sentimen
            - Word cloud untuk ulasan positif dan negatif
            - Bar chart frekuensi kata dominan
            - Visualisasi topik dari model LDA
            """
        ),
        make_code_cell(
            """
            visualization_command = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "visualization.py"),
            ]
            subprocess.run(visualization_command, check=True)
            """
        ),
        make_code_cell(
            """
            pd.read_csv(VIS_DIR / "sentiment_distribution.csv")
            """
        ),
    ]
)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix() or "."
    except ValueError:
        return str(path)


def ensure_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(
                f"Expected a directory, but found a file: {path}"
            )
        print(f"Folder already exists: {relative_path(path)}")
        return

    path.mkdir(parents=True, exist_ok=True)
    print(f"Folder created: {relative_path(path)}")


def ensure_file(path: Path, content: str = "") -> None:
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"Expected a file, but found a directory: {path}")

        if path.stat().st_size == 0 and content:
            path.write_text(content, encoding="utf-8")
            print(f"File initialized: {relative_path(path)}")
        else:
            print(f"File already exists: {relative_path(path)}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"File created: {relative_path(path)}")


def main() -> None:
    directories = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "processed" / "annotation_batches",
        PROJECT_ROOT / "data" / "final",
        PROJECT_ROOT / "data" / "final" / "visualizations",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "models" / "indobert",
        PROJECT_ROOT / "models" / "lda",
        PROJECT_ROOT / "src",
    ]

    files = {
        PROJECT_ROOT / "data" / "raw" / "bjb_reviews_raw.csv": "",
        PROJECT_ROOT / "data" / "processed" / "bjb_reviews_clean.csv": "",
        PROJECT_ROOT / "data" / "processed" / "bjb_reviews_semilabeled.csv": "",
        PROJECT_ROOT / "data" / "processed" / "bjb_reviews_manual_review.csv": "",
        PROJECT_ROOT / "data" / "processed" / "bjb_reviews_labeled_final.csv": "",
        PROJECT_ROOT / "data" / "final" / "bjb_sentiment_result.csv": "",
        PROJECT_ROOT / "notebooks" / "01_scraping.ipynb": SCRAPING_NOTEBOOK_CONTENT,
        PROJECT_ROOT
        / "notebooks"
        / "02_preprocessing.ipynb": PREPROCESSING_NOTEBOOK_CONTENT,
        PROJECT_ROOT
        / "notebooks"
        / "03_indobert_training.ipynb": INDOBERT_NOTEBOOK_CONTENT,
        PROJECT_ROOT
        / "notebooks"
        / "04_lda_topic_modeling.ipynb": LDA_NOTEBOOK_CONTENT,
        PROJECT_ROOT
        / "notebooks"
        / "05_visualization.ipynb": VISUALIZATION_NOTEBOOK_CONTENT,
        PROJECT_ROOT / "src" / "__init__.py": "",
        PROJECT_ROOT / "src" / "scraper.py": "",
        PROJECT_ROOT / "src" / "preprocessing.py": "",
        PROJECT_ROOT / "src" / "labeling.py": "",
        PROJECT_ROOT / "src" / "finalize_labels.py": "",
        PROJECT_ROOT / "src" / "train_indobert.py": "",
        PROJECT_ROOT / "src" / "sentiment.py": "",
        PROJECT_ROOT / "src" / "lda_model.py": "",
        PROJECT_ROOT / "src" / "visualization.py": "",
        PROJECT_ROOT / "src" / "utils.py": "",
        PROJECT_ROOT / "requirements.txt": REQUIREMENTS_CONTENT,
        PROJECT_ROOT / "LABELING_GUIDE.md": LABELING_GUIDE_CONTENT,
        PROJECT_ROOT / "README.md": README_CONTENT,
    }

    print(f"Project root: {PROJECT_ROOT}")

    for directory in directories:
        ensure_directory(directory)

    for file_path, content in files.items():
        ensure_file(file_path, content)

    print("Project setup completed.")


if __name__ == "__main__":
    main()
