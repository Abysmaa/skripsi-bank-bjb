# Analisis Sentimen dan Topic Modeling Aspek Keamanan Aplikasi Digi bank bjb

Project ini digunakan untuk penelitian skripsi data science yang berfokus pada
analisis sentimen ulasan aplikasi Digi bank bjb menggunakan IndoBERT serta
topic modeling LDA untuk mengidentifikasi isu utama terkait keamanan, akses,
aktivasi, transaksi, rekening, dan saldo.

## Ringkasan Project

Pipeline aktif project saat ini:

1. Scraping ulasan Google Play Store
2. Preprocessing teks dan infer label awal dari score
3. Semi-labeling dan anotasi manual
4. Finalisasi dataset training
5. Training IndoBERT
6. Inferensi sentimen seluruh dataset
7. Topic modeling LDA untuk review negatif terkait keamanan
8. Visualisasi hasil
9. Dashboard Streamlit untuk presentasi bimbingan dan sidang

Hasil terbaru yang sudah tersedia di repo:

- data mentah hasil scraping dan merge: `6.431` review
- data yang dipakai untuk analisis: `6.396` review
- hasil inferensi sentimen: `3.479` positif dan `2.917` negatif
- akurasi test IndoBERT terbaru: `0.9275`
- dokumen LDA terbaru: `1.065`

## Struktur Folder

```text
skripsi-bank-bjb/
|-- app.py
|-- README.md
|-- LABELING_GUIDE.md
|-- bimbingan.txt
|-- catatan.txt
|-- data/
|   |-- raw/
|   |   `-- bjb_reviews_raw.csv
|   |-- processed/
|   |   |-- bjb_reviews_clean.csv
|   |   |-- bjb_reviews_labeled.csv
|   |   |-- bjb_reviews_labeled_final.csv
|   |   `-- annotation_batches/
|   |       |-- batch_001.csv
|   |       `-- batch_001_backup.csv
|   `-- final/
|       |-- bjb_sentiment_result.csv
|       `-- visualizations/
|-- models/
|   |-- indobert/
|   `-- lda/
|-- notebooks/
|-- src/
|   |-- scraper.py
|   |-- preprocessing.py
|   |-- labeling.py
|   |-- finalize_labels.py
|   |-- auto_annotate.py
|   |-- check_annotation_progress.py
|   |-- train_indobert.py
|   |-- sentiment.py
|   |-- lda_model.py
|   |-- visualization.py
|   `-- utils.py
`-- requirements.txt
```

## Modul `src`

Modul yang aktif dipakai sekarang:

- `scraper.py`
  Untuk scraping review `com.bjbsmb` dengan filter tanggal, checkpoint resume,
  merge ke CSV lama, dan dedup `review_id`.
- `preprocessing.py`
  Untuk cleaning teks, infer label awal dari `score`, dan penandaan
  `is_security`.
- `labeling.py`
  Untuk membuat `semi_label`, `manual_label`, dan batch anotasi manual.
- `finalize_labels.py`
  Untuk menggabungkan `manual_label` dari batch dengan `semi_label` menjadi
  `final_label`.
- `auto_annotate.py`
  Untuk mengisi sebagian `manual_label` otomatis bila teks bertentangan dengan
  `semi_label`.
- `check_annotation_progress.py`
  Untuk memantau progres pengisian `manual_label` pada batch anotasi.
- `train_indobert.py`
  Untuk training IndoBERT dengan opsi binary dan balancing.
- `sentiment.py`
  Untuk inferensi sentimen pada seluruh dataset hasil preprocessing/final.
- `lda_model.py`
  Untuk LDA pada review negatif terkait keamanan dengan custom stopword domain.
- `visualization.py`
  Untuk membuat grafik distribusi sentimen, wordcloud, dan gambar top terms
  LDA.
- `utils.py`
  Untuk path project, logging, dan helper CSV.

## Yang Ditambahkan di Pipeline Baru

Perubahan atau penambahan utama dibanding pipeline awal:

- scraping 2024-2025 dengan `continuation_token`, checkpoint JSON, dan merge ke
  file raw lama
- `clean_content` dan `is_security` pada preprocessing baru
- `batch_001.csv` untuk anotasi manual yang lebih sederhana
- `auto_annotate.py` untuk koreksi otomatis `manual_label`
- `check_annotation_progress.py` untuk memantau progres anotasi
- training IndoBERT terbaru pada dataset final baru dengan mode biner
- LDA terbaru dengan custom stopword domain agar top terms lebih bersih
- sinkronisasi dashboard Streamlit ke hasil training, inferensi, dan LDA baru

## Yang Masih Ada Tapi Tidak Lagi Menjadi Jalur Utama

File atau artefak berikut masih ada di repo, tetapi saat ini bukan jalur utama
pipeline:

- `data/processed/bjb_reviews_semilabeled.csv`
  Artefak lama dari pipeline semi-label sebelumnya.
- `data/processed/bjb_reviews_manual_review.csv`
  Template lama sebelum pipeline batch anotasi baru dipakai.
- `batch_001_backup.csv`
  Backup dari file anotasi otomatis. Bukan file kerja utama.
- `models/indobert/checkpoints/`
  Checkpoint training Hugging Face. Berguna untuk audit/troubleshooting, tetapi
  bukan file yang dipakai langsung saat inferensi akhir.
- `notebooks/*.ipynb`
  Notebook eksplorasi/eksperimen. Pipeline utama sekarang memakai script di
  folder `src/`.
- `setup_project.py`
  Berguna untuk setup awal struktur project, tetapi tidak perlu dijalankan
  untuk workflow inti saat ini jika folder project sudah lengkap.
- `scrape_2024_2025.out.log`, `scrape_2024_2025.err.log`,
  `streamlit.out.log`, `streamlit.err.log`
  File log sesi lokal, bukan bagian artefak ilmiah utama.

## Yang Perlu Diperhatikan

- Command `streamlit run app.py` mungkin tidak tersedia global di Windows Anda.
  Di project ini yang aman adalah memakai interpreter `.venv`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

- Dataset final training sekarang memakai kolom `clean_content`, bukan lagi
  `content_clean` sebagai sumber utama baru. Script inti sudah disesuaikan agar
  tetap kompatibel.

- LDA saat ini dijalankan pada review negatif terkait keamanan setelah
  filtering keyword, bukan pada seluruh review.

## Virtual Environment

Project ini memakai virtual environment lokal `.venv`.

Aktivasi di PowerShell:

```powershell
cd "C:\skripshitt\analisis sentimen\skripsi-bank-bjb"
.\.venv\Scripts\Activate.ps1
```

Install dependency:

```powershell
pip install -r requirements.txt
```

## Contoh Pipeline Aktif

### 1. Scraping

```powershell
.\.venv\Scripts\python.exe src\scraper.py
```

### 2. Preprocessing

```powershell
.\.venv\Scripts\python.exe src\preprocessing.py --infer-label
```

### 3. Labeling

```powershell
.\.venv\Scripts\python.exe src\labeling.py
```

### 4. Cek atau bantu anotasi

```powershell
.\.venv\Scripts\python.exe src\check_annotation_progress.py
.\.venv\Scripts\python.exe src\auto_annotate.py
```

### 5. Finalisasi label

```powershell
.\.venv\Scripts\python.exe src\finalize_labels.py --batch-size 100
```

### 6. Training IndoBERT

```powershell
.\.venv\Scripts\python.exe src\train_indobert.py `
  --input data\processed\bjb_reviews_labeled_final.csv `
  --label-column final_label `
  --binary `
  --balance-strategy undersample `
  --epochs 3 `
  --train-batch-size 8 `
  --eval-batch-size 8
```

### 7. Inferensi sentimen

```powershell
.\.venv\Scripts\python.exe src\sentiment.py --batch-size 16
```

### 8. Topic modeling LDA

```powershell
.\.venv\Scripts\python.exe src\lda_model.py --topics 4 --passes 20 --words-per-topic 10
```

### 9. Visualisasi

```powershell
.\.venv\Scripts\python.exe src\visualization.py
```

### 10. Dashboard Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Artefak Output

Artefak utama yang aktif dipakai:

- `data/raw/bjb_reviews_raw.csv`
- `data/processed/bjb_reviews_clean.csv`
- `data/processed/bjb_reviews_labeled.csv`
- `data/processed/bjb_reviews_labeled_final.csv`
- `data/final/bjb_sentiment_result.csv`
- `models/indobert/training_summary.json`
- `models/indobert/test_predictions.csv`
- `models/lda/lda_topic_terms.csv`
- `models/lda/lda_document_topics.csv`
- `models/lda/lda_visualization.html`
- `data/final/visualizations/*`

## Analisis Project

### Kekuatan project

- pipeline end-to-end sudah lengkap dari scraping sampai dashboard
- ada kombinasi kuantitatif dan tematik: IndoBERT + LDA
- ada anotasi manual sehingga label akhir tidak murni bergantung pada score
- ada dashboard presentasi yang memudahkan bimbingan dan sidang
- hasil terbaru memakai dataset jauh lebih besar, sehingga lebih representatif

### Titik yang sudah bagus

- training script, inferensi, dan dashboard sudah sinkron ke hasil terbaru
- LDA sudah diperbaiki dengan custom stopword sehingga top terms lebih tajam
- ada helper anotasi (`auto_annotate.py` dan `check_annotation_progress.py`)
- struktur output sudah cukup rapi untuk kebutuhan skripsi

### Titik yang masih lemah atau perlu dicatat

- `README.md`, `catatan.txt`, `bimbingan.txt`, Bab 4, Bab 5, dan outline PPT
  perlu dijaga tetap konsisten ketika angka hasil berubah
- masih ada artefak lama di `data/processed/` yang bisa membingungkan jika
  tidak dibedakan antara pipeline lama dan baru
- topik LDA masih memerlukan interpretasi peneliti, jadi jangan dianggap
  otomatis final tanpa pembacaan konteks
- belum ada baseline model klasik jika dosen meminta pembanding metodologis
- beberapa token hasil LDA masih bisa diperketat lagi bila Anda ingin narasi
  topik lebih tajam

### Rekomendasi praktis

- gunakan `bjb_reviews_labeled_final.csv`, `bjb_sentiment_result.csv`, dan
  artefak `models/lda/` sebagai sumber angka utama
- jadikan script di `src/` sebagai workflow resmi, bukan notebook
- jika menulis skripsi, update Bab 4, Bab 5, ringkasan hasil, dan slide dari
  angka artefak terbaru, bukan dari eksperimen awal
- simpan file backup anotasi, tetapi gunakan hanya `batch_001.csv` sebagai file
  kerja aktif

## Dashboard

Dashboard lokal tersedia di:

```text
http://localhost:8501
```

Fitur dashboard:

- ringkasan distribusi sentimen
- evaluasi IndoBERT
- eksplorasi review hasil prediksi
- topik LDA dan contoh dokumen
- rekomendasi untuk Bank BJB
- kesimpulan sidang
- demo prediksi review baru
