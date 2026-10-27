# 🔎 Analisis Sentimen Ulasan Digi bank bjb

**Penelitian skripsi | IndoBERT untuk klasifikasi sentimen · LDA untuk pemetaan topik**

Proyek ini menganalisis ulasan pengguna aplikasi Digi bank bjb di Google Play
Store. Tujuannya adalah merangkum polaritas sentimen pengguna dan memahami
tema keluhan yang muncul pada ulasan negatif terpilih—terutama seputar
aktivasi, akses akun, transaksi, dan saldo.

> **Ruang lingkup:** hasil penelitian menggambarkan pengalaman dan persepsi
> yang disampaikan dalam ulasan. Hasil ini bukan audit, pengujian, atau bukti
> keamanan teknis aplikasi.

## ✨ Sekilas Hasil

| Indikator | Hasil |
|---|---:|
| Ulasan setelah preprocessing | **6.396** |
| Sentimen positif | **3.479 (54,39%)** |
| Sentimen negatif | **2.917 (45,61%)** |
| Akurasi IndoBERT pada data uji | **92,75%** |
| Data uji | **993 ulasan** |
| Ulasan terpilih untuk LDA | **1.065** |
| Jumlah topik LDA | **4** |

Angka inferensi sentimen menggunakan klasifikasi dua kelas. Ulasan netral tidak
ditampilkan sebagai kelas tersendiri pada hasil inferensi. Data LDA merupakan
bagian dari ulasan negatif yang lolos penyaringan; proporsi topik tidak
menyatakan prevalensi masalah pada seluruh pengguna.

## 🧭 Alur Penelitian

1. Mengumpulkan dan menggabungkan ulasan Google Play Store.
2. Membersihkan teks, membentuk label awal dari rating, dan menyiapkan koreksi
   label.
3. Melatih dan mengevaluasi IndoBERT pada klasifikasi sentimen biner.
4. Menginferensikan sentimen pada ulasan hasil preprocessing.
5. Memilih ulasan negatif yang relevan untuk pemodelan topik LDA.
6. Menyajikan hasil melalui visualisasi dan dashboard Streamlit.

## 📌 Temuan Utama

- Sentimen positif lebih banyak daripada sentimen negatif dengan selisih
  **8,78 poin persentase**.
- Pada data uji, IndoBERT mencatat akurasi **92,75%** dan F1-score tertimbang
  **92,75%**.
- Empat tema utama pada dokumen LDA terpilih berkaitan dengan aktivasi dan
  pembukaan rekening, aktivasi pasca-pembaruan, login dan password, serta
  transaksi dan saldo.
- Temuan LDA berlaku untuk kumpulan dokumen yang dianalisis, bukan untuk
  keseluruhan pengguna atau seluruh ulasan.

## 📚 Dokumen Penelitian

- [Ringkasan Hasil](RINGKASAN_HASIL.md)
- [Panduan Labeling](LABELING_GUIDE.md)

Bab skripsi dan materi sidang tidak disertakan dalam repository ini.

## 🚀 Deploy Dashboard di Streamlit Community Cloud

Dashboard dapat diterbitkan sebagai aplikasi Streamlit Cloud dari repository
GitHub ini. Bobot IndoBERT tersimpan menggunakan Git LFS; Community Cloud
mengambil file LFS saat proses deploy.

1. Masuk ke [Streamlit Community Cloud](https://share.streamlit.io/) dan pilih
   **Create app**.
2. Pilih repository `Abysmaa/skripsi-bank-bjb`, branch `main`, dan file
   `app.py`.
3. Pada **Advanced settings**, gunakan Python 3.12.
4. Pilih **Deploy** dan tunggu proses instalasi dependency serta pemuatan
   model selesai.

Jika menjalankan aplikasi secara lokal, pastikan Git LFS tersedia, lalu ambil
model dan pasang dependency:

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Jalankan dashboard Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Secara default dashboard tersedia di `http://localhost:8501`.

Dashboard menampilkan ringkasan sentimen, evaluasi model, eksplorasi ulasan,
hasil topik LDA, rekomendasi, dan demo prediksi teks.

## Struktur Folder

```text
skripsi-bank-bjb/
|-- app.py
|-- README.md
|-- LABELING_GUIDE.md
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

## 🔍 Catatan Interpretasi

- Label awal dibentuk dari rating. Sebagian label dikoreksi menggunakan aturan
  frasa; proses ini tidak boleh ditafsirkan sebagai anotasi manusia independen
  untuk seluruh data.
- Model akhir menggunakan dua kelas. Karena itu, distribusi inferensi tidak
  menampilkan sentimen netral sebagai kelas tersendiri.
- LDA dijalankan pada ulasan negatif terpilih setelah penyaringan kata kunci,
  bukan pada seluruh dataset.
- Topik LDA perlu dibaca bersama kata dominan dan isi ulasan. Nama topik adalah
  interpretasi, bukan label otomatis yang diberikan model.

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

## 🗂️ Artefak Output

Hasil utama yang dihasilkan atau digunakan oleh pipeline:

- `data/raw/bjb_reviews_raw.csv` — ulasan mentah
- `data/processed/bjb_reviews_clean.csv` — ulasan setelah preprocessing
- `data/processed/bjb_reviews_labeled_final.csv` — label final
- `data/final/bjb_sentiment_result.csv` — hasil inferensi sentimen
- `models/indobert/training_summary.json` — ringkasan evaluasi IndoBERT
- `models/indobert/test_predictions.csv` — prediksi pada data uji
- `models/lda/lda_topic_terms.csv` — kata dominan topik
- `models/lda/lda_document_topics.csv` — topik dominan per dokumen
- `data/final/visualizations/` — visualisasi hasil

Untuk ringkasan hasil publik, lihat [Ringkasan Hasil](RINGKASAN_HASIL.md).
Gunakan angka dari artefak terbaru secara konsisten saat memperbarui materi
presentasi.
