# BAB III METODOLOGI PENELITIAN

## 3.1 Pendekatan Penelitian

Penelitian ini menggunakan pendekatan kuantitatif dengan teknik *text mining*
dan *natural language processing*. Analisis dilakukan terhadap ulasan pengguna
aplikasi Digi bank bjb untuk memperoleh dua jenis hasil. Pertama, sentimen
ulasan diklasifikasikan menjadi positif dan negatif menggunakan IndoBERT.
Kedua, topik dominan pada ulasan negatif yang berkaitan dengan keamanan,
akses, dan keandalan layanan diidentifikasi menggunakan Latent Dirichlet
Allocation (LDA).

IndoBERT digunakan sebagai metode klasifikasi terawasi karena model dilatih
dengan data yang telah memiliki label sentimen. LDA digunakan sebagai metode
tidak terawasi karena topik dibentuk berdasarkan pola kemunculan kata tanpa
label topik yang ditentukan pada setiap dokumen. Hasil kedua metode kemudian
diinterpretasikan untuk menjelaskan persepsi pengguna. Penelitian ini tidak
melakukan pengujian kerentanan atau audit keamanan teknis aplikasi.

## 3.2 Objek dan Sumber Data

Objek penelitian adalah ulasan pengguna aplikasi Digi bank bjb pada Google Play
Store dengan ID aplikasi `com.bjbsmb`. Data diambil menggunakan pustaka
`google-play-scraper` dengan pengaturan bahasa Indonesia (`id`), negara
Indonesia (`id`), dan urutan ulasan terbaru.

Data hasil scraping digabungkan dengan data mentah yang telah tersedia,
kemudian dideduplikasi berdasarkan `review_id`. Dataset akhir memuat 6.431
ulasan dengan rentang tanggal aktual 13 Juli 2024 sampai 18 April 2026.
Perbedaan rentang tersebut dengan rentang pada satu sesi scraping terjadi
karena file akhir merupakan hasil penggabungan beberapa pengumpulan data.

Atribut data mentah yang disimpan meliputi:

| Atribut | Keterangan |
|---|---|
| `review_id` | Identitas unik ulasan |
| `user_name` | Nama pengguna yang tampil pada ulasan |
| `score` | Rating pengguna dalam skala 1–5 |
| `content` | Teks ulasan |
| `at` | Waktu ulasan diterbitkan |
| `app_version` | Versi aplikasi saat ulasan dibuat |
| `thumbs_up_count` | Jumlah penilaian membantu |
| `reply_content` | Balasan pengelola aplikasi |
| `replied_at` | Waktu balasan diterbitkan |

Identitas pengguna tidak digunakan sebagai fitur klasifikasi maupun sebagai
dasar interpretasi hasil.

## 3.3 Perangkat Penelitian

Penelitian diimplementasikan menggunakan Python 3.12. Pustaka utama yang
digunakan meliputi:

1. `pandas` dan `numpy` untuk pengolahan data;
2. `google-play-scraper` untuk pengumpulan ulasan;
3. `transformers`, `torch`, dan `datasets` untuk pelatihan dan inferensi
   IndoBERT;
4. `scikit-learn` untuk pembagian data dan evaluasi model;
5. `gensim` untuk pemodelan topik LDA;
6. `pyLDAvis`, `matplotlib`, dan `wordcloud` untuk visualisasi; serta
7. `streamlit` untuk dashboard hasil penelitian.

Seluruh proses utama dijalankan melalui skrip pada folder `src`, sedangkan
dataset, model, metrik, dan visualisasi disimpan sebagai artefak terpisah agar
hasil dapat ditelusuri kembali.

## 3.4 Tahapan Penelitian

Tahapan penelitian terdiri atas:

1. pengumpulan dan penggabungan ulasan Google Play Store;
2. preprocessing dan deduplikasi data;
3. pembentukan semi-label berdasarkan rating;
4. penyiapan dan koreksi label pada batch anotasi;
5. finalisasi dataset berlabel;
6. penyiapan, pelatihan, dan evaluasi IndoBERT;
7. inferensi sentimen pada seluruh data hasil preprocessing;
8. pemilihan ulasan negatif terkait keamanan dan layanan;
9. pemodelan topik menggunakan LDA; dan
10. visualisasi serta interpretasi hasil.

Setiap tahap menghasilkan file keluaran yang menjadi masukan bagi tahap
berikutnya. Dengan alur tersebut, angka pada hasil penelitian dapat diverifikasi
dari dataset dan ringkasan model yang tersimpan.

## 3.5 Pengumpulan Data

Scraping dilakukan secara bertahap dengan jumlah maksimal 200 ulasan pada
setiap permintaan. Sistem menggunakan *continuation token* untuk mengambil
halaman berikutnya dan menyimpan checkpoint sehingga proses dapat dilanjutkan
apabila terhenti. Jeda 1,5 detik diterapkan antarpermintaan.

Ulasan dari proses scraping digabungkan dengan file data mentah yang sudah ada.
Baris dengan `review_id` yang sama hanya disimpan satu kali. Data selanjutnya
diurutkan berdasarkan waktu ulasan dari yang terbaru. Proses tersebut
menghasilkan file `data/raw/bjb_reviews_raw.csv`.

## 3.6 Preprocessing Teks

Preprocessing dilakukan untuk menghasilkan teks yang konsisten sebelum
pelabelan, pelatihan, dan analisis topik. Tahapannya adalah:

1. mengubah seluruh huruf menjadi huruf kecil;
2. menghapus URL, mention, dan hashtag;
3. menghapus karakter non-ASCII;
4. menghapus karakter selain huruf, angka, dan spasi;
5. merapikan spasi berlebih;
6. menghapus baris dengan teks bersih kosong; dan
7. menghapus duplikasi berdasarkan `review_id`.

Proses ini menghasilkan 6.396 ulasan pada
`data/processed/bjb_reviews_clean.csv`. Preprocessing untuk IndoBERT tidak
menerapkan stemming dan penghapusan stopword karena konteks urutan kata tetap
dipertahankan sebagai masukan model transformer. Penghapusan stopword khusus
baru diterapkan pada tahap LDA.

Pada tahap yang sama, ulasan ditandai sebagai terkait keamanan apabila memuat
kata atau frasa seperti `login`, `password`, `OTP`, `PIN`, `verifikasi`,
`akun`, `sidik jari`, dan `keamanan`. Penandaan ini merupakan penyaringan
berbasis kata kunci, bukan penilaian keamanan teknis.

## 3.7 Pelabelan Sentimen

### 3.7.1 Pembentukan Semi-label

Semi-label dibentuk dari rating pengguna dengan ketentuan:

| Rating | Semi-label |
|---:|---|
| 1–2 | Negatif |
| 3 | Netral |
| 4–5 | Positif |

Rating digunakan sebagai label awal karena merepresentasikan evaluasi eksplisit
pengguna. Namun, rating dan isi teks dapat bertentangan sehingga diperlukan
tahap koreksi.

### 3.7.2 Pemilihan Batch Anotasi

Batch anotasi terdiri atas seluruh ulasan dengan rating 3, ditambah sampel acak
10% dari ulasan rating 1–2 dan 10% dari ulasan rating 4–5. Pengambilan sampel
menggunakan *random seed* 42. Berdasarkan data aktif, batch tersebut berjumlah
963 ulasan.

Koreksi label pada implementasi aktif dibantu oleh aturan frasa positif dan
negatif. Label koreksi hanya diisi ketika hasil deteksi teks berbeda dari
semi-label. Dari 963 ulasan, terdapat 429 koreksi otomatis, terdiri atas 151
label negatif, 210 netral, dan 68 positif. Sebanyak 534 baris tanpa koreksi
tetap menggunakan semi-label. Dengan demikian, proses ini merupakan pelabelan
semi-otomatis berbasis rating dan aturan kata, bukan anotasi manual penuh oleh
beberapa anotator.

### 3.7.3 Finalisasi Label

Apabila label koreksi tersedia, label tersebut digunakan sebagai
`final_label`. Jika tidak tersedia, sistem menggunakan `semi_label`. Dataset
final terdiri atas 2.522 ulasan negatif, 360 ulasan netral, dan 3.514 ulasan
positif. Dataset disimpan pada
`data/processed/bjb_reviews_labeled_final.csv`.

## 3.8 Klasifikasi Sentimen Menggunakan IndoBERT

### 3.8.1 Penyiapan Dataset

Model dasar yang digunakan adalah `indobenchmark/indobert-base-p1`. Eksperimen
akhir menggunakan klasifikasi biner sehingga 360 ulasan netral tidak
disertakan dalam pelatihan. Teks kosong dan duplikasi berdasarkan kombinasi
teks bersih serta label juga dihapus.

Setelah proses tersebut, tersedia 2.481 ulasan negatif dan 2.695 ulasan
positif. Metode *random undersampling* diterapkan pada kelas positif sehingga
masing-masing kelas berjumlah 2.481 ulasan atau 4.962 ulasan secara keseluruhan.
Penyeimbangan dilakukan sebelum pembagian data.

Data dibagi secara terstratifikasi menggunakan *random seed* 42. Sebanyak 20%
digunakan sebagai data uji. Dari 80% data yang tersisa, 20% digunakan sebagai
data validasi. Hasil pembagian terdiri atas 3.175 data latih, 794 data
validasi, dan 993 data uji.

### 3.8.2 Tokenisasi dan Fine-tuning

Teks ditokenisasi menggunakan tokenizer IndoBERT. Padding diterapkan secara
dinamis pada setiap batch, sedangkan teks yang melebihi batas dipotong pada
panjang maksimal 256 token. Parameter pelatihan ditampilkan pada tabel berikut.

| Parameter | Nilai |
|---|---:|
| Model dasar | `indobenchmark/indobert-base-p1` |
| Jumlah kelas | 2 |
| Epoch | 3 |
| Learning rate | 0,00002 |
| Batch size pelatihan | 8 |
| Batch size evaluasi | 8 |
| Panjang token maksimal | 256 |
| Weight decay | 0,01 |
| Random seed | 42 |
| Metrik pemilihan model terbaik | F1-score tertimbang |

Evaluasi dan penyimpanan checkpoint dilakukan pada setiap epoch. Model dengan
F1-score tertimbang terbaik pada data validasi dimuat kembali sebagai model
akhir.

### 3.8.3 Evaluasi Model

Performa model diukur menggunakan confusion matrix, accuracy, precision,
recall, dan F1-score. Komponen confusion matrix terdiri atas *true positive*
(TP), *true negative* (TN), *false positive* (FP), dan *false negative* (FN).

Rumus accuracy adalah:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Rumus precision adalah:

`Precision = TP / (TP + FP)`

Rumus recall adalah:

`Recall = TP / (TP + FN)`

Rumus F1-score adalah:

`F1-score = 2 × (Precision × Recall) / (Precision + Recall)`

Selain nilai per kelas, penelitian menggunakan F1-score tertimbang. Nilai ini
menghitung rata-rata F1-score setiap kelas dengan mempertimbangkan jumlah data
aktual pada kelas tersebut.

### 3.8.4 Inferensi Sentimen

Model akhir digunakan untuk memprediksi 6.396 ulasan hasil preprocessing dengan
batch size 16. Nilai probabilitas kelas dihitung menggunakan fungsi softmax.
Kelas dengan probabilitas terbesar menjadi `predicted_label`, sedangkan nilai
probabilitasnya disimpan sebagai `confidence`. Hasil inferensi disimpan pada
`data/final/bjb_sentiment_result.csv`.

## 3.9 Topic Modeling Menggunakan LDA

### 3.9.1 Pemilihan Dokumen

LDA diterapkan pada ulasan yang memenuhi dua syarat, yaitu diprediksi negatif
oleh IndoBERT dan memuat kata kunci terkait keamanan, akses, rekening, atau
transaksi. Kata kunci mencakup antara lain `akun`, `aktivasi`, `blokir`,
`login`, `OTP`, `password`, `PIN`, `rekening`, `transfer`, `transaksi`, dan
`verifikasi`.

Setelah penyaringan dan penghapusan dokumen tanpa token, terdapat 1.065 dokumen
yang digunakan untuk LDA.

### 3.9.2 Persiapan Token

Teks bersih dipecah berdasarkan spasi. Token dengan panjang kurang dari empat
karakter dan token dalam daftar custom stopword dihapus. Daftar tersebut
mencakup kata umum bahasa Indonesia serta kata domain yang terlalu umum,
seperti `aplikasi`, `bank`, `bjb`, dan `digi`.

Dictionary dibentuk dari token yang tersisa. Kata yang muncul pada kurang dari
tiga dokumen atau lebih dari 85% dokumen dihapus. Setiap dokumen kemudian
diubah menjadi representasi *bag-of-words* berupa pasangan ID kata dan
frekuensinya.

### 3.9.3 Pembentukan dan Interpretasi Topik

LDA memodelkan setiap dokumen sebagai campuran beberapa topik dan setiap topik
sebagai distribusi probabilitas kata. Konfigurasi model adalah:

| Parameter | Nilai |
|---|---:|
| Jumlah topik | 4 |
| Passes | 20 |
| Kata yang ditampilkan per topik | 10 |
| Alpha | Otomatis |
| Eta | Otomatis |
| Random seed | 42 |

Topik dominan suatu dokumen ditentukan dari topik dengan probabilitas terbesar.
Penamaan topik tidak dihasilkan secara otomatis oleh model. Peneliti memberi
nama berdasarkan sepuluh kata dengan bobot tertinggi dan pembacaan contoh
dokumen pada setiap topik.

## 3.10 Visualisasi dan Penyajian Hasil

Hasil penelitian disajikan dalam bentuk grafik distribusi sentimen, word cloud
sentimen positif dan negatif, grafik kata dominan setiap topik, serta
visualisasi interaktif pyLDAvis. Dashboard Streamlit digunakan untuk
menampilkan ringkasan data, evaluasi IndoBERT, hasil LDA, contoh ulasan, dan
demo prediksi. Visualisasi dan dashboard berfungsi sebagai media penyajian,
bukan sebagai metode evaluasi model.

## 3.11 Validitas dan Reproduksibilitas

Reproduksibilitas dijaga dengan menggunakan *random seed* 42 pada sampling,
pembagian data, penyeimbangan, dan LDA. Data latih, data uji, model, tokenizer,
prediksi, metrik evaluasi, dan keluaran LDA disimpan sebagai artefak. Data uji
dipisahkan dari data latih dan validasi agar pengukuran performa dilakukan pada
data yang tidak digunakan untuk mengoptimalkan model.

Validitas penelitian tetap dibatasi oleh penggunaan rating dan aturan kata
sebagai dasar pelabelan, belum adanya pengukuran kesepakatan antar-anotator,
serta penggunaan kata kunci untuk memilih dokumen LDA. Keterbatasan tersebut
dipertimbangkan dalam interpretasi hasil dan dijelaskan kembali pada Bab IV.
