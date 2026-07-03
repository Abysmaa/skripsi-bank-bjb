# Outline PPT Sidang

## Slide 1: Judul

- Analisis Sentimen dan Topic Modeling Aspek Keamanan Aplikasi Digi bank bjb
- Metode: IndoBERT dan LDA
- Nama, NIM, Program Studi, Tahun

## Slide 2: Latar Belakang

- Mobile banking menjadi layanan penting dalam aktivitas keuangan digital
- Ulasan pengguna mencerminkan kepuasan, keluhan, dan persepsi keamanan
- Dibutuhkan analisis otomatis untuk memetakan sentimen dan isu dominan

## Slide 3: Rumusan Masalah

- Bagaimana sentimen pengguna terhadap aplikasi Digi bank bjb?
- Seberapa baik IndoBERT mengklasifikasikan sentimen ulasan?
- Topik apa saja yang dominan pada ulasan terkait keamanan dan keandalan?

## Slide 4: Tujuan Penelitian

- Mengklasifikasikan sentimen ulasan pengguna
- Mengukur performa model IndoBERT
- Mengidentifikasi topik utama dengan LDA
- Memberikan insight untuk evaluasi layanan aplikasi

## Slide 5: Metodologi

- Sumber data: ulasan pengguna aplikasi Digi bank bjb
- Preprocessing: cleaning, normalisasi, stopword removal, stemming
- Labeling: semi-label + anotasi manual
- Klasifikasi: IndoBERT
- Topic modeling: LDA

## Slide 6: Alur Penelitian

- Scraping data
- Preprocessing
- Labeling dan finalisasi label
- Training IndoBERT
- Inferensi sentimen
- Topic modeling LDA
- Visualisasi dan interpretasi

## Slide 7: Dataset

- Total ulasan terprediksi: 1.489
- Distribusi hasil prediksi:
- Positif: 966 review atau 64,88%
- Negatif: 523 review atau 35,12%

## Slide 8: Hasil Evaluasi IndoBERT

- Akurasi data uji: 0,9487
- F1-score tertimbang: 0,9487
- Precision negatif: 0,9400
- Recall negatif: 0,9592
- Precision positif: 0,9579
- Recall positif: 0,9381

## Slide 9: Interpretasi Hasil Sentimen

- Sentimen positif lebih dominan
- Pengguna banyak menilai aplikasi membantu transaksi dan cukup mudah digunakan
- Sentimen negatif terkonsentrasi pada masalah inti layanan

## Slide 10: Hasil Topic Modeling

- Topik 0: aktivasi ulang, kesalahan kredensial, jaringan
- Topik 1: transaksi error dan ketidakstabilan aplikasi
- Topik 2: pembukaan rekening, aktivasi, saldo atau dana
- Topik 3: login, password, dan gangguan pasca-update

## Slide 11: Pembahasan

- Masalah utama ada pada proses inti, bukan hanya fitur tambahan
- Persepsi keamanan dipengaruhi oleh kelancaran login, aktivasi, dan transaksi
- Update aplikasi menjadi titik kritis yang sering memicu keluhan

## Slide 12: Kesimpulan

- Sentimen positif dominan, tetapi keluhan negatif masih signifikan
- IndoBERT efektif untuk klasifikasi sentimen
- LDA berhasil mengidentifikasi isu utama terkait keamanan dan keandalan

## Slide 13: Saran

- Perbaiki alur aktivasi dan pemulihan akun
- Tingkatkan stabilitas login dan transaksi
- Lengkapi data administratif pembukaan rekening
- Perjelas penanganan transaksi gagal

## Slide 14: Penutup

- Terima kasih
- Sesi tanya jawab

## Catatan Penyampaian

- Tekankan angka utama sejak awal: 64,88% positif dan akurasi 0,9487
- Saat membahas topik, fokus pada implikasi bisnis dan pengalaman pengguna
- Jangan membaca semua isi slide, cukup jelaskan poin pentingnya
