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

- Total ulasan setelah preprocessing: 6.396
- Distribusi hasil prediksi:
- Positif: 3.479 ulasan atau 54,39%
- Negatif: 2.917 ulasan atau 45,61%
- Catatan: inferensi menggunakan dua kelas; ulasan netral tidak menjadi kelas
  tersendiri

## Slide 8: Hasil Evaluasi IndoBERT

- Data uji: 993 ulasan
- Akurasi data uji: 92,75%
- F1-score tertimbang: 92,75%
- F1-score negatif: 92,84%
- F1-score positif: 92,65%
- Confusion matrix: 467 negatif dan 454 positif terklasifikasi benar

## Slide 9: Interpretasi Hasil Sentimen

- Sentimen positif lebih dominan
- Selisih sentimen positif dan negatif sebesar 8,78 poin persentase
- Proporsi negatif tetap besar dan perlu diperhatikan
- Hasil menggambarkan ulasan yang dianalisis, bukan seluruh pengguna aplikasi

## Slide 10: Hasil Topic Modeling

- Data LDA: 1.065 ulasan negatif terpilih
- Topik 0 (20,85%): aktivasi, pembukaan rekening, dan koneksi
- Topik 1 (21,97%): aktivasi pasca-pembaruan dan jaringan
- Topik 2 (31,08%): login, password, dan akses akun
- Topik 3 (26,10%): transaksi, saldo, dan transfer

## Slide 11: Pembahasan

- Keluhan dalam ulasan terpilih berkaitan dengan akses akun dan transaksi
- Persepsi keamanan pengguna beririsan dengan keandalan akses dan kejelasan
  status layanan
- Temuan ini adalah persepsi pengguna, bukan bukti adanya masalah keamanan
  teknis

## Slide 12: Kesimpulan

- Sentimen positif dominan, tetapi keluhan negatif masih signifikan
- Akurasi IndoBERT pada data uji sebesar 92,75%
- LDA mengelompokkan ulasan negatif terpilih ke dalam empat tema utama
- Hasil menjelaskan polaritas dan tema ulasan, bukan kondisi keamanan teknis

## Slide 13: Saran

- Prioritaskan perbaikan alur aktivasi ulang, login, dan pemulihan akun
- Tingkatkan kejelasan pesan kesalahan dan status transaksi
- Pantau ulasan berdasarkan waktu dan versi aplikasi
- Kembangkan penelitian lanjutan sesuai keterbatasan yang telah diuraikan

## Slide 14: Penutup

- Terima kasih
- Sesi tanya jawab

## Catatan Penyampaian

- Tekankan angka utama sejak awal: 54,39% positif, 45,61% negatif, dan
  akurasi uji 92,75%
- Saat membahas topik, fokus pada implikasi bisnis dan pengalaman pengguna
- Tegaskan bahwa hasil menggambarkan persepsi dari ulasan, bukan audit keamanan
  teknis
- Jangan membaca semua isi slide, cukup jelaskan poin pentingnya
