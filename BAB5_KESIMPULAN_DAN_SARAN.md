# BAB V KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Berdasarkan analisis sentimen menggunakan IndoBERT dan pemodelan topik
menggunakan LDA terhadap ulasan aplikasi Digi bank bjb, diperoleh kesimpulan
sebagai berikut.

1. Dari 6.396 ulasan yang dianalisis, IndoBERT memprediksi 3.479 ulasan
   (54,39%) sebagai sentimen positif dan 2.917 ulasan (45,61%) sebagai sentimen
   negatif. Sentimen positif masih dominan, tetapi selisih 8,78 poin persentase
   menunjukkan bahwa keluhan pengguna tetap memiliki proporsi yang besar.

2. Model IndoBERT memperoleh akurasi 92,75% dan F1-score tertimbang 92,75% pada
   993 data uji. Model mengklasifikasikan dengan benar 467 ulasan negatif dan
   454 ulasan positif. Recall negatif sebesar 93,96% menunjukkan bahwa model
   mampu mendeteksi sebagian besar ulasan keluhan pada data uji.

3. LDA terhadap 1.065 ulasan negatif yang relevan dengan keamanan, akses, dan
   layanan transaksi menghasilkan empat topik. Topik tersebut meliputi
   aktivasi dan pembukaan rekening (20,85%), aktivasi pasca-pembaruan (21,97%),
   login dan password (31,08%), serta transaksi, saldo, dan transfer (26,10%).
   Topik login dan password menjadi kelompok terbesar.

4. Keluhan pengguna terutama terkonsentrasi pada proses inti layanan, yaitu
   aktivasi, autentikasi, akses akun, kestabilan jaringan, serta kepastian
   transaksi dan saldo. Bagi pengguna, persepsi keamanan berkaitan erat dengan
   kemampuan mengakses akun secara konsisten dan memperoleh status transaksi
   yang jelas.

5. Kombinasi IndoBERT dan LDA dapat memberikan dua jenis informasi yang saling
   melengkapi. IndoBERT mengukur polaritas ulasan secara kuantitatif, sedangkan
   LDA membantu menjelaskan isu yang mendasari ulasan negatif secara tematik.
   Meskipun demikian, hasil penelitian menggambarkan persepsi pengguna dan
   tidak dapat dianggap sebagai audit atau bukti keamanan teknis aplikasi.

## 5.2 Saran

### 5.2.1 Saran bagi Pengelola Aplikasi

1. Memprioritaskan perbaikan alur aktivasi ulang, login, dan pemulihan akun,
   terutama setelah pembaruan aplikasi, karena dua topik terkait aktivasi dan
   login mencakup 53,05% dokumen LDA.

2. Menyediakan pesan kesalahan yang spesifik dan dapat ditindaklanjuti untuk
   kegagalan password, nomor terdaftar, jaringan, dan verifikasi. Pesan perlu
   membedakan gangguan perangkat, koneksi, kredensial, dan sistem.

3. Mengurangi kebutuhan percobaan aktivasi berulang yang menggunakan pulsa dan
   menyediakan mekanisme verifikasi alternatif ketika proses utama gagal.

4. Meningkatkan stabilitas transaksi serta menampilkan status transfer,
   pembaruan saldo, estimasi penyelesaian, dan prosedur pengembalian dana secara
   jelas ketika transaksi gagal atau tertunda.

5. Melakukan pengujian regresi khusus pada fungsi login, aktivasi, pembukaan
   rekening, transfer, dan pembaruan saldo sebelum setiap rilis aplikasi.

6. Memantau ulasan secara berkala berdasarkan versi aplikasi dan waktu rilis
   agar peningkatan keluhan setelah pembaruan dapat dideteksi lebih cepat.

### 5.2.2 Saran bagi Penelitian Selanjutnya

1. Mengembangkan klasifikasi tiga kelas agar sentimen netral tidak dipaksa
   menjadi negatif atau positif.

2. Memperkuat anotasi dengan lebih dari satu anotator serta melaporkan tingkat
   kesepakatan antar-anotator untuk mengurangi subjektivitas label.

3. Membandingkan IndoBERT dengan model dasar seperti Naive Bayes, Support
   Vector Machine, atau model transformer bahasa Indonesia lainnya agar
   peningkatan performa dapat dinilai secara empiris.

4. Menentukan jumlah topik LDA melalui perbandingan koherensi, stabilitas
   topik, dan penilaian interpretabilitas, bukan hanya menetapkan jumlah topik
   sejak awal.

5. Menambahkan analisis temporal dan versi aplikasi untuk mengukur perubahan
   sentimen sebelum dan sesudah pembaruan.

6. Memperluas sumber data ke kanal lain, seperti media sosial dan data
   pengaduan resmi, dengan tetap memperhatikan izin, privasi, dan etika
   penelitian.

7. Jika tujuan penelitian diperluas untuk menilai keamanan teknis, analisis
   ulasan perlu dilengkapi dengan metode audit keamanan yang sah dan
   berizin. Analisis sentimen tidak cukup untuk menyimpulkan keberadaan
   kerentanan pada sistem.
