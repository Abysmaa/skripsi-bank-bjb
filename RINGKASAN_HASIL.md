# Ringkasan Hasil dan Pembahasan

Dokumen ini merangkum hasil utama analisis sentimen dan topic modeling pada
ulasan aplikasi Digi bank bjb berdasarkan artefak terbaru project.

## 1. Gambaran Hasil Utama

Setelah preprocessing, terdapat 6.396 ulasan yang dianalisis. IndoBERT
memprediksi 3.479 ulasan (54,39%) sebagai sentimen positif dan 2.917 ulasan
(45,61%) sebagai sentimen negatif.

Sentimen positif lebih banyak dengan selisih 8,78 poin persentase. Namun,
proporsi negatif yang masih besar menunjukkan bahwa keluhan pengguna tetap
menjadi temuan penting. Hasil ini menggambarkan persepsi dalam ulasan yang
dianalisis dan tidak mewakili seluruh pengguna aplikasi.

## 2. Kinerja Model IndoBERT

Model IndoBERT dilatih menggunakan dataset final yang dijelaskan pada Bab III.
Untuk tahap klasifikasi, eksperimen dijalankan dalam skema biner, yaitu label
`negative` dan `positive`.

Hasil evaluasi model pada data uji sebanyak 993 ulasan adalah:

- Akurasi data uji: 92,75%
- F1-score tertimbang data uji: 92,75%
- Precision kelas negatif: 91,75%
- Recall kelas negatif: 93,96%
- F1-score kelas negatif: 92,84%
- Precision kelas positif: 93,80%
- Recall kelas positif: 91,53%
- F1-score kelas positif: 92,65%

Hasil tersebut menunjukkan kinerja yang relatif seimbang pada dua kelas dalam
data uji. Recall negatif sebesar 93,96% menunjukkan bahwa 467 dari 497 ulasan
negatif pada data uji dikenali sebagai negatif. Metrik ini menilai kinerja pada
data uji berlabel; hasilnya tidak dengan sendirinya menjamin ketepatan semua
prediksi pada keseluruhan data.

Confusion matrix pada data uji menunjukkan:

- 467 ulasan negatif terklasifikasi benar sebagai negatif
- 454 ulasan positif terklasifikasi benar sebagai positif
- 30 ulasan negatif salah diprediksi sebagai positif
- 42 ulasan positif salah diprediksi sebagai negatif

Hasil inferensi pada keseluruhan data mencakup dua kelas. Karena model akhir
bersifat biner, 360 ulasan berlabel netral tidak menjadi kelas tersendiri pada
inferensi; pada perbandingan dengan label akhir, 310 dipetakan ke negatif dan
50 ke positif.

## 3. Interpretasi Distribusi Sentimen

Dominasi sentimen positif menunjukkan bahwa sebagian besar pengguna masih
menilai aplikasi bermanfaat untuk transaksi harian. Review positif yang muncul
berulang umumnya menekankan kemudahan penggunaan, kelengkapan fitur, kecepatan
transaksi, dan kepuasan setelah pembaruan aplikasi.

Di sisi lain, kelompok sentimen negatif memperlihatkan pola keluhan yang cukup
konsisten. Review negatif banyak berisi pengalaman kegagalan login, password
atau sidik jari yang dianggap salah, aktivasi ulang yang bermasalah, indikator
jaringan yang tidak stabil, transaksi gagal, serta hambatan saat membuka
rekening atau mengakses fitur tertentu.

Dengan demikian, meskipun citra umum aplikasi masih positif, kualitas
pengalaman pengguna belum sepenuhnya stabil pada titik-titik proses yang
bersifat kritis.

## 4. Hasil Topic Modeling LDA

Topic modeling dilakukan pada 1.065 ulasan yang diprediksi negatif dan lolos
penyaringan kata kunci terkait keamanan, akses, rekening, atau transaksi.
Model LDA menghasilkan empat topik:

- Topik 0: 222 dokumen atau 20,85% — aktivasi, pembukaan rekening, dan koneksi
- Topik 1: 234 dokumen atau 21,97% — aktivasi pasca-pembaruan dan jaringan
- Topik 2: 331 dokumen atau 31,08% — login, password, dan akses akun
- Topik 3: 278 dokumen atau 26,10% — transaksi, saldo, dan transfer

Proporsi tersebut menunjukkan pembagian topik dominan pada dokumen yang masuk
ke analisis LDA, bukan prevalensi masalah pada seluruh pengguna.

### Topik 0: Aktivasi, pembukaan rekening, dan koneksi

Kata dominan topik ini adalah `aktivasi`, `buka`, `gabisa`, `rekening`,
`koneksi`, `jelek`, `pulsa`, `gagal`, `login`, dan `terputus`. Topik ini
menggambarkan hambatan pada aktivasi, pembukaan rekening, koneksi, dan akses
awal layanan.

### Topik 1: Aktivasi pasca-pembaruan dan gangguan jaringan

Kata dominan pada topik ini adalah `aktivasi`, `update`, `pulsa`, `jaringan`,
`gagal`, `masuk`, `baru`, `rekening`, `kali`, dan `nomor`. Topik ini
menggambarkan keluhan mengenai aktivasi setelah pembaruan, jaringan, dan
kegagalan masuk.

### Topik 2: Login, password, dan hambatan akses akun

Kata dominan pada topik ini adalah `login`, `aktivasi`, `susah`, `jaringan`,
`salah`, `ribet`, `data`, `password`, `masuk`, dan `ulang`. Topik terbesar ini
menggambarkan kesulitan login, password yang dinyatakan salah, serta hambatan
akses akun.

### Topik 3: Transaksi, saldo, dan transfer

Kata dominan pada topik ini adalah `transaksi`, `saldo`, `lama`, `transfer`,
`rekening`, `gagal`, `sering`, `hijau`, `hari`, dan `indikator`. Topik ini
berkaitan dengan transaksi yang lambat atau gagal, saldo, dan transfer.

## 5. Pembahasan Temuan

Jika dilihat secara menyeluruh, hasil penelitian memperlihatkan bahwa ulasan
positif lebih banyak daripada ulasan negatif, tetapi keluhan yang muncul
cenderung terpusat pada beberapa area yang sangat penting bagi mobile banking.
Area tersebut adalah:

- autentikasi dan aktivasi akun
- kestabilan jaringan atau koneksi saat login
- kelancaran transaksi
- validitas data saat pembukaan rekening
- rasa aman terhadap saldo dan proses transfer

Hasil tersebut menunjukkan bahwa keluhan dalam kumpulan ulasan terpilih
berkaitan dengan proses inti layanan. Temuan ini merupakan gambaran pengalaman
dan persepsi yang disampaikan pengguna, bukan bukti adanya masalah keamanan
teknis pada aplikasi.

Temuan ini juga menunjukkan bahwa peningkatan fitur dan antarmuka saja belum
cukup. Pengguna tetap akan memberikan ulasan negatif apabila proses dasar
seperti login, aktivasi, transfer, dan verifikasi identitas belum berjalan
stabil dan konsisten.

## 6. Implikasi Praktis

Berdasarkan hasil sentimen dan topik, beberapa area prioritas yang dapat menjadi
bahan rekomendasi adalah:

- menyederhanakan alur aktivasi ulang dan pemulihan akun
- memperbaiki validasi password, PIN, sidik jari, dan nomor terdaftar
- meningkatkan reliabilitas koneksi aplikasi saat login dan transaksi
- memperbarui data wilayah administratif untuk pembukaan rekening online
- memperkuat penanganan transaksi gagal agar tidak menimbulkan kekhawatiran
  terkait saldo terpotong

## 7. Keterbatasan

Beberapa batasan yang perlu diingat saat menafsirkan hasil:

- hasil inferensi menggunakan dua kelas, sehingga ulasan netral tidak
  ditampilkan sebagai kelas tersendiri
- dokumen LDA merupakan ulasan negatif yang diprediksi model dan lolos
  penyaringan kata kunci; keluhan lain dapat tidak terwakili
- hasil ulasan merepresentasikan persepsi pengguna yang menulis ulasan, bukan
  keseluruhan pengguna atau kondisi teknis aplikasi
- ulasan singkat, sarkastik, atau ambigu tetap berpotensi menimbulkan salah
  interpretasi

## 8. Kesimpulan Singkat

Secara keseluruhan, sentimen positif lebih banyak daripada sentimen negatif
pada 6.396 ulasan yang dianalisis. Evaluasi pada data uji mencatat akurasi
IndoBERT sebesar 92,75%, sedangkan LDA memetakan empat kelompok isu pada 1.065
ulasan negatif terpilih. Temuan tematik terutama berkaitan dengan aktivasi,
login, jaringan, transaksi, saldo, dan transfer. Hasil ini merangkum persepsi
yang muncul dalam ulasan dan tidak dimaksudkan sebagai penilaian keamanan
teknis aplikasi.
