# Ringkasan Hasil dan Pembahasan

Dokumen ini merangkum hasil utama analisis sentimen dan topic modeling pada
ulasan aplikasi Digi bank bjb berdasarkan artefak terbaru project.

## 1. Gambaran Hasil Utama

Jumlah ulasan yang berhasil diprediksi adalah 1.489 review. Dari seluruh review
tersebut, 966 review atau 64,88% diprediksi sebagai sentimen positif, sedangkan
523 review atau 35,12% diprediksi sebagai sentimen negatif.

Secara umum, hasil ini menunjukkan bahwa persepsi pengguna terhadap aplikasi
masih didominasi sentimen positif. Namun demikian, proporsi sentimen negatif
yang mencapai lebih dari sepertiga total ulasan tetap cukup besar dan tidak bisa
diabaikan, terutama karena keluhan negatif banyak berkaitan dengan aspek
aktivasi, login, transaksi, jaringan, dan pembukaan rekening.

## 2. Kinerja Model IndoBERT

Model IndoBERT dilatih menggunakan dataset final hasil penggabungan semi-label
dan anotasi manual. Untuk tahap klasifikasi, eksperimen dijalankan dalam skema
biner, yaitu label `negative` dan `positive`.

Berdasarkan file `models/indobert/training_summary.json`, model menghasilkan:

- Akurasi data uji: 0,9487
- F1-score tertimbang data uji: 0,9487
- Precision kelas negatif: 0,9400
- Recall kelas negatif: 0,9592
- Precision kelas positif: 0,9579
- Recall kelas positif: 0,9381

Hasil tersebut menunjukkan bahwa model memiliki performa yang tinggi dan relatif
seimbang pada kedua kelas. Recall kelas negatif sedikit lebih tinggi daripada
kelas positif. Hal ini berarti model cenderung cukup baik dalam menangkap review
bernuansa keluhan, yang penting karena tujuan penelitian juga berkaitan dengan
identifikasi masalah yang dirasakan pengguna.

Confusion matrix pada data uji menunjukkan:

- 94 review negatif terklasifikasi benar sebagai negatif
- 91 review positif terklasifikasi benar sebagai positif
- 4 review negatif salah diprediksi sebagai positif
- 6 review positif salah diprediksi sebagai negatif

Jumlah salah klasifikasi yang relatif kecil ini memperkuat bahwa model cukup
andal untuk digunakan sebagai alat bantu analisis sentimen dalam penelitian.

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

Topic modeling dilakukan pada 224 dokumen yang berfokus pada review negatif dan
relevan dengan aspek keamanan atau risiko penggunaan. Model LDA menghasilkan 4
topik utama dengan sebaran yang relatif merata:

- Topik 0: 65 dokumen atau 29,02%
- Topik 1: 61 dokumen atau 27,23%
- Topik 2: 43 dokumen atau 19,20%
- Topik 3: 55 dokumen atau 24,55%

Berdasarkan kata dominan dan contoh dokumen, interpretasi tiap topik adalah
sebagai berikut.

### Topik 0: Masalah aktivasi ulang, kesalahan kredensial, dan gangguan jaringan

Kata yang paling sering muncul pada topik ini adalah `aktivasi`, `jaring`,
`salah`, `update`, `susah`, `gagal`, dan `ulang`. Topik ini menggambarkan
keluhan pengguna yang mengalami hambatan saat masuk ke aplikasi, khususnya
setelah update, ketika password dianggap salah, atau saat harus melakukan
aktivasi ulang tetapi sistem tidak merespons dengan baik.

Topik ini menunjukkan bahwa aspek keamanan yang seharusnya melindungi akses
pengguna justru dalam beberapa kasus dirasakan sebagai penghambat karena proses
verifikasi dan aktivasi dianggap terlalu rentan gagal.

### Topik 1: Ketidakstabilan aplikasi saat transaksi dan penggunaan harian

Kata dominan pada topik ini antara lain `aplikasi`, `transaksi`, `bank`, `bjb`,
`digi`, `rekening`, `transfer`, dan `hari`. Topik ini menggambarkan keluhan
terhadap performa aplikasi saat digunakan untuk transaksi, termasuk transfer,
akses rekening, atau penggunaan rutin yang sering terganggu error.

Makna penting dari topik ini adalah bahwa persepsi keamanan pengguna tidak hanya
terbentuk oleh perlindungan akun, tetapi juga oleh kestabilan sistem transaksi.
Ketika transaksi sering gagal atau terhambat, pengguna dapat memandang aplikasi
sebagai tidak andal dan berisiko.

### Topik 2: Hambatan pembukaan rekening, aktivasi, dan masalah saldo atau dana

Kata dominan pada topik ini meliputi `aktivasi`, `rekening`, `gagal`, `buka`,
`harus`, `camat`, `login`, dan `uang`. Topik ini memperlihatkan adanya masalah
yang berkaitan dengan pembukaan rekening, data wilayah yang tidak tersedia,
aktivasi akun baru, serta kasus saldo terpotong atau uang tidak kembali setelah
transaksi gagal.

Topik ini penting karena menyentuh dua isu sensitif sekaligus, yaitu akses awal
ke layanan dan rasa aman terhadap dana nasabah. Dalam konteks layanan keuangan,
keluhan seperti saldo terpotong atau transaksi gagal berpotensi menurunkan
kepercayaan secara signifikan.

### Topik 3: Login, password, dan gangguan akses pasca-pembaruan

Kata dominan pada topik ini adalah `aplikasi`, `salah`, `login`, `jaring`,
`aktivasi`, `update`, `password`, `nomor`, dan `tolong`. Topik ini sangat dekat
dengan pengalaman pengguna yang tidak bisa masuk ke aplikasi karena password
dianggap salah, nomor tidak terbaca, atau akses menjadi bermasalah setelah
pembaruan sistem.

Topik ini menunjukkan bahwa proses pembaruan aplikasi merupakan titik kritis.
Jika setelah update pengguna justru gagal login atau harus melakukan aktivasi
ulang dengan prosedur yang rumit, maka persepsi terhadap keamanan dan
keandalan aplikasi akan ikut menurun.

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

Artinya, masalah utama bukan terletak pada seluruh fitur aplikasi, melainkan
pada proses yang paling menentukan kepercayaan pengguna. Dalam konteks mobile
banking, gangguan kecil pada proses login, verifikasi, atau transaksi bisa
langsung dianggap sebagai masalah keamanan maupun kualitas layanan.

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

Beberapa keterbatasan yang perlu dicatat:

- model klasifikasi akhir dijalankan dalam skema biner, sehingga kelas neutral
  tidak menjadi fokus utama pada inferensi akhir
- topic modeling sangat dipengaruhi oleh hasil preprocessing dan kata yang
  tersisa setelah normalisasi
- ulasan yang sangat singkat, sarkastik, atau ambigu tetap berpotensi
  menimbulkan salah interpretasi

## 8. Kesimpulan Singkat

Secara keseluruhan, aplikasi Digi bank bjb memperoleh sentimen positif yang
lebih dominan, tetapi masih memiliki titik lemah yang konsisten pada aspek
aktivasi, login, jaringan, transaksi, dan pembukaan rekening. Model IndoBERT
menunjukkan performa yang baik untuk memetakan sentimen pengguna, sedangkan LDA
berhasil mengekstraksi isu utama yang relevan dengan keamanan dan keandalan
layanan. Dengan demikian, kombinasi IndoBERT dan LDA efektif digunakan untuk
memberikan gambaran kuantitatif sekaligus tematik terhadap pengalaman pengguna
mobile banking.
