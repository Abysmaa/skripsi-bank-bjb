# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Gambaran Umum Data Penelitian

Data penelitian berasal dari ulasan pengguna aplikasi Digi bank bjb pada
Google Play Store. Hasil pengumpulan dan penggabungan data menghasilkan 6.431
ulasan dengan rentang waktu 13 Juli 2024 sampai 18 April 2026. Setelah
preprocessing, penghapusan data duplikat berdasarkan `review_id`, dan
penghapusan teks kosong, terdapat 6.396 ulasan yang dapat dianalisis.

Distribusi label akhir sebelum klasifikasi biner terdiri atas 2.522 ulasan
negatif, 360 ulasan netral, dan 3.514 ulasan positif. Pada tahap pelatihan
IndoBERT, kelas netral tidak digunakan karena eksperimen akhir dirancang sebagai
klasifikasi biner. Data negatif dan positif kemudian diseimbangkan menggunakan
metode *undersampling* agar evaluasi model tidak didominasi oleh kelas dengan
jumlah data lebih besar.

Tabel 4.1 merangkum data yang digunakan pada setiap tahap penelitian.

| Tahap | Jumlah data |
|---|---:|
| Data mentah | 6.431 ulasan |
| Data setelah preprocessing | 6.396 ulasan |
| Data latih IndoBERT | 3.175 ulasan |
| Data validasi IndoBERT | 794 ulasan |
| Data uji IndoBERT | 993 ulasan |
| Data untuk LDA | 1.065 ulasan |

Pembagian data IndoBERT dilakukan secara terstratifikasi dengan *random seed*
42. Data uji mencakup 20% data hasil penyeimbangan, sedangkan data validasi
mencakup 20% dari sisa data setelah pemisahan data uji.

## 4.2 Hasil Klasifikasi Sentimen Menggunakan IndoBERT

Model yang digunakan adalah `indobenchmark/indobert-base-p1` dengan dua kelas,
yaitu `negative` dan `positive`. Pelatihan dilakukan selama tiga *epoch*. Hasil
evaluasi pada data validasi dan data uji disajikan pada Tabel 4.2.

| Data evaluasi | Akurasi | F1-score tertimbang |
|---|---:|---:|
| Validasi | 93,58% | 93,57% |
| Uji | 92,75% | 92,75% |

Nilai akurasi uji sebesar 92,75% menunjukkan bahwa 921 dari 993 ulasan pada
data uji diklasifikasikan dengan benar. Nilai F1-score tertimbang yang hampir
sama dengan akurasi juga menunjukkan bahwa kinerja model relatif seimbang pada
kedua kelas. Rincian hasil evaluasi per kelas ditampilkan pada Tabel 4.3.

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negatif | 91,75% | 93,96% | 92,84% | 497 |
| Positif | 93,80% | 91,53% | 92,65% | 496 |
| Rata-rata makro | 92,78% | 92,75% | 92,75% | 993 |

Recall kelas negatif sebesar 93,96% berarti model mampu menemukan 467 dari 497
ulasan negatif pada data uji. Kemampuan mendeteksi keluhan ini penting karena
ulasan negatif menjadi masukan utama untuk pemodelan topik terkait keamanan,
akses, dan keandalan layanan.

Confusion matrix hasil pengujian ditampilkan pada Tabel 4.4.

| Label aktual | Prediksi negatif | Prediksi positif |
|---|---:|---:|
| Negatif | 467 | 30 |
| Positif | 42 | 454 |

Sebanyak 30 ulasan negatif salah diprediksi sebagai positif, sedangkan 42
ulasan positif salah diprediksi sebagai negatif. Kesalahan dapat muncul pada
ulasan yang memuat sentimen campuran, misalnya memuji manfaat aplikasi tetapi
sekaligus mengeluhkan pembaruan, pemblokiran, atau kegagalan akses. Secara
keseluruhan, hasil pengujian menunjukkan bahwa IndoBERT memiliki kinerja yang
baik untuk memisahkan sentimen positif dan negatif pada data penelitian.

## 4.3 Distribusi Hasil Prediksi Sentimen

Model IndoBERT kemudian digunakan untuk memprediksi 6.396 ulasan hasil
preprocessing. Distribusi hasil prediksi ditampilkan pada Tabel 4.5.

| Sentimen | Jumlah | Persentase |
|---|---:|---:|
| Positif | 3.479 | 54,39% |
| Negatif | 2.917 | 45,61% |
| **Total** | **6.396** | **100,00%** |

Sentimen positif lebih dominan dengan selisih 562 ulasan atau 8,78 poin
persentase. Temuan ini menunjukkan bahwa lebih banyak pengguna menyampaikan
pengalaman positif, antara lain kemudahan penggunaan dan manfaat aplikasi
dalam mendukung transaksi. Namun, proporsi negatif sebesar 45,61% menunjukkan
bahwa keluhan tetap merupakan bagian yang besar dari keseluruhan data.

Rata-rata tingkat keyakinan prediksi model adalah 0,993. Nilai ini menunjukkan
tingkat keyakinan internal model terhadap kelas yang dipilih, tetapi tidak
dapat diperlakukan sebagai pengganti akurasi. Keandalan klasifikasi tetap
dinilai berdasarkan hasil pada data uji yang terpisah.

Perbandingan label akhir dengan hasil prediksi seluruh data menunjukkan bahwa
model memprediksi benar 2.465 dari 2.522 ulasan berlabel negatif dan 3.372 dari
3.514 ulasan berlabel positif. Pada 360 ulasan berlabel netral yang tidak
menjadi kelas model, 310 dipetakan ke negatif dan 50 dipetakan ke positif.
Pemetaan tersebut merupakan konsekuensi penggunaan klasifikasi biner; hasilnya
tidak berarti bahwa kelas netral secara konseptual tidak ada.

## 4.4 Hasil Topic Modeling Menggunakan LDA

LDA diterapkan untuk menjelaskan isu yang mendasari sentimen negatif. Dokumen
dipilih dari hasil prediksi negatif yang memuat kata kunci terkait keamanan,
akses akun, atau layanan transaksi. Setelah filtering dan persiapan token,
terdapat 1.065 dokumen yang digunakan untuk membentuk empat topik.

Distribusi topik dominan disajikan pada Tabel 4.6.

| Topik | Jumlah dokumen | Persentase |
|---|---:|---:|
| Topik 0 | 222 | 20,85% |
| Topik 1 | 234 | 21,97% |
| Topik 2 | 331 | 31,08% |
| Topik 3 | 278 | 26,10% |
| **Total** | **1.065** | **100,00%** |

Topik 2 memiliki proporsi terbesar, sedangkan Topik 0 memiliki proporsi
terkecil. Walaupun demikian, seluruh topik mempunyai jumlah dokumen yang cukup
besar. Artinya, keluhan tidak terkonsentrasi pada satu masalah saja, tetapi
tersebar pada persoalan aktivasi, akses akun, koneksi, dan transaksi.

## 4.5 Interpretasi Topik

### 4.5.1 Topik 0: Aktivasi, pembukaan rekening, dan koneksi

Kata dominan Topik 0 adalah `aktivasi`, `buka`, `gabisa`, `rekening`,
`koneksi`, `jelek`, `pulsa`, `gagal`, `login`, dan `terputus`. Topik ini
mencakup 222 dokumen atau 20,85% dari dokumen LDA.

Kombinasi kata tersebut menggambarkan hambatan pada akses awal layanan, seperti
aktivasi yang gagal, kesulitan membuka rekening, koneksi terputus, dan pulsa
yang digunakan ketika proses tidak berhasil. Masalah pada tahap awal dapat
membentuk persepsi negatif sebelum pengguna memanfaatkan fitur utama aplikasi.

### 4.5.2 Topik 1: Aktivasi pasca-pembaruan dan gangguan jaringan

Kata dominan Topik 1 adalah `aktivasi`, `update`, `pulsa`, `jaringan`, `gagal`,
`masuk`, `baru`, `rekening`, `kali`, dan `nomor`. Topik ini mencakup 234
dokumen atau 21,97%.

Topik ini menunjukkan keluhan mengenai aktivasi ulang dan kegagalan masuk
setelah pembaruan aplikasi. Kemunculan kata `nomor`, `pulsa`, dan `kali`
mengindikasikan proses verifikasi atau aktivasi yang harus dicoba berulang.
Pembaruan aplikasi dengan migrasi akses yang tidak lancar dapat menurunkan
kepercayaan pengguna terhadap konsistensi layanan.

### 4.5.3 Topik 2: Login, password, dan hambatan akses akun

Kata dominan Topik 2 adalah `login`, `aktivasi`, `susah`, `jaringan`, `salah`,
`ribet`, `data`, `password`, `masuk`, dan `ulang`. Dengan 331 dokumen atau
31,08%, topik ini merupakan topik terbesar.

Topik tersebut menggambarkan kesulitan login, password yang dinyatakan salah,
aktivasi ulang, serta proses akses yang dianggap rumit. Dominasi topik ini
menunjukkan bahwa autentikasi merupakan titik masalah paling sering muncul
dalam kumpulan dokumen LDA. Mekanisme autentikasi memang diperlukan untuk
melindungi akun, tetapi kegagalan berulang dapat membuat pengguna menilai
layanan tidak andal.

### 4.5.4 Topik 3: Transaksi, saldo, dan transfer

Kata dominan Topik 3 adalah `transaksi`, `saldo`, `lama`, `transfer`,
`rekening`, `gagal`, `sering`, `hijau`, `hari`, dan `indikator`. Topik ini
mencakup 278 dokumen atau 26,10%.

Topik ini berkaitan dengan transaksi yang lambat atau gagal, perubahan saldo,
transfer, serta indikator jaringan atau layanan. Dalam mobile banking,
ketidakjelasan status transaksi dan saldo dapat menimbulkan kekhawatiran
terhadap dana pengguna. Oleh karena itu, kecepatan, konsistensi status, dan
kejelasan penyelesaian transaksi menjadi bagian penting dari persepsi
keamanan dan keandalan.

## 4.6 Pembahasan

Hasil klasifikasi memperlihatkan bahwa sentimen positif masih menjadi kelompok
terbesar. Meskipun demikian, perbandingan 54,39% positif dan 45,61% negatif
menunjukkan distribusi yang relatif berdekatan. Dengan demikian, dominasi
sentimen positif tidak dapat diartikan bahwa pengalaman pengguna telah bebas
dari masalah.

Hasil LDA memperjelas isi kelompok negatif. Sebanyak 53,05% dokumen LDA berada
pada Topik 1 dan Topik 2 yang sama-sama berkaitan dengan aktivasi, login,
jaringan, dan akses akun. Sebanyak 26,10% dokumen lainnya berada pada Topik 3
yang berkaitan dengan transaksi, saldo, dan transfer. Pola tersebut menunjukkan
bahwa keluhan terutama muncul pada proses inti yang menentukan apakah pengguna
dapat masuk ke akun dan menyelesaikan transaksi.

Temuan ini juga menunjukkan bahwa persepsi keamanan pengguna tidak hanya
dibentuk oleh keberadaan password atau proses verifikasi. Pengguna menilai
layanan aman ketika akses dapat dilakukan secara konsisten, status transaksi
jelas, dan dana dapat dipantau tanpa ketidakpastian. Oleh sebab itu, aspek
keamanan dalam ulasan beririsan dengan keandalan dan kualitas pengalaman
pengguna.

Hasil penelitian tidak dapat digunakan untuk menyatakan ada atau tidaknya
kerentanan teknis pada aplikasi Digi bank bjb. Data ulasan hanya merekam
persepsi dan pengalaman yang disampaikan pengguna. Pemeriksaan keamanan teknis
tetap memerlukan metode lain, seperti pengujian penetrasi, audit kode, atau
evaluasi kontrol keamanan.

## 4.7 Implikasi Praktis

Berdasarkan frekuensi dan isi topik, prioritas perbaikan dapat diarahkan pada:

1. penyederhanaan aktivasi ulang dan pemulihan akun, terutama setelah
   pembaruan aplikasi;
2. peningkatan kejelasan pesan kesalahan untuk password, nomor terdaftar,
   jaringan, dan verifikasi;
3. pengurangan pengulangan proses yang menggunakan pulsa ketika aktivasi gagal;
4. peningkatan stabilitas login, transfer, dan pembaruan saldo;
5. penyediaan status transaksi gagal dan estimasi penyelesaian dana secara
   jelas; serta
6. penyempurnaan proses pembukaan rekening dan validitas data wilayah.

Dashboard Streamlit yang dikembangkan pada penelitian ini dapat membantu
peneliti menelusuri distribusi sentimen, hasil evaluasi, topik LDA, dan contoh
ulasan secara interaktif. Dashboard berfungsi sebagai media penyajian hasil,
bukan sebagai bagian dari proses evaluasi model.

## 4.8 Keterbatasan Penelitian

Penelitian ini memiliki beberapa keterbatasan:

1. data hanya berasal dari Google Play Store sehingga tidak mewakili seluruh
   pengguna atau kanal pengaduan Bank BJB;
2. label awal banyak dipengaruhi skor ulasan, meskipun sebagian data melalui
   proses koreksi, sehingga potensi bias label tetap ada;
3. model akhir menggunakan klasifikasi biner sehingga ulasan netral dipaksa
   masuk ke kelas negatif atau positif saat inferensi;
4. ulasan yang singkat, ambigu, mengandung ironi, atau memuat sentimen campuran
   masih dapat salah diklasifikasikan;
5. dokumen LDA dipilih menggunakan prediksi negatif dan kata kunci, sehingga
   isu relevan yang tidak memuat kata kunci dapat terlewat;
6. jumlah empat topik ditetapkan pada eksperimen, sedangkan kualitas topik belum
   dibandingkan secara sistematis menggunakan beberapa jumlah topik dan metrik
   koherensi; dan
7. penamaan topik merupakan interpretasi peneliti berdasarkan kata dominan dan
   dokumen anggota, bukan label yang dihasilkan otomatis oleh LDA.

## 4.9 Kesimpulan Bab

Analisis terhadap 6.396 ulasan menghasilkan 3.479 sentimen positif (54,39%) dan
2.917 sentimen negatif (45,61%). IndoBERT memperoleh akurasi uji 92,75% dan
F1-score tertimbang 92,75%, sehingga memiliki kinerja yang baik untuk
klasifikasi biner pada data penelitian. LDA terhadap 1.065 ulasan negatif
terkait keamanan dan akses menghasilkan empat topik: aktivasi dan pembukaan
rekening, aktivasi pasca-pembaruan, login dan password, serta transaksi dan
saldo. Hasil tersebut menunjukkan bahwa persepsi negatif pengguna terutama
berkaitan dengan keandalan proses akses akun dan transaksi, bukan menjadi bukti
langsung mengenai keamanan teknis aplikasi.
