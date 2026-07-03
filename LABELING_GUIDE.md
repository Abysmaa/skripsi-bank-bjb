# Panduan Labeling Sentimen

File ini digunakan sebagai acuan saat mengisi kolom `manual_label` pada:

- `data/processed/annotation_batches/batch_001.csv`

Tujuannya agar label manual konsisten dengan pipeline baru proyek skripsi.

## Label yang valid

Gunakan hanya tiga label berikut:

- `negatif`
- `netral`
- `positif`

Jangan gunakan label lama seperti:

- `negative`
- `neutral`
- `positive`

## Kolom yang perlu diisi

Pada file batch anotasi, kolom yang perlu diisi manual hanya:

- `manual_label`

Kolom lain seperti `review_id`, `clean_content`, `score`, dan `semi_label`
dipakai sebagai referensi.

## Aturan utama penentuan label

### `negatif`

Gunakan `negatif` jika isi review didominasi keluhan, hambatan, kegagalan, atau
kritik terhadap aplikasi.

Contoh indikasi:

- gagal login
- aktivasi gagal
- OTP tidak masuk
- password atau PIN salah
- aplikasi error, lemot, loading terus, atau force close
- transaksi gagal
- dana tidak masuk atau saldo bermasalah
- verifikasi gagal
- akun terblokir
- tidak bisa masuk aplikasi

Contoh dari batch:

- `log out aja` → `negatif`
- `bjb digi lagi gangguan kah min ga bisa transaksi` → `negatif`
- `sering error` → `negatif`

### `positif`

Gunakan `positif` jika review berisi kepuasan, pujian, kemudahan penggunaan,
atau pengalaman baik yang jelas.

Contoh indikasi:

- sangat membantu
- transaksi lancar
- aplikasi bagus
- mudah digunakan
- fiturnya lengkap
- update terbaru memudahkan

Contoh dari batch:

- `sangat membantu` → `positif`
- `update terbarunya sangat memudahkan dalam segala transaksi` → `positif`
- `lancar` → `positif`

### `netral`

Gunakan `netral` jika isi review ambigu, informatif, berupa pertanyaan, atau
campuran positif dan negatif tetapi tidak ada dominasi yang jelas.

Contoh indikasi:

- pertanyaan tanpa emosi yang jelas
- komentar terlalu singkat dan konteksnya kurang
- saran biasa tanpa pujian atau keluhan dominan
- ada plus dan minus yang seimbang

Contoh dari batch:

- `fitur udh lengkap cuman tampilannya kurang bagus` → cenderung `netral`
- `tidak ada notifikasi dana masuk atau dana keluar ya` → cenderung `netral`

## Aturan prioritas

Kalau satu review mengandung unsur positif dan negatif sekaligus, gunakan aturan
berikut:

- pilih `negatif` jika inti review adalah keluhan
- pilih `positif` jika inti review adalah kepuasan
- pilih `netral` jika dua sisi terlihat seimbang atau maksud review kurang jelas

## Cara membaca `semi_label`

Kolom `semi_label` adalah label otomatis dari skor bintang. Kolom ini hanya
sebagai label awal, bukan kebenaran mutlak.

Contoh:

- skor `3` otomatis menjadi `netral`, tetapi kalau isi review jelas mengeluh
  seperti `sering error`, maka `manual_label` sebaiknya diisi `negatif`
- skor `5` otomatis menjadi `positif`, tetapi kalau isi review ternyata sarkastik
  atau mengandung keluhan nyata, peneliti boleh mengubahnya

## Langkah kerja yang disarankan

1. Buka file `data/processed/annotation_batches/batch_001.csv`
2. Isi kolom `manual_label`
3. Gunakan hanya `negatif`, `netral`, atau `positif`
4. Simpan file setelah selesai atau secara berkala
5. Cek progres dengan:

```powershell
.\.venv\Scripts\python.exe src\check_annotation_progress.py
```

6. Setelah anotasi cukup, finalisasi ulang label:

```powershell
.\.venv\Scripts\python.exe src\finalize_labels.py --batch-size 100
```

## Saran praktis anotasi

- fokus dulu ke review dengan `score = 3` karena paling ambigu
- jika teks sangat singkat tetapi jelas bernada keluhan, pilih `negatif`
- jika teks sangat singkat tetapi jelas bernada pujian, pilih `positif`
- jika ragu, baca konteks kata kerja utamanya: `gagal`, `error`, `tidak bisa`,
  `membantu`, `lancar`, `bagus`

## Checklist sebelum finalisasi

- semua label yang diisi hanya `negatif`, `netral`, atau `positif`
- tidak ada typo seperti `positiv`, `negatif `, atau `Neutral`
- file batch sudah disimpan
- progres anotasi sudah dicek
