# Iris Data Classification

Proyek ini adalah implementasi dari alur kerja Machine Learning dasar dalam Python untuk mengklasifikasikan dataset spesies bunga Iris. Alur pembelajaran mencakup pre-processing data, ekplorasi dataset, visualisasi, pembuatan model klasifikasi menggunakan beberapa algoritma pendekatan (Naive Bayes, Decision Tree, Support Vector Machine/SVM), sampai dengan evaluasi model serta *cross-validation*.

## Persyaratan Awal
- **Python 3.x** terinstal pada sistem kamu.

## Cara Menjalankan Proyek 🚀

**1. Clone Repositori Ini (Opsional bila dijalankan di lokal)**
Bila di sistem lain, kamu bisa download repositori ini terlebih dahulu:
```bash
git clone https://github.com/radishguyy/ty-data-model.git
cd ty-data-model
```

**2. Buat Virtual Environment (*Disarankan*)**
Ini berguna agar dependensi proyek ini terisolasi dan tidak merusak sistem / project Python lokalmu yang lain.
```bash
python3 -m venv venv
```

**3. Aktivasi Virtual Environment**
- Khusus di sistem **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```
- Khusus di sistem **Windows**:
  ```bash
  venv\Scripts\activate
  ```

**4. Instal Dependensi yang Dibutuhkan**
Setelah masuk ke *virtual environment*, silakan install library yang dipakai dalam proses visualisasi maupun machine learning:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

**5. Jalankan Machine Learning Pipeline-nya**
Mulai skrip `model.py` untuk menjalankan pipeline end-to-end:
```bash
python model.py
```
> 💡 *Catatan:* Nanti jendela *pop-up* akan terbuka untuk menampilkan beberapa grafik / visualisasi (seperti Decision Tree & plot grafik data). Kamu hanya perlu meng-X / menutup plot atau gambar satu-satu untuk membiarkan script model lanjut mengeksekusi sisa fiturnya hingga tuntas.
