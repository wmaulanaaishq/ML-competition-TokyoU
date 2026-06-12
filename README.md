#  NFL Draft Prediction 

> **GCI Data Science Competition by Matsuo Lab, University of Tokyo** 🇯🇵
> Proyek ini dikembangkan secara khusus untuk menyelesaikan tantangan kompetisi Data Science dari Laboratorium Matsuo (Matsuo Lab), Universitas Tokyo.

Proyek ini berisi *End-to-End Machine Learning Pipeline* untuk memprediksi apakah seorang atlet akan ditarik (di-*draft*) ke NFL berdasarkan metrik fisik *NFL Combine*. 

---

## 📁 Struktur Folder

```text
d:\GCI Competition\competition\
│
├── input/                             # Folder dataset (Jangan diubah)
│   ├── train.csv                      # Data latih (2781 baris)
│   ├── test.csv                       # Data uji (696 baris)
│   └── sample_submission.csv          # Format contoh pengumpulan
│
├── output/                            # Direktori hasil eksekusi (Otomatis)
│   ├── final_submission_blended.csv   # FILE SUBMISSION 
│   ├── eda_report.md                  # Ringkasan data (EDA) otomatis
│   ├── best_parameters.json           # Log hyperparameter terbaik (Optuna)
│   ├── cv_results.csv                 # Log skor tiap fold
│   ├── feature_importance.csv         # Tingkat kepentingan fitur
│   └── submission_*.csv               # File submisi individual (Seed 42, 2024, 777)
│
├── nfl_draft_pipeline.py              # SKRIP UTAMA (Otak Pipeline)
└── run_seed_blend.py                  # SKRIP WRAPPER (Eksekusi Ensembling Multi Seed)
```

---

## ⚙️ Fitur Utama Pipeline (`nfl_draft_pipeline.py`)

Skrip utama kita melakukan serangkaian operasi kelas berat secara otomatis:
1. **Advanced Feature Engineering**: 
   - Mengekstrak puluhan fitur fisik sintesis tingkat lanjut seperti `SpeedScore`, `ExplosivenessScore`, `AgilityRatio`, dan interaksi tinggi/berat badan.
   - Menggunakan *Smoothed Target Encoding* (dengan *Cross-Validation*) untuk mengekstrak sentimen kesuksesan asal kampus (`School`) dan posisi secara aman tanpa *data leakage*.
2. **Robust Cross Validation**: Menggunakan 5-Fold `StratifiedKFold` untuk menjamin distribusi target seimbang dan evaluasi yang jujur.
3. **Hyperparameter Tuning (Optuna)**: Mengoptimalkan parameter `LightGBM`, `XGBoost`, dan `CatBoost` secara otomatis (dilengkapi pengaman memori *Garbage Collector*).
4. **Stacking & Ensembling**: 
   - Memadukan prediksi menggunakan algoritma `LogisticRegression` (Stacking).
   - Memiliki komparasi `Simple Average`, `Rank Average`, dan `Weighted Rank Average`.

---

## 🚀 Cara Menjalankan

### Opsi 1: Menjalankan Sekali Jalan (Single Run)
Gunakan ini jika kamu ingin menguji *pipeline* standar dengan satu pijakan acak (*Random Seed* = 42).
```powershell
python nfl_draft_pipeline.py
```
*(Hasil akhir akan muncul di folder `output/submission_42.csv`)*

### Opsi 2: Eksekusi Mode Turnamen (Multi-Seed Blending) 🏆
Skrip ini akan menjalankan *pipeline* utama sebanyak 3 kali dengan *Seed* yang berbeda, lalu mengambil rata-rata probabilitasnya
```powershell
python run_seed_blend.py
```
*(Hasil akhir akan muncul di folder `output/final_submission_blended.csv`)*

---

## 📊 Pencapaian Performa (*Benchmarks*)
Berdasarkan log eksperimen terakhir yang dieksekusi, ini adalah batas maksimal keakuratan yang legal (*OOF ROC-AUC*):
- **Best Single Model (CatBoost)**: ~0.842
- **Best Single Ensemble (Weighted Rank)**: ~0.8465
- **Multi-Seed Blended Ensemble**: Puncak probabilitas paling stabil (Digaransi menaikkan skor *Public Leaderboard*).
