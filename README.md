<div align="center">

# 🔥 Sistem Deteksi Dini Kebakaran Hutan
### Provinsi Riau · 2015–2025

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-MODIS%2F061-4285F4?style=flat-square&logo=google&logoColor=white)](https://earthengine.google.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

Sistem analisis multi-variabel berbasis **Google Earth Engine + Machine Learning** untuk deteksi dini dan estimasi risiko kebakaran hutan di Provinsi Riau, Indonesia.

[Mulai Cepat](#-mulai-cepat) · [Arsitektur Model](#-model-machine-learning) · [Dataset](#-dataset--variabel) · [Dashboard](#-streamlit-dashboard)

</div>

---

## 📌 Tentang Proyek

Proyek ini membangun pipeline end-to-end untuk:

- **Mengekspor** data kebakaran historis 2015–2025 dari Google Earth Engine (MODIS Collection 6.1)
- **Melatih** beberapa model Machine Learning (Random Forest, Gradient Boosting, LSTM) untuk klasifikasi risiko dan estimasi area terbakar
- **Memvisualisasikan** hasil analisis, feature importance (SHAP), peta hotspot, dan estimasi bulanan 2026 melalui dashboard Streamlit interaktif

> **Periode kritis:** Juli–Oktober (musim kemarau). Tahun El Niño (2015, 2019, 2023) menunjukkan lonjakan kebakaran **1,5–3× lipat** dibanding tahun normal.

---

## 📁 Struktur Proyek

```
kebakaran-hutan/
│
├── 1_export_gee_modis061.py          # Export data GEE MODIS/061 → CSV lokal
├── 2_Colab_ML_fire_prediction.py     # Pipeline ML: preprocessing, training, estimasi
├── 3_streamlit_dashboard.py          # Dashboard interaktif Streamlit
│
├── riau_fire_monthly_stats_modis061.csv   # ← Output Step 1 (GEE export)
├── riau_fire_processed.csv                # ← Output Step 2 (ML pipeline)
├── riau_fire_estimasi_2026.csv            # ← Estimasi bulanan 2026
│
├── best_risk_model_riau.pkl          # Model klasifikasi risiko terbaik
├── area_regressor_fallback.pkl       # Model regresi area terbakar
├── requirements.txt
└── README.md
```

---

## 🚀 Mulai Cepat

### Prasyarat

- Python **3.11** (TensorFlow belum mendukung Python 3.12+)
- Akun [Google Earth Engine](https://earthengine.google.com) (gratis non-komersial)

```bash
# Clone / buka folder proyek
cd kebakaran-hutan

# Install dependensi
pip install -r requirements.txt

# Autentikasi GEE (sekali saja, membuka browser)
earthengine authenticate
```

---

## 🔄 Pipeline 3 Langkah

```
┌──────────────────┐     CSV      ┌──────────────────┐    PKL/CSV   ┌──────────────────┐
│  Step 1: GEE     │ ──────────► │  Step 2: ML      │ ──────────► │  Step 3: Dash    │
│  Export MODIS    │             │  Training        │             │  Streamlit       │
│  Collection 6.1  │             │  & Estimasi      │             │  Dashboard       │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

### Step 1 — Export GEE (MODIS Collection 6.1)

```bash
python 1_export_gee_modis061.py
```

Script ini mengambil data langsung dari GEE untuk 132 bulan (2015–2025) dan menyimpannya sebagai `riau_fire_monthly_stats_modis061.csv` dengan metadata verifikasi `modis_collection_version=MODIS/061`.

> **Estimasi waktu:** 5–15 menit (120 GEE API calls). Memerlukan koneksi internet.

<details>
<summary>Koleksi GEE yang digunakan</summary>

| Koleksi | Variabel | Resolusi |
|---------|----------|----------|
| `MODIS/061/MCD64A1` | Area terbakar (ha) | 500 m |
| `MODIS/061/MOD11A2` | LST siang hari (°C) | 1 km |
| `MODIS/061/MOD13A2` | NDVI | 1 km |
| `UCSB-CHG/CHIRPS/DAILY` | Curah hujan total (mm) | ~5 km |
| `ECMWF/ERA5_LAND/MONTHLY_AGGR` | Kecepatan angin 10m (m/s) | ~11 km |

</details>

---

### Step 2 — Training Model ML

```bash
python 2_Colab_ML_fire_prediction.py
```

Pipeline ini melakukan:
1. **Audit & imputasi** data sensor (LST/NDVI bernilai 0 atau null dianggap suspect)
2. **Feature engineering** — lag features, rolling mean, trigger score, dll.
3. **Perbandingan model** dengan TimeSeriesSplit cross-validation
4. **Training model terbaik** + SHAP / permutation importance
5. **Estimasi bulanan 2026** menggunakan klimatologi historis

Output: model `.pkl`, `riau_fire_processed.csv`, `riau_fire_estimasi_2026.csv`

> **Catatan:** Jika `riau_fire_monthly_stats_modis061.csv` tidak ditemukan, script otomatis fallback ke data sintetis berbasis pola iklim Riau.

---

### Step 3 — Jalankan Dashboard

```bash
streamlit run 3_streamlit_dashboard.py
```

Buka browser: **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Dataset & Variabel

| Variabel | Sumber | Keterangan |
|----------|--------|------------|
| 🔥 Area Terbakar | `MODIS/061/MCD64A1` | Burned area bulanan (ha) |
| 🌡️ LST Siang | `MODIS/061/MOD11A2` | Land Surface Temperature (°C) |
| 🌿 NDVI | `MODIS/061/MOD13A2` | Indeks vegetasi hijau |
| 🌧️ Curah Hujan | `UCSB-CHG/CHIRPS/DAILY` | Total hujan bulanan (mm) |
| 💨 Kecepatan Angin | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | Angin 10m (m/s) |
| 🔥 Fire Risk Index | Dihitung | Kombinasi weighted LST, NDVI, rain, wind |

### Fire Risk Index (FRI)

$$\text{FRI} = 0.35 \cdot \frac{\text{LST}-20}{20} + 0.35 \cdot (1-\text{NDVI}) + 0.20 \cdot \left(1-\frac{\text{rain}}{300}\right) + 0.10 \cdot \frac{\text{wind}}{10}$$

| Kategori | FRI | Warna |
|----------|-----|-------|
| Rendah | < 0.30 | 🟢 |
| Sedang | 0.30 – 0.50 | 🟡 |
| Tinggi | 0.50 – 0.70 | 🟠 |
| Sangat Tinggi | ≥ 0.70 | 🔴 |

---

## 🤖 Model Machine Learning

### Klasifikasi Risiko

| Model | Keterangan |
|-------|------------|
| **Random Forest** | Model utama, balanced class weights, GridSearchCV |
| **Gradient Boosting** | Alternatif, robust terhadap outlier |
| **Hist Gradient Boosting** | Cepat, mendukung missing values native |
| **Extra Trees** | Variance rendah |
| **SVM** | Baseline non-linear |
| **Logistic Regression** | Baseline linear |

- **Split:** TimeSeriesSplit (80/20, tidak random) — menghormati urutan waktu
- **Feature set:** 25 variabel (raw + lag-1, lag-3, rolling-3, rolling-6, trigger flags)
- **Metric utama:** Balanced Accuracy + F1 Macro

### Estimasi Area (Regresi)

| Model | Keterangan |
|-------|------------|
| **Random Forest Regressor** | Model utama area terbakar |
| **Bidirectional LSTM** *(opsional)* | Sequence 6 bulan → BiLSTM(64) → LSTM(32) → Dense, Huber loss |

> **LSTM memerlukan Python ≤ 3.11 dan TensorFlow ≥ 2.13.** Jika tidak tersedia, pipeline otomatis fallback ke RF Regressor.

---

## ⚡ Pemicu Kebakaran

| Sinyal | Threshold | Bobot |
|--------|-----------|-------|
| 🌧️ Kekeringan | Curah hujan < 100 mm/bulan | ★★★ |
| 🌡️ Panas ekstrem | LST > 38 °C | ★★★ |
| 🌿 Vegetasi kering | NDVI < 0.4 | ★★ |
| 💨 Angin kencang | Kecepatan > 4 m/s | ★ |

**Trigger Score** = jumlah kondisi terpenuhi (0–4). Skor ≥ 3 = risiko sangat tinggi.

---

## 🗺️ Kabupaten Prioritas

Kabupaten dengan frekuensi hotspot tertinggi di Riau:

| # | Kabupaten | Karakteristik |
|---|-----------|---------------|
| 1 | **Pelalawan** | Lahan gambut dominan |
| 2 | **Bengkalis** | Pulau, ekosistem gambut |
| 3 | **Siak** | Perkebunan + gambut |
| 4 | **Kepulauan Meranti** | Gambut dalam |
| 5 | **Rokan Hilir** | Perbatasan Sumatera Utara |
| 6 | **Indragiri Hulu** | Kawasan hutan |
| 7 | **Kuantan Singingi** | Hutan dataran tinggi |

---

## 🖥️ Streamlit Dashboard

Menu yang tersedia di dashboard:

| Menu | Isi |
|------|-----|
| **Overview** | Metrik utama, tren historis, distribusi risiko tahunan |
| **Peta** | Peta hotspot interaktif per bulan (historis & estimasi 2026) |
| **Analisis Variabel** | Scatter, korelasi, SHAP/permutation importance |
| **Model & Kualitas Data** | Perbandingan model, confusion matrix, audit sensor |
| **Estimasi 2026** | Proyeksi 12 bulan, probabilitas per kategori risiko |
| **Tabel Data** | Export CSV historis, estimasi, dan feature importance |

---

## ⚠️ Catatan Penting

> [!IMPORTANT]
> Selalu gunakan `riau_fire_monthly_stats_modis061.csv` (output Step 1) sebagai input. Data tanpa kolom `modis_collection_version=MODIS/061` akan ditandai **belum terverifikasi** di dashboard.

> [!WARNING]
> **Python 3.14** tidak didukung TensorFlow. Gunakan Python **3.11** jika membutuhkan modul LSTM. Semua fitur lain (Random Forest, dashboard, GEE export) berjalan normal di Python 3.14.

> [!NOTE]
> Estimasi 2026 adalah proyeksi berbasis tren historis dan klimatologi, **bukan** prediksi cuaca real-time. Untuk sistem produksi, integrasikan dengan data **SIPONGI KLHK** dan **BMKG**.

---

## 📦 Requirements

```
earthengine-api>=0.1.370
geemap>=0.28.0
# tensorflow>=2.13.0  # Python <=3.11 only
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
streamlit>=1.28.0
streamlit-folium>=0.15.0
folium>=0.14.0
seaborn>=0.12.0
matplotlib>=3.7.0
joblib>=1.3.0
shap>=0.45.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<div align="center">

Dibuat untuk keperluan **penelitian remote sensing & kebakaran hutan** di Provinsi Riau, Indonesia.

*Data MODIS © NASA · CHIRPS © UCSB-CHG · ERA5 © ECMWF via Google Earth Engine*

</div>
