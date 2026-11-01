# ============================================================
# DETEKSI, PERBANDINGAN MODEL, DAN ESTIMASI KEBAKARAN HUTAN
# Provinsi Riau, historis 2015-2025 dan estimasi bulanan 2026
# ============================================================

# %% [markdown]
# # Analisis & Deteksi Dini Kebakaran Hutan Riau
#
# Notebook/script ini melakukan:
# 1. Load data GEE export dari Colab atau folder lokal.
# 2. Audit kualitas data, terutama anomali 2023-2025.
# 3. Feature engineering untuk data bulanan.
# 4. Membandingkan beberapa model machine learning dengan hyperparameter.
# 5. Memilih model klasifikasi risiko terbaik berdasarkan holdout time-series.
# 6. Membuat feature importance memakai SHAP jika tersedia, fallback ke permutation importance.
# 7. Melatih model area kebakaran dan membuat estimasi setiap bulan tahun 2026.
# 8. Mengekspor CSV, HTML, gambar, dan model untuk Streamlit.

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    TENSORFLOW_AVAILABLE = False

import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

RANDOM_STATE = 42
HIST_START_YEAR = 2015
HIST_END_YEAR = 2025
FORECAST_YEAR = 2026
ROOT_DIR = Path.cwd()
OUTPUT_DIR = Path("/content") if Path("/content").exists() else ROOT_DIR
MODIS_COLLECTION_VERSION = "MODIS/061"
GEE_COLLECTIONS = [
    "MODIS/061/MCD64A1",
    "MODIS/061/MOD14A1",
    "MODIS/061/MYD14A1",
    "MODIS/061/MOD11A2",
    "MODIS/061/MOD13A2",
    "UCSB-CHG/CHIRPS/DAILY",
    "ECMWF/ERA5_LAND/MONTHLY_AGGR",
]
MODIS061_RAW_FILENAMES = [
    "riau_fire_monthly_stats_modis061.csv",
    "riau_fire_monthly_stats_MODIS061.csv",
    "riau_fire_monthly_stats_collection061.csv",
]

MONTH_NAMES_SHORT = [
    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
    "Jul", "Ags", "Sep", "Okt", "Nov", "Des",
]
MONTH_NAMES_FULL = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]
RISK_LABELS = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
RISK_MAP = dict(enumerate(RISK_LABELS))
RISK_COLOR = {
    0: "#2ecc71",
    1: "#f39c12",
    2: "#e67e22",
    3: "#e74c3c",
}


def out_path(filename):
    return OUTPUT_DIR / filename


print(f"Output directory: {OUTPUT_DIR}")
print(f"MODIS collection version: {MODIS_COLLECTION_VERSION}")
print(f"SHAP tersedia: {SHAP_AVAILABLE}")
print(f"TensorFlow tersedia: {TENSORFLOW_AVAILABLE}")


# %% [markdown]
# ## 1. Load Data

# %%
def compute_fire_risk_index(lst, ndvi, rain, wind):
    """Hitung ulang FRI dari komponen cuaca/vegetasi yang sudah dibersihkan."""
    return (
        np.clip((lst - 20) / 20, 0, 1) * 0.35
        + np.clip(1 - ndvi, 0, 1) * 0.35
        + np.clip(1 - rain / 300, 0, 1) * 0.20
        + np.clip(wind / 10, 0, 1) * 0.10
    )


def categorize_risk(fri):
    if fri < 0.30:
        return 0
    if fri < 0.50:
        return 1
    if fri < 0.70:
        return 2
    return 3


def make_synthetic_data():
    """Fallback jika file GEE export belum tersedia."""
    np.random.seed(RANDOM_STATE)
    dates = pd.date_range(
        start=f"{HIST_START_YEAR}-01-01",
        end=f"{HIST_END_YEAR}-12-01",
        freq="MS",
    )
    seasonal_rain = np.array([280, 250, 230, 210, 190, 130, 90, 70, 80, 160, 220, 270])
    elnino_factor = {2015: 1.8, 2019: 1.6, 2023: 1.3}
    records = []

    for d in dates:
        y, m = d.year, d.month
        elnino = elnino_factor.get(y, 1.0)
        rain = max(10, seasonal_rain[m - 1] / elnino + np.random.normal(0, 25))
        lst = 28 + 8 * np.sin((m - 2) * np.pi / 6) + (elnino - 1) * 3
        lst = lst + np.random.normal(0, 0.8)
        ndvi = 0.65 - 0.25 * np.clip(1 - rain / 200, 0, 1) + np.random.normal(0, 0.03)
        ndvi = float(np.clip(ndvi, -0.1, 0.9))
        wind = max(0.5, 2.5 + 1.5 * (1 - rain / 300) + np.random.normal(0, 0.3))
        fri = float(np.clip(compute_fire_risk_index(lst, ndvi, rain, wind), 0, 1))

        fire_area = 0.0
        fire_prob = np.clip((fri - 0.3) / 0.5, 0, 1) * elnino
        if fri > 0.45 and np.random.random() < fire_prob:
            fire_area = np.random.lognormal(np.log(max(500, fri * 80000)), 0.8)
            if y == 2015 and m in [9, 10]:
                fire_area *= 3

        records.append(
            {
                "year": y,
                "month": m,
                "date_label": f"{y}-{m:02d}",
                "date": d,
                "fire_area_ha": round(fire_area, 1),
                "mean_lst_celsius": round(lst, 2),
                "total_rain_mm": round(rain, 1),
                "mean_ndvi": round(ndvi, 4),
                "mean_wind_ms": round(wind, 2),
                "fire_risk_index": round(fri, 4),
                "has_fire_event": int(fire_area > 100),
            }
        )

    synthetic = pd.DataFrame(records)
    synthetic.to_csv(out_path("riau_fire_synthetic.csv"), index=False)
    return synthetic, "synthetic"


def load_monthly_data():
    candidates = [
        Path("/content/riau_fire_monthly_stats_modis061.csv"),
        ROOT_DIR / "riau_fire_monthly_stats_modis061.csv",
        Path("/content/drive/MyDrive/riau_fire_monthly_stats_modis061.csv"),
        Path("/content/riau_fire_monthly_stats.csv"),
        ROOT_DIR / "riau_fire_monthly_stats.csv",
        Path("/content/drive/MyDrive/riau_fire_monthly_stats.csv"),
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)

    print("CSV tidak ditemukan. Fallback ke data sintetis.")
    return make_synthetic_data()


def infer_modis_collection_source(raw_df, source_path):
    """Tentukan apakah CSV benar-benar berasal dari MODIS/061 atau belum terverifikasi."""
    metadata_cols = [
        "modis_collection_version",
        "modis_collection",
        "source_modis_collection",
        "gee_modis_collection",
    ]
    for col in metadata_cols:
        if col in raw_df.columns and raw_df[col].notna().any():
            value = str(raw_df[col].dropna().iloc[0])
            return value, "061" in value

    source_name = str(source_path).replace("\\", "/").lower()
    if any(name.lower() in source_name for name in MODIS061_RAW_FILENAMES) or "modis061" in source_name or "061" in source_name:
        return "MODIS/061 (terdeteksi dari nama file)", True

    return (
        "Belum terverifikasi - CSV tidak memuat metadata MODIS/061",
        False,
    )


df_raw, data_source = load_monthly_data()
source_modis_collection, source_modis061_verified = infer_modis_collection_source(df_raw, data_source)
print(f"Data source: {data_source}")
print(f"Raw shape: {df_raw.shape}")
print(f"Source MODIS collection: {source_modis_collection}")
if not source_modis061_verified and data_source != "synthetic":
    print(
        "PERINGATAN: CSV belum bisa dipastikan berasal dari MODIS/061. "
        "Export ulang dari GEE Collection 6.1, lalu simpan sebagai "
        "riau_fire_monthly_stats_modis061.csv atau tambahkan kolom "
        "modis_collection_version=MODIS/061."
    )


# %% [markdown]
# ## 2. Audit Kualitas Data dan Preprocessing
#
# Pada CSV lokal yang diperiksa, 2023-2025 memang ada baris datanya, tetapi banyak
# variabel sensor bernilai 0, terutama `mean_lst_celsius` dan `mean_ndvi`. Untuk Riau,
# LST 0 C dan NDVI 0 sepanjang banyak bulan sangat tidak realistis. Ini biasanya
# terjadi karena reducer GEE menerima pixel kosong/masked lalu terisi 0. Solusi teknis:
#
# - re-export GEE tanpa mengubah masked value menjadi 0;
# - cek `count()` pixel valid per bulan untuk MODIS LST/NDVI/MCD64A1;
# - untuk analisis sementara, imputasi variabel sensor memakai klimatologi bulanan;
# - tetap simpan flag kualitas agar angka hasil imputasi tidak dianggap observasi murni.

# %%
def ensure_date_columns(df):
    df = df.copy()
    if "date_label" not in df.columns:
        df["date_label"] = df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2)
    df["date"] = pd.to_datetime(df.get("date", df["date_label"]), errors="coerce")
    missing_date = df["date"].isna()
    if missing_date.any():
        df.loc[missing_date, "date"] = pd.to_datetime(df.loc[missing_date, "date_label"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    return df


def impute_by_month(df, column, mask, fallback):
    valid = (~mask) & df[column].notna()
    monthly = df.loc[valid].groupby("month")[column].median()
    global_value = df.loc[valid, column].median()
    if pd.isna(global_value):
        global_value = fallback

    replacement = df.loc[mask, "month"].map(monthly).fillna(global_value)
    df.loc[mask, column] = replacement
    return df


df = ensure_date_columns(df_raw)
df["source_modis_collection"] = source_modis_collection
df["source_modis061_verified"] = int(source_modis061_verified)

numeric_cols = [
    "fire_area_ha",
    "mean_lst_celsius",
    "total_rain_mm",
    "mean_ndvi",
    "mean_wind_ms",
    "fire_risk_index",
    "has_fire_event",
]
for col in numeric_cols:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")

future_rows = df[df["year"] >= FORECAST_YEAR].copy()
if len(future_rows) > 0:
    print(
        f"Ditemukan {len(future_rows)} baris {FORECAST_YEAR}+ di CSV. "
        "Baris ini tidak dipakai untuk training agar estimasi 2026 tetap out-of-sample."
    )

df = df[
    (df["year"] >= HIST_START_YEAR)
    & (df["year"] <= HIST_END_YEAR)
    & (df["month"].between(1, 12))
].copy()
df = df.sort_values("date").reset_index(drop=True)

df["fire_area_ha"] = df["fire_area_ha"].fillna(0).clip(lower=0)
df["has_fire_event"] = df["has_fire_event"].fillna((df["fire_area_ha"] > 100).astype(int)).astype(int)

df["lst_missing_suspect"] = (
    df["mean_lst_celsius"].isna()
    | (df["mean_lst_celsius"] <= 5)
    | (df["mean_lst_celsius"] > 60)
)
df["ndvi_missing_suspect"] = (
    df["mean_ndvi"].isna()
    | (df["mean_ndvi"] <= 0.02)
    | (df["mean_ndvi"] > 1)
)
df["rain_missing_suspect"] = df["total_rain_mm"].isna() | (df["total_rain_mm"] < 0)
df["wind_missing_suspect"] = df["mean_wind_ms"].isna() | (df["mean_wind_ms"] < 0)
df["fri_missing_suspect"] = df["fire_risk_index"].isna() | (~df["fire_risk_index"].between(0, 1))

df = impute_by_month(df, "mean_lst_celsius", df["lst_missing_suspect"], fallback=30.0)
df = impute_by_month(df, "mean_ndvi", df["ndvi_missing_suspect"], fallback=0.60)
df = impute_by_month(df, "total_rain_mm", df["rain_missing_suspect"], fallback=220.0)
df = impute_by_month(df, "mean_wind_ms", df["wind_missing_suspect"], fallback=2.5)

recompute_fri = (
    df["lst_missing_suspect"]
    | df["ndvi_missing_suspect"]
    | df["rain_missing_suspect"]
    | df["wind_missing_suspect"]
    | df["fri_missing_suspect"]
)
df.loc[recompute_fri, "fire_risk_index"] = compute_fire_risk_index(
    df.loc[recompute_fri, "mean_lst_celsius"],
    df.loc[recompute_fri, "mean_ndvi"],
    df.loc[recompute_fri, "total_rain_mm"],
    df.loc[recompute_fri, "mean_wind_ms"],
)
df["fire_risk_index"] = df["fire_risk_index"].clip(0, 1)
df["fri_recomputed_after_imputation"] = recompute_fri.astype(int)

quality_summary = df.groupby("year").agg(
    months=("date_label", "size"),
    total_fire_ha=("fire_area_ha", "sum"),
    fire_months=("has_fire_event", "sum"),
    mean_fri=("fire_risk_index", "mean"),
    lst_suspect_months=("lst_missing_suspect", "sum"),
    ndvi_suspect_months=("ndvi_missing_suspect", "sum"),
    rain_suspect_months=("rain_missing_suspect", "sum"),
    wind_suspect_months=("wind_missing_suspect", "sum"),
    fri_recomputed_months=("fri_recomputed_after_imputation", "sum"),
).reset_index()

zero_area_suspect_years = quality_summary.loc[
    (quality_summary["months"] >= 10)
    & (quality_summary["total_fire_ha"] <= 0)
    & (
        (quality_summary["lst_suspect_months"] >= 6)
        | (quality_summary["ndvi_suspect_months"] >= 6)
    ),
    "year",
].tolist()

df["fire_area_missing_suspect"] = df["year"].isin(zero_area_suspect_years).astype(int)
df["fire_area_model_ha"] = df["fire_area_ha"].copy()
reliable_area = df["fire_area_missing_suspect"].eq(0)
monthly_area_proxy = df.loc[reliable_area].groupby("month")["fire_area_ha"].median()
global_area_proxy = df.loc[reliable_area, "fire_area_ha"].median()
if pd.isna(global_area_proxy):
    global_area_proxy = 0.0

area_impute_mask = df["fire_area_missing_suspect"].eq(1) & df["fire_area_ha"].eq(0)
df.loc[area_impute_mask, "fire_area_model_ha"] = (
    df.loc[area_impute_mask, "month"].map(monthly_area_proxy).fillna(global_area_proxy)
)
df["fire_area_model_source"] = np.where(
    area_impute_mask,
    "monthly_climatology_for_model",
    "observed",
)

quality_summary = df.groupby("year").agg(
    months=("date_label", "size"),
    total_fire_ha=("fire_area_ha", "sum"),
    model_input_fire_ha=("fire_area_model_ha", "sum"),
    fire_months=("has_fire_event", "sum"),
    mean_fri=("fire_risk_index", "mean"),
    lst_suspect_months=("lst_missing_suspect", "sum"),
    ndvi_suspect_months=("ndvi_missing_suspect", "sum"),
    fri_recomputed_months=("fri_recomputed_after_imputation", "sum"),
    fire_area_missing_suspect_months=("fire_area_missing_suspect", "sum"),
).reset_index()
quality_summary.to_csv(out_path("data_quality_summary.csv"), index=False)

print("\nRingkasan kualitas data per tahun:")
print(quality_summary.round(3).to_string(index=False))
if zero_area_suspect_years:
    print(
        "\nTahun dengan area kebakaran nol dan sensor LST/NDVI mencurigakan: "
        + ", ".join(map(str, zero_area_suspect_years))
    )
    print(
        "Solusi utama: re-export GEE dan simpan jumlah pixel valid per bulan; "
        "script ini memakai imputasi klimatologi hanya sebagai solusi sementara untuk model."
    )


# %% [markdown]
# ## 3. Feature Engineering

# %%
AREA_MODEL_COL = "fire_area_model_ha"

df["drought_signal"] = (df["total_rain_mm"] < 100).astype(int)
df["heat_signal"] = (df["mean_lst_celsius"] > 38).astype(int)
df["dry_veg_signal"] = (df["mean_ndvi"] < 0.4).astype(int)
df["wind_signal"] = (df["mean_wind_ms"] > 4).astype(int)
df["trigger_score"] = (
    df["drought_signal"]
    + df["heat_signal"]
    + df["dry_veg_signal"]
    + df["wind_signal"]
)

for lag in [1, 2, 3]:
    df[f"fire_area_lag{lag}"] = df[AREA_MODEL_COL].shift(lag)
    df[f"fri_lag{lag}"] = df["fire_risk_index"].shift(lag)
    df[f"rain_lag{lag}"] = df["total_rain_mm"].shift(lag)

df["fire_area_roll3"] = df[AREA_MODEL_COL].shift(1).rolling(3).mean()
df["rain_roll3"] = df["total_rain_mm"].shift(1).rolling(3).mean()
df["fri_roll3"] = df["fire_risk_index"].shift(1).rolling(3).mean()
df["lst_roll3"] = df["mean_lst_celsius"].shift(1).rolling(3).mean()

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["risk_category"] = df["fire_risk_index"].apply(categorize_risk)
df["risk_label"] = df["risk_category"].map(RISK_MAP)
df["month_name"] = df["month"].apply(lambda x: MONTH_NAMES_SHORT[int(x) - 1])

df_model = df.dropna().reset_index(drop=True)

print(f"\nDataset final untuk model: {df_model.shape}")
print("\nDistribusi kategori risiko:")
print(df_model["risk_label"].value_counts().to_string())


# %% [markdown]
# ## 4. EDA Ringkas

# %%
fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=[
        "Area terbakar observasi per bulan",
        "Fire Risk Index bulanan",
        "Curah hujan vs area model",
        "Suhu permukaan vs FRI",
        "Jumlah pemicu kebakaran",
        "Rata-rata area terbakar per bulan kalender",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
)

colors_ts = [
    "#e74c3c" if v > 10000 else "#e67e22" if v > 1000 else "#f39c12" if v > 100 else "#bfc9ca"
    for v in df_model["fire_area_ha"]
]
fig.add_trace(
    go.Bar(
        x=df_model["date_label"],
        y=df_model["fire_area_ha"],
        marker_color=colors_ts,
        name="Area observasi",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_model["date_label"],
        y=df_model["fire_risk_index"],
        mode="lines",
        line=dict(color="#c0392b", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(192,57,43,0.15)",
        name="FRI",
    ),
    row=1,
    col=2,
)
fig.add_hline(y=0.50, line_dash="dash", line_color="orange", row=1, col=2)
fig.add_hline(y=0.70, line_dash="dash", line_color="red", row=1, col=2)

fig.add_trace(
    go.Scatter(
        x=df_model["total_rain_mm"],
        y=df_model[AREA_MODEL_COL],
        mode="markers",
        marker=dict(
            color=df_model["risk_category"],
            colorscale="YlOrRd",
            size=7,
            opacity=0.75,
        ),
        name="Hujan vs area model",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_model["mean_lst_celsius"],
        y=df_model["fire_risk_index"],
        mode="markers",
        marker=dict(
            color=df_model[AREA_MODEL_COL],
            colorscale="Reds",
            size=7,
            showscale=True,
        ),
        name="LST vs FRI",
    ),
    row=2,
    col=2,
)

triggers = df_model[["drought_signal", "heat_signal", "dry_veg_signal", "wind_signal"]].sum()
fig.add_trace(
    go.Bar(
        x=["Kekeringan", "Suhu Tinggi", "Vegetasi Kering", "Angin Kencang"],
        y=triggers.values,
        marker_color=["#3498db", "#e74c3c", "#27ae60", "#9b59b6"],
        name="Pemicu",
    ),
    row=3,
    col=1,
)

monthly_avg = df_model.groupby("month")["fire_area_ha"].mean().reindex(range(1, 13), fill_value=0)
fig.add_trace(
    go.Bar(
        x=MONTH_NAMES_SHORT,
        y=monthly_avg.values,
        marker_color=[
            "#e74c3c" if v > 10000 else "#e67e22" if v > 1000 else "#3498db"
            for v in monthly_avg.values
        ],
        name="Rata-rata bulanan",
    ),
    row=3,
    col=2,
)

fig.update_layout(
    height=900,
    title_text=f"EDA Kebakaran Hutan Riau {HIST_START_YEAR}-{HIST_END_YEAR}",
    showlegend=False,
    template="plotly_white",
)
fig.write_html(out_path("eda_riau_fire.html"))
print(f"EDA plot saved: {out_path('eda_riau_fire.html')}")


# %% [markdown]
# ## 5. Bandingkan Beberapa Model Machine Learning
#
# Target klasifikasi adalah kategori risiko dari FRI: Rendah, Sedang, Tinggi,
# Sangat Tinggi. Fitur tidak memakai FRI bulan berjalan agar model tidak sekadar
# mengulang threshold FRI; model belajar dari cuaca, vegetasi, lag, rolling, dan
# sinyal pemicu.

# %%
FEATURES = [
    "mean_lst_celsius",
    "total_rain_mm",
    "mean_ndvi",
    "mean_wind_ms",
    "month_sin",
    "month_cos",
    "fire_area_lag1",
    "fire_area_lag2",
    "fire_area_lag3",
    "fri_lag1",
    "fri_lag2",
    "fri_lag3",
    "rain_lag1",
    "rain_lag2",
    "rain_lag3",
    "fire_area_roll3",
    "rain_roll3",
    "fri_roll3",
    "lst_roll3",
    "drought_signal",
    "heat_signal",
    "dry_veg_signal",
    "wind_signal",
    "trigger_score",
]
TARGET = "risk_category"

X_df = df_model[FEATURES].copy()
y = df_model[TARGET].astype(int).values

split_idx = int(len(X_df) * 0.80)
X_train = X_df.iloc[:split_idx]
X_test = X_df.iloc[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

n_splits = max(2, min(5, len(X_train_sc) // 24))
ts_cv = TimeSeriesSplit(n_splits=n_splits)

model_grid = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [6, 10, None],
            "min_samples_leaf": [1, 2],
        },
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [6, 10, None],
            "min_samples_leaf": [1, 2],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.08],
            "max_depth": [2, 3],
        },
    },
    "Hist Gradient Boosting": {
        "model": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "max_iter": [150, 250],
            "learning_rate": [0.04, 0.08],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 0.1],
        },
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1.0, 5.0],
        },
    },
    "SVC RBF": {
        "model": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "params": {
            "C": [0.5, 1.0, 3.0],
            "gamma": ["scale", 0.1],
        },
    },
}

comparison_rows = []
fitted_models = {}

for model_name, config in model_grid.items():
    print(f"\nTraining dan tuning: {model_name}")
    try:
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="f1_weighted",
            cv=ts_cv,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
        grid.fit(X_train_sc, y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test_sc)
        row = {
            "model_name": model_name,
            "cv_f1_weighted": grid.best_score_,
            "holdout_f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "holdout_accuracy": accuracy_score(y_test, y_pred),
            "holdout_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "best_params": str(grid.best_params_),
            "status": "ok",
        }
        fitted_models[model_name] = model
        print(
            f"{model_name}: holdout F1={row['holdout_f1_weighted']:.3f}, "
            f"accuracy={row['holdout_accuracy']:.3f}"
        )
    except Exception as exc:
        row = {
            "model_name": model_name,
            "cv_f1_weighted": np.nan,
            "holdout_f1_weighted": np.nan,
            "holdout_accuracy": np.nan,
            "holdout_balanced_accuracy": np.nan,
            "best_params": "",
            "status": f"failed: {exc}",
        }
        print(f"{model_name} gagal: {exc}")

    comparison_rows.append(row)

model_comparison = pd.DataFrame(comparison_rows)
model_comparison = model_comparison.sort_values(
    ["holdout_f1_weighted", "cv_f1_weighted"],
    ascending=False,
    na_position="last",
).reset_index(drop=True)
model_comparison.to_csv(out_path("model_comparison_results.csv"), index=False)

if not fitted_models:
    raise RuntimeError("Tidak ada model klasifikasi yang berhasil dilatih.")

BEST_MODEL_NAME = model_comparison.loc[model_comparison["status"].eq("ok"), "model_name"].iloc[0]
best_model = fitted_models[BEST_MODEL_NAME]
y_pred_best = best_model.predict(X_test_sc)

print("\n" + "=" * 70)
print("PERBANDINGAN MODEL MACHINE LEARNING")
print("=" * 70)
print(model_comparison.round(4).to_string(index=False))
print(f"\nModel terbaik yang dipakai untuk estimasi risiko: {BEST_MODEL_NAME}")

print("\nClassification report model terbaik:")
print(
    classification_report(
        y_test,
        y_pred_best,
        labels=[0, 1, 2, 3],
        target_names=RISK_LABELS,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1, 2, 3])
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=RISK_LABELS,
    yticklabels=RISK_LABELS,
)
plt.title(f"Confusion Matrix - {BEST_MODEL_NAME}")
plt.ylabel("Aktual")
plt.xlabel("Prediksi")
plt.tight_layout()
plt.savefig(out_path("confusion_matrix_best_model.png"), dpi=150)
plt.close()


# %% [markdown]
# ## 6. Feature Importance dengan SHAP
#
# SHAP menjelaskan kontribusi setiap fitur terhadap output model. Nilai yang
# dipakai di sini adalah rata-rata absolut kontribusi SHAP pada data test. Jika
# paket SHAP belum terpasang, script otomatis memakai permutation importance:
# fitur diacak satu per satu, lalu penurunan skor F1 diukur.

# %%
def align_class_probabilities(model, prob, class_labels=(0, 1, 2, 3)):
    aligned = np.zeros(len(class_labels))
    for cls, value in zip(model.classes_, prob):
        if int(cls) in class_labels:
            aligned[class_labels.index(int(cls))] = value
    return aligned


def shap_to_feature_scores(shap_values, n_features):
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)

    feature_axes = [idx for idx, size in enumerate(arr.shape) if size == n_features]
    if not feature_axes:
        raise ValueError(f"Tidak dapat menemukan axis fitur pada shape SHAP {arr.shape}")

    feature_axis = feature_axes[0]
    arr = np.moveaxis(arr, feature_axis, -1)
    return np.abs(arr).reshape(-1, n_features).mean(axis=0)


importance_tables = []

if hasattr(best_model, "feature_importances_"):
    native_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": best_model.feature_importances_,
            "method": "model_native",
        }
    )
    importance_tables.append(native_importance)

perm = permutation_importance(
    best_model,
    X_test_sc,
    y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="f1_weighted",
)
permutation_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": perm.importances_mean,
        "importance_std": perm.importances_std,
        "method": "permutation_importance",
    }
)
importance_tables.append(permutation_df)

shap_df = pd.DataFrame()
if SHAP_AVAILABLE:
    try:
        shap_sample_size = min(80, len(X_test_sc))
        shap_background_size = min(60, len(X_train_sc))
        X_shap = pd.DataFrame(X_test_sc[:shap_sample_size], columns=FEATURES)
        X_background = pd.DataFrame(X_train_sc[-shap_background_size:], columns=FEATURES)

        if hasattr(best_model, "feature_importances_"):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_shap)
        else:
            explainer = shap.Explainer(best_model.predict_proba, X_background)
            shap_values = explainer(X_shap)

        shap_scores = shap_to_feature_scores(shap_values, len(FEATURES))
        shap_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": shap_scores,
                "method": "shap_mean_abs",
            }
        )
        importance_tables.append(shap_df)
        print("SHAP berhasil dihitung untuk model terbaik.")
    except Exception as exc:
        print(f"SHAP tidak berhasil dihitung, fallback ke permutation importance: {exc}")

if not shap_df.empty:
    feature_importance = shap_df.sort_values("importance", ascending=False).reset_index(drop=True)
else:
    feature_importance = permutation_df.sort_values("importance", ascending=False).reset_index(drop=True)

feature_importance.to_csv(out_path("feature_importance_best_model.csv"), index=False)
all_importances = pd.concat(importance_tables, ignore_index=True, sort=False)
all_importances.to_csv(out_path("feature_importance_all_methods.csv"), index=False)

top_fi = feature_importance.head(15).sort_values("importance")
plt.figure(figsize=(9, 6))
plt.barh(top_fi["feature"], top_fi["importance"], color="#2f80ed")
plt.title(f"Feature Importance - {BEST_MODEL_NAME} ({feature_importance['method'].iloc[0]})")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(out_path("feature_importance_best_model.png"), dpi=150)
plt.close()

print("\nTop feature importance:")
print(feature_importance.head(12).round(5).to_string(index=False))


# %% [markdown]
# ## 7. Model Area Kebakaran dan Estimasi 2026

# %%
SEQ_LEN = 6
TS_FEATURES = [
    AREA_MODEL_COL,
    "fire_risk_index",
    "total_rain_mm",
    "mean_lst_celsius",
    "mean_ndvi",
    "mean_wind_ms",
    "drought_signal",
    "heat_signal",
    "trigger_score",
]


def make_sequences(data, seq_len):
    Xs, ys = [], []
    for idx in range(len(data) - seq_len):
        Xs.append(data[idx:idx + seq_len])
        ys.append(data[idx + seq_len, 0])
    return np.array(Xs), np.array(ys)


def inverse_fire_area(scaled_vals, scaler_obj, n_features):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals
    return scaler_obj.inverse_transform(dummy)[:, 0]


def build_month_profile(month, historical_df):
    month_df = historical_df[historical_df["month"].eq(month)]
    if len(month_df) == 0:
        rain = [280, 250, 230, 210, 190, 130, 90, 70, 80, 160, 220, 270][month - 1]
        lst = 28 + 6 * np.sin((month - 2) * np.pi / 6)
        ndvi = 0.62 - 0.15 * np.clip(1 - rain / 200, 0, 1)
        wind = 2.5 + 1.2 * (1 - rain / 300)
    else:
        rain = float(month_df["total_rain_mm"].median())
        lst = float(month_df["mean_lst_celsius"].median())
        ndvi = float(month_df["mean_ndvi"].median())
        wind = float(month_df["mean_wind_ms"].median())

    fri = float(np.clip(compute_fire_risk_index(lst, ndvi, rain, wind), 0, 1))
    drought = int(rain < 100)
    heat = int(lst > 38)
    dry_veg = int(ndvi < 0.4)
    wind_signal = int(wind > 4)

    return {
        "mean_lst_celsius": lst,
        "total_rain_mm": rain,
        "mean_ndvi": ndvi,
        "mean_wind_ms": wind,
        "fire_risk_index": fri,
        "drought_signal": drought,
        "heat_signal": heat,
        "dry_veg_signal": dry_veg,
        "wind_signal": wind_signal,
        "trigger_score": drought + heat + dry_veg + wind_signal,
    }


def lag(values, n, default=0.0):
    if len(values) >= n:
        return float(values[-n])
    return float(default)


def build_future_features(month, fire_history, fri_history, rain_history, historical_df):
    profile = build_month_profile(month, historical_df)
    fire_lags = [lag(fire_history, 1), lag(fire_history, 2), lag(fire_history, 3)]
    fri_lags = [lag(fri_history, 1), lag(fri_history, 2), lag(fri_history, 3)]
    rain_lags = [lag(rain_history, 1), lag(rain_history, 2), lag(rain_history, 3)]

    feature_row = {
        "mean_lst_celsius": profile["mean_lst_celsius"],
        "total_rain_mm": profile["total_rain_mm"],
        "mean_ndvi": profile["mean_ndvi"],
        "mean_wind_ms": profile["mean_wind_ms"],
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "fire_area_lag1": fire_lags[0],
        "fire_area_lag2": fire_lags[1],
        "fire_area_lag3": fire_lags[2],
        "fri_lag1": fri_lags[0],
        "fri_lag2": fri_lags[1],
        "fri_lag3": fri_lags[2],
        "rain_lag1": rain_lags[0],
        "rain_lag2": rain_lags[1],
        "rain_lag3": rain_lags[2],
        "fire_area_roll3": float(np.mean(fire_lags)),
        "rain_roll3": float(np.mean(rain_lags)),
        "fri_roll3": float(np.mean(fri_lags)),
        "lst_roll3": float(df_model["mean_lst_celsius"].tail(3).mean()),
        "drought_signal": profile["drought_signal"],
        "heat_signal": profile["heat_signal"],
        "dry_veg_signal": profile["dry_veg_signal"],
        "wind_signal": profile["wind_signal"],
        "trigger_score": profile["trigger_score"],
        "fire_risk_index": profile["fire_risk_index"],
    }
    return feature_row


ts_data = df_model[TS_FEATURES].values.astype(np.float32)
ts_scaler = MinMaxScaler()
ts_scaled = ts_scaler.fit_transform(ts_data)
X_seq, y_seq = make_sequences(ts_scaled, SEQ_LEN)

area_model_name = "RandomForestRegressor fallback"
pred_2026_ha = None

if TENSORFLOW_AVAILABLE and len(X_seq) >= 24:
    split_ts = int(len(X_seq) * 0.80)
    Xs_train, Xs_test = X_seq[:split_ts], X_seq[split_ts:]
    ys_train, ys_test = y_seq[:split_ts], y_seq[split_ts:]

    tf.random.set_seed(RANDOM_STATE)
    model_lstm = Sequential(
        [
            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(SEQ_LEN, len(TS_FEATURES)),
                )
            ),
            Dropout(0.25),
            LSTM(32, return_sequences=False),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model_lstm.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]
    history = model_lstm.fit(
        Xs_train,
        ys_train,
        epochs=150,
        batch_size=16,
        validation_data=(Xs_test, ys_test),
        callbacks=callbacks,
        verbose=1,
    )

    ys_pred_scaled = model_lstm.predict(Xs_test).flatten()
    ys_pred_ha = inverse_fire_area(ys_pred_scaled, ts_scaler, len(TS_FEATURES))
    ys_test_ha = inverse_fire_area(ys_test, ts_scaler, len(TS_FEATURES))
    mae = mean_absolute_error(ys_test_ha, ys_pred_ha)
    rmse = np.sqrt(mean_squared_error(ys_test_ha, ys_pred_ha))
    print(f"\nLSTM test MAE : {mae:,.1f} ha")
    print(f"LSTM test RMSE: {rmse:,.1f} ha")

    plt.figure(figsize=(12, 5))
    plt.plot(ys_test_ha, label="Aktual/model input", color="steelblue")
    plt.plot(ys_pred_ha, label="Prediksi", color="crimson", linestyle="--")
    plt.title("LSTM - Prediksi vs Aktual Area Kebakaran")
    plt.xlabel("Indeks bulan test")
    plt.ylabel("Area terbakar (ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("lstm_prediction_test.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training Loss LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("lstm_loss_curve.png"), dpi=150)
    plt.close()

    last_sequence = ts_scaled[-SEQ_LEN:].copy()
    predictions_scaled = []
    predictions_ha = []

    for month in range(1, 13):
        seq_input = last_sequence[-SEQ_LEN:].reshape(1, SEQ_LEN, len(TS_FEATURES))
        pred_scaled = float(model_lstm.predict(seq_input, verbose=0)[0, 0])
        pred_ha = float(np.clip(inverse_fire_area(np.array([pred_scaled]), ts_scaler, len(TS_FEATURES))[0], 0, None))
        predictions_scaled.append(pred_scaled)
        predictions_ha.append(pred_ha)

        profile = build_month_profile(month, df_model)
        next_raw = np.array(
            [
                pred_ha,
                profile["fire_risk_index"],
                profile["total_rain_mm"],
                profile["mean_lst_celsius"],
                profile["mean_ndvi"],
                profile["mean_wind_ms"],
                profile["drought_signal"],
                profile["heat_signal"],
                profile["trigger_score"],
            ],
            dtype=np.float32,
        ).reshape(1, -1)
        next_scaled = ts_scaler.transform(next_raw)[0]
        last_sequence = np.vstack([last_sequence[1:], next_scaled])

    pred_2026_ha = np.array(predictions_ha)
    area_model_name = "Bidirectional LSTM"

else:
    print("\nTensorFlow tidak tersedia atau data sequence terlalu pendek. Memakai RandomForestRegressor fallback.")
    area_regressor = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    area_regressor.fit(X_train_sc, df_model[AREA_MODEL_COL].iloc[:split_idx])


# %% [markdown]
# ## 8. Estimasi Risiko dan Area Kebakaran per Bulan Tahun 2026

# %%
results_2026 = []
fire_history = df_model[AREA_MODEL_COL].tail(3).tolist()
fri_history = df_model["fire_risk_index"].tail(3).tolist()
rain_history = df_model["total_rain_mm"].tail(3).tolist()

for idx, month in enumerate(range(1, 13)):
    feat = build_future_features(month, fire_history, fri_history, rain_history, df_model)
    fvec = np.array([[feat[f] for f in FEATURES]])
    fvec_sc = scaler.transform(fvec)

    risk_pred = int(best_model.predict(fvec_sc)[0])
    raw_prob = best_model.predict_proba(fvec_sc)[0]
    risk_prob = align_class_probabilities(best_model, raw_prob)

    if pred_2026_ha is None:
        fire_ha = float(np.clip(area_regressor.predict(fvec_sc)[0], 0, None))
    else:
        fire_ha = float(np.clip(pred_2026_ha[idx], 0, None))

    results_2026.append(
        {
            "month": month,
            "month_name": MONTH_NAMES_FULL[month - 1],
            "date_label": f"{FORECAST_YEAR}-{month:02d}",
            "pred_fire_ha": round(fire_ha, 1),
            "pred_risk_cat": risk_pred,
            "pred_risk_label": RISK_MAP[risk_pred],
            "risk_color": RISK_COLOR[risk_pred],
            "prob_rendah": round(float(risk_prob[0]), 3),
            "prob_sedang": round(float(risk_prob[1]), 3),
            "prob_tinggi": round(float(risk_prob[2]), 3),
            "prob_sgt_tinggi": round(float(risk_prob[3]), 3),
            "trigger_score": feat["trigger_score"],
            "mean_lst": round(feat["mean_lst_celsius"], 1),
            "total_rain": round(feat["total_rain_mm"], 1),
            "mean_ndvi": round(feat["mean_ndvi"], 3),
            "mean_wind_ms": round(feat["mean_wind_ms"], 2),
            "fire_risk_index": round(feat["fire_risk_index"], 3),
            "risk_model_name": BEST_MODEL_NAME,
            "area_model_name": area_model_name,
        }
    )

    fire_history.append(fire_ha)
    fri_history.append(feat["fire_risk_index"])
    rain_history.append(feat["total_rain_mm"])

df_2026 = pd.DataFrame(results_2026)

print("\n" + "=" * 70)
print(f"ESTIMASI KEBAKARAN HUTAN RIAU {FORECAST_YEAR}")
print("=" * 70)
print(
    df_2026[
        [
            "month_name",
            "pred_fire_ha",
            "pred_risk_label",
            "trigger_score",
            "mean_lst",
            "total_rain",
            "mean_ndvi",
        ]
    ].to_string(index=False)
)
print(f"\nTotal estimasi area terbakar {FORECAST_YEAR}: {df_2026['pred_fire_ha'].sum():,.0f} ha")
print(f"Bulan puncak estimasi: {df_2026.loc[df_2026['pred_fire_ha'].idxmax(), 'month_name']}")


# %% [markdown]
# ## 9. Visualisasi dan Export

# %%
fig2 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        f"Estimasi area terbakar {FORECAST_YEAR} (ha)",
        f"Estimasi kategori risiko {FORECAST_YEAR}",
        f"Skor pemicu per bulan {FORECAST_YEAR}",
        "Probabilitas risiko per bulan",
    ],
)
fig2.add_trace(
    go.Bar(
        x=df_2026["month_name"],
        y=df_2026["pred_fire_ha"],
        marker_color=df_2026["risk_color"],
        name="Area (ha)",
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=df_2026["month_name"],
        y=df_2026["pred_risk_cat"],
        mode="lines+markers",
        marker=dict(color=df_2026["risk_color"], size=10),
        line=dict(color="gray", width=1.5),
        name="Kategori risiko",
    ),
    row=1,
    col=2,
)
fig2.update_yaxes(
    tickvals=[0, 1, 2, 3],
    ticktext=["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
    row=1,
    col=2,
)
fig2.add_trace(
    go.Bar(
        x=df_2026["month_name"],
        y=df_2026["trigger_score"],
        marker_color=[
            "#e74c3c" if v >= 3 else "#f39c12" if v >= 2 else "#3498db"
            for v in df_2026["trigger_score"]
        ],
        name="Skor pemicu",
    ),
    row=2,
    col=1,
)
for col_name, color, label in [
    ("prob_rendah", "#2ecc71", "Rendah"),
    ("prob_sedang", "#f39c12", "Sedang"),
    ("prob_tinggi", "#e67e22", "Tinggi"),
    ("prob_sgt_tinggi", "#e74c3c", "Sangat Tinggi"),
]:
    fig2.add_trace(
        go.Bar(
            x=df_2026["month_name"],
            y=df_2026[col_name],
            name=label,
            marker_color=color,
        ),
        row=2,
        col=2,
    )
fig2.update_layout(
    height=720,
    barmode="stack",
    title_text=f"Estimasi Kebakaran Hutan Riau {FORECAST_YEAR}",
    template="plotly_white",
)
fig2.write_html(out_path("estimasi_2026_riau.html"))

df_model.to_csv(out_path("riau_fire_processed.csv"), index=False)
df_2026.to_csv(out_path("riau_fire_estimasi_2026.csv"), index=False)

df_annual = df_model.groupby("year").agg(
    total_fire_ha=("fire_area_ha", "sum"),
    model_input_fire_ha=(AREA_MODEL_COL, "sum"),
    max_fire_ha=("fire_area_ha", "max"),
    fire_months=("has_fire_event", "sum"),
    mean_fri=("fire_risk_index", "mean"),
    mean_rain=("total_rain_mm", "mean"),
    mean_lst=("mean_lst_celsius", "mean"),
    suspect_sensor_months=("fri_recomputed_after_imputation", "sum"),
    suspect_fire_area_months=("fire_area_missing_suspect", "sum"),
).reset_index()
df_annual.to_csv(out_path("riau_fire_annual_summary.csv"), index=False)

run_metadata = pd.DataFrame(
    [
        {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "data_source": data_source,
            "expected_modis_collection_version": MODIS_COLLECTION_VERSION,
            "source_modis_collection": source_modis_collection,
            "source_modis061_verified": source_modis061_verified,
            "gee_collections": "; ".join(GEE_COLLECTIONS),
            "historical_period": f"{HIST_START_YEAR}-{HIST_END_YEAR}",
            "forecast_year": FORECAST_YEAR,
            "best_risk_model": BEST_MODEL_NAME,
            "area_model": area_model_name,
            "shap_available": SHAP_AVAILABLE,
        }
    ]
)
run_metadata.to_csv(out_path("model_run_metadata.csv"), index=False)

joblib.dump(best_model, out_path("best_risk_model_riau.pkl"))
joblib.dump(scaler, out_path("risk_model_scaler.pkl"))
joblib.dump(ts_scaler, out_path("area_lstm_scaler.pkl"))
if TENSORFLOW_AVAILABLE and "model_lstm" in globals():
    model_lstm.save(out_path("lstm_model_riau.h5"))
if "area_regressor" in globals():
    joblib.dump(area_regressor, out_path("area_regressor_fallback.pkl"))

print("\nFile export:")
for filename in [
    "riau_fire_processed.csv",
    "riau_fire_estimasi_2026.csv",
    "riau_fire_annual_summary.csv",
    "data_quality_summary.csv",
    "model_comparison_results.csv",
    "feature_importance_best_model.csv",
    "model_run_metadata.csv",
    "eda_riau_fire.html",
    "estimasi_2026_riau.html",
]:
    print(f"- {out_path(filename)}")

print("\nModel tersimpan:")
print(f"- {out_path('best_risk_model_riau.pkl')}")
print(f"- {out_path('risk_model_scaler.pkl')}")
print("\nSelesai.")
