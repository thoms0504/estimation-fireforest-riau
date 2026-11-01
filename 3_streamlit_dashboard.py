# ============================================================
# STREAMLIT DASHBOARD - KEBAKARAN HUTAN RIAU
# Analisis historis 2015-2025, kualitas data, model terbaik,
# feature importance, peta, dan estimasi bulanan 2026.
# Jalankan: streamlit run 3_streamlit_dashboard.py
# ============================================================

import os
import warnings

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Deteksi Kebakaran Hutan Riau",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONTH_NAMES_SHORT = [
    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
    "Jul", "Ags", "Sep", "Okt", "Nov", "Des",
]
MONTH_NAMES_FULL = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]
RISK_LABELS = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
RISK_COLORS = {
    "Rendah": "#2e7d32",
    "Sedang": "#f9a825",
    "Tinggi": "#ef6c00",
    "Sangat Tinggi": "#c62828",
}
HOTSPOT_CENTERS = {
    "Pelalawan": (-0.38, 102.78),
    "Bengkalis": (1.48, 102.10),
    "Siak": (1.12, 102.00),
    "Kepulauan Meranti": (1.02, 102.55),
    "Rokan Hilir": (2.15, 100.90),
    "Indragiri Hulu": (-0.30, 102.30),
    "Indragiri Hilir": (-0.40, 103.10),
    "Kampar": (0.35, 101.15),
    "Kuantan Singingi": (-0.50, 101.50),
}

st.markdown(
    """
<style>
    .block-container { padding-top: 1.4rem; }
    .page-title {
        padding: 16px 20px;
        border-left: 6px solid #c62828;
        background: #fff7f2;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .page-title h1 { margin: 0; font-size: 1.55rem; color: #2b2b2b; }
    .page-title p { margin: 4px 0 0; color: #5f6368; }
    .note-box {
        padding: 12px 14px;
        border-radius: 8px;
        background: #f7f9fc;
        border: 1px solid #dbe3ef;
        color: #263238;
    }
</style>
""",
    unsafe_allow_html=True,
)


def compute_fire_risk_index(lst, ndvi, rain, wind):
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


def impute_by_month(df, column, mask, fallback):
    valid = (~mask) & df[column].notna()
    monthly = df.loc[valid].groupby("month")[column].median()
    global_value = df.loc[valid, column].median()
    if pd.isna(global_value):
        global_value = fallback
    df.loc[mask, column] = df.loc[mask, "month"].map(monthly).fillna(global_value)
    return df


def synthetic_data():
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    seasonal_rain = np.array([280, 250, 230, 210, 190, 130, 90, 70, 80, 160, 220, 270])
    elnino = {2015: 1.8, 2019: 1.6, 2023: 1.3}
    rows = []

    for d in dates:
        y, m = d.year, d.month
        factor = elnino.get(y, 1.0)
        rain = max(10, seasonal_rain[m - 1] / factor + np.random.normal(0, 25))
        lst = 28 + 8 * np.sin((m - 2) * np.pi / 6) + (factor - 1) * 3 + np.random.normal(0, 0.8)
        ndvi = np.clip(0.65 - 0.25 * np.clip(1 - rain / 200, 0, 1) + np.random.normal(0, 0.03), -0.1, 0.9)
        wind = max(0.5, 2.5 + 1.5 * (1 - rain / 300) + np.random.normal(0, 0.3))
        fri = float(np.clip(compute_fire_risk_index(lst, ndvi, rain, wind), 0, 1))
        risk = categorize_risk(fri)
        fire = 0.0
        if fri > 0.45 and np.random.random() < np.clip((fri - 0.3) / 0.5, 0, 1) * factor:
            fire = np.random.lognormal(np.log(max(500, fri * 80000)), 0.8)
            if y == 2015 and m in [9, 10]:
                fire *= 3

        rows.append(
            {
                "year": y,
                "month": m,
                "date_label": f"{y}-{m:02d}",
                "date": d,
                "fire_area_ha": round(fire, 1),
                "mean_lst_celsius": round(lst, 2),
                "total_rain_mm": round(rain, 1),
                "mean_ndvi": round(float(ndvi), 4),
                "mean_wind_ms": round(wind, 2),
                "fire_risk_index": round(fri, 4),
                "has_fire_event": int(fire > 100),
                "risk_category": risk,
                "risk_label": RISK_LABELS[risk],
            }
        )
    return pd.DataFrame(rows)


def prepare_raw_data(df_raw):
    df = df_raw.copy()
    if "date_label" not in df.columns:
        df["date_label"] = df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2)
    df["date"] = pd.to_datetime(df.get("date", df["date_label"]), errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df[(df["year"] >= 2015) & (df["year"] <= 2025) & (df["month"].between(1, 12))].copy()

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

    df["fire_area_ha"] = df["fire_area_ha"].fillna(0).clip(lower=0)
    df["has_fire_event"] = df["has_fire_event"].fillna((df["fire_area_ha"] > 100).astype(int)).astype(int)

    df["lst_missing_suspect"] = df["mean_lst_celsius"].isna() | (df["mean_lst_celsius"] <= 5) | (df["mean_lst_celsius"] > 60)
    df["ndvi_missing_suspect"] = df["mean_ndvi"].isna() | (df["mean_ndvi"] <= 0.02) | (df["mean_ndvi"] > 1)
    df["rain_missing_suspect"] = df["total_rain_mm"].isna() | (df["total_rain_mm"] < 0)
    df["wind_missing_suspect"] = df["mean_wind_ms"].isna() | (df["mean_wind_ms"] < 0)
    df["fri_missing_suspect"] = df["fire_risk_index"].isna() | (~df["fire_risk_index"].between(0, 1))

    df = impute_by_month(df, "mean_lst_celsius", df["lst_missing_suspect"], 30.0)
    df = impute_by_month(df, "mean_ndvi", df["ndvi_missing_suspect"], 0.60)
    df = impute_by_month(df, "total_rain_mm", df["rain_missing_suspect"], 220.0)
    df = impute_by_month(df, "mean_wind_ms", df["wind_missing_suspect"], 2.5)

    recompute = (
        df["lst_missing_suspect"]
        | df["ndvi_missing_suspect"]
        | df["rain_missing_suspect"]
        | df["wind_missing_suspect"]
        | df["fri_missing_suspect"]
    )
    df.loc[recompute, "fire_risk_index"] = compute_fire_risk_index(
        df.loc[recompute, "mean_lst_celsius"],
        df.loc[recompute, "mean_ndvi"],
        df.loc[recompute, "total_rain_mm"],
        df.loc[recompute, "mean_wind_ms"],
    )
    df["fire_risk_index"] = df["fire_risk_index"].clip(0, 1)
    df["fri_recomputed_after_imputation"] = recompute.astype(int)

    yearly = df.groupby("year").agg(
        months=("date_label", "size"),
        total_fire_ha=("fire_area_ha", "sum"),
        lst_suspect_months=("lst_missing_suspect", "sum"),
        ndvi_suspect_months=("ndvi_missing_suspect", "sum"),
    ).reset_index()
    suspect_years = yearly.loc[
        (yearly["months"] >= 10)
        & (yearly["total_fire_ha"] <= 0)
        & ((yearly["lst_suspect_months"] >= 6) | (yearly["ndvi_suspect_months"] >= 6)),
        "year",
    ].tolist()
    df["fire_area_missing_suspect"] = df["year"].isin(suspect_years).astype(int)
    df["fire_area_model_ha"] = df["fire_area_ha"]
    reliable = df["fire_area_missing_suspect"].eq(0)
    monthly_proxy = df.loc[reliable].groupby("month")["fire_area_ha"].median()
    fallback_area = df.loc[reliable, "fire_area_ha"].median()
    if pd.isna(fallback_area):
        fallback_area = 0.0
    mask = df["fire_area_missing_suspect"].eq(1) & df["fire_area_ha"].eq(0)
    df.loc[mask, "fire_area_model_ha"] = df.loc[mask, "month"].map(monthly_proxy).fillna(fallback_area)
    df["fire_area_model_source"] = np.where(mask, "monthly_climatology_for_model", "observed")

    df["drought_signal"] = (df["total_rain_mm"] < 100).astype(int)
    df["heat_signal"] = (df["mean_lst_celsius"] > 38).astype(int)
    df["dry_veg_signal"] = (df["mean_ndvi"] < 0.4).astype(int)
    df["wind_signal"] = (df["mean_wind_ms"] > 4).astype(int)
    df["trigger_score"] = df[["drought_signal", "heat_signal", "dry_veg_signal", "wind_signal"]].sum(axis=1)
    df["risk_category"] = df["fire_risk_index"].apply(categorize_risk)
    df["risk_label"] = df["risk_category"].map(dict(enumerate(RISK_LABELS)))
    df["month_name"] = df["month"].apply(lambda x: MONTH_NAMES_SHORT[int(x) - 1])
    df["date"] = pd.to_datetime(df["date_label"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fallback_forecast(df):
    rows = []
    for month in range(1, 13):
        hist = df[df["month"].eq(month)]
        rain = float(hist["total_rain_mm"].median()) if len(hist) else 200.0
        lst = float(hist["mean_lst_celsius"].median()) if len(hist) else 30.0
        ndvi = float(hist["mean_ndvi"].median()) if len(hist) else 0.60
        wind = float(hist["mean_wind_ms"].median()) if len(hist) else 2.5
        fri = float(np.clip(compute_fire_risk_index(lst, ndvi, rain, wind), 0, 1))
        risk = categorize_risk(fri)
        area_base = float(hist["fire_area_model_ha"].median()) if len(hist) else 0.0
        area = max(0.0, area_base * (0.8 + fri))
        rows.append(
            {
                "month": month,
                "month_name": MONTH_NAMES_FULL[month - 1],
                "date_label": f"2026-{month:02d}",
                "pred_fire_ha": round(area, 1),
                "pred_risk_cat": risk,
                "pred_risk_label": RISK_LABELS[risk],
                "risk_color": list(RISK_COLORS.values())[risk],
                "prob_rendah": 1.0 if risk == 0 else 0.0,
                "prob_sedang": 1.0 if risk == 1 else 0.0,
                "prob_tinggi": 1.0 if risk == 2 else 0.0,
                "prob_sgt_tinggi": 1.0 if risk == 3 else 0.0,
                "trigger_score": int(rain < 100) + int(lst > 38) + int(ndvi < 0.4) + int(wind > 4),
                "mean_lst": round(lst, 1),
                "total_rain": round(rain, 1),
                "mean_ndvi": round(ndvi, 3),
                "mean_wind_ms": round(wind, 2),
                "fire_risk_index": round(fri, 3),
                "risk_model_name": "fallback climatology",
                "area_model_name": "fallback climatology",
            }
        )
    return pd.DataFrame(rows)


def load_optional_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def file_signature():
    paths = [
        "riau_fire_monthly_stats_modis061.csv",
        "riau_fire_monthly_stats.csv",
        "riau_fire_processed.csv",
        "riau_fire_estimasi_2026.csv",
        "model_comparison_results.csv",
        "feature_importance_best_model.csv",
        "data_quality_summary.csv",
        "model_run_metadata.csv",
    ]
    signature = []
    for path in paths:
        if os.path.exists(path):
            signature.append((path, os.path.getmtime(path), os.path.getsize(path)))
        else:
            signature.append((path, None, None))
    return tuple(signature)


def raw_csv_newer_than_processed():
    raw_path = (
        "riau_fire_monthly_stats_modis061.csv"
        if os.path.exists("riau_fire_monthly_stats_modis061.csv")
        else "riau_fire_monthly_stats.csv"
    )
    processed_path = "riau_fire_processed.csv"
    if not os.path.exists(raw_path) or not os.path.exists(processed_path):
        return False
    return os.path.getmtime(raw_path) > os.path.getmtime(processed_path)


@st.cache_data
def load_data(signature):
    if os.path.exists("riau_fire_processed.csv"):
        df = pd.read_csv("riau_fire_processed.csv")
        df["date"] = pd.to_datetime(df["date_label"], errors="coerce")
        if "month_name" not in df.columns:
            df["month_name"] = df["month"].apply(lambda x: MONTH_NAMES_SHORT[int(x) - 1])
    elif os.path.exists("riau_fire_monthly_stats_modis061.csv"):
        df = prepare_raw_data(pd.read_csv("riau_fire_monthly_stats_modis061.csv"))
        df["source_modis_collection"] = "MODIS/061 (terdeteksi dari nama file)"
        df["source_modis061_verified"] = 1
    elif os.path.exists("riau_fire_monthly_stats.csv"):
        df = prepare_raw_data(pd.read_csv("riau_fire_monthly_stats.csv"))
        if "source_modis061_verified" not in df.columns:
            df["source_modis_collection"] = "Belum terverifikasi - CSV tidak memuat metadata MODIS/061"
            df["source_modis061_verified"] = 0
    else:
        df = prepare_raw_data(synthetic_data())

    required_flags = [
        "lst_missing_suspect",
        "ndvi_missing_suspect",
        "fri_recomputed_after_imputation",
        "fire_area_missing_suspect",
    ]
    for col in required_flags:
        if col not in df.columns:
            df[col] = 0
    if "fire_area_model_ha" not in df.columns:
        df["fire_area_model_ha"] = df["fire_area_ha"]
    if "risk_label" not in df.columns:
        df["risk_category"] = df["fire_risk_index"].apply(categorize_risk)
        df["risk_label"] = df["risk_category"].map(dict(enumerate(RISK_LABELS)))
    if "month_name" not in df.columns:
        df["month_name"] = df["month"].apply(lambda x: MONTH_NAMES_SHORT[int(x) - 1])

    df26 = pd.read_csv("riau_fire_estimasi_2026.csv") if os.path.exists("riau_fire_estimasi_2026.csv") else fallback_forecast(df)
    model_comparison = load_optional_csv("model_comparison_results.csv")
    feature_importance = load_optional_csv("feature_importance_best_model.csv")
    quality = load_optional_csv("data_quality_summary.csv")
    run_metadata = load_optional_csv("model_run_metadata.csv")

    if feature_importance.empty:
        candidate_cols = [
            "mean_lst_celsius",
            "total_rain_mm",
            "mean_ndvi",
            "mean_wind_ms",
            "fire_risk_index",
            "trigger_score",
        ]
        corr_rows = []
        for col in candidate_cols:
            corr = df[[col, "fire_area_model_ha"]].corr().iloc[0, 1]
            corr_rows.append({"feature": col, "importance": abs(corr), "method": "fallback_abs_correlation"})
        feature_importance = pd.DataFrame(corr_rows).sort_values("importance", ascending=False)

    if quality.empty:
        quality = df.groupby("year").agg(
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

    return df, df26, model_comparison, feature_importance, quality, run_metadata


df, df26, model_comparison, feature_importance, quality_summary, run_metadata = load_data(file_signature())
raw_is_newer = raw_csv_newer_than_processed()
metadata_missing = run_metadata.empty
modis_collection_label = "Belum terverifikasi - target MODIS/061"
source_modis061_verified = False
if not run_metadata.empty and "source_modis_collection" in run_metadata.columns:
    modis_collection_label = str(run_metadata["source_modis_collection"].iloc[0])
    if "source_modis061_verified" in run_metadata.columns:
        source_modis061_verified = str(run_metadata["source_modis061_verified"].iloc[0]).lower() in ["true", "1", "yes"]
elif "source_modis_collection" in df.columns:
    modis_collection_label = str(df["source_modis_collection"].dropna().iloc[0])
    if "source_modis061_verified" in df.columns:
        source_modis061_verified = bool(int(df["source_modis061_verified"].dropna().iloc[0]))

with st.sidebar:
    st.markdown("### Kebakaran Hutan Riau")
    menu = st.radio(
        "Menu",
        [
            "Overview",
            "Peta",
            "Analisis Variabel",
            "Model & Kualitas Data",
            "Estimasi 2026",
            "Tabel Data",
        ],
    )
    st.markdown("---")
    year_range = st.slider(
        "Rentang tahun",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )
    selected_months = st.multiselect(
        "Bulan",
        options=list(range(1, 13)),
        default=list(range(1, 13)),
        format_func=lambda x: MONTH_NAMES_SHORT[x - 1],
    )
    min_fire = st.number_input("Minimum area observasi (ha)", value=0, step=100)
    st.markdown("---")
    map_tile = st.selectbox(
        "Tile peta",
        ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Stamen Terrain"],
        index=0,
        help="OpenStreetMap tidak memerlukan API key. CartoDB/Stamen memerlukan koneksi internet.",
    )
    st.markdown("---")
    st.caption(f"Data historis: {len(df)} bulan")
    st.caption(f"Bulan dengan kebakaran: {int(df['has_fire_event'].sum())}")
    st.caption(f"Bulan sensor/imputasi: {int(df['fri_recomputed_after_imputation'].sum())}")
    st.caption(f"Koleksi MODIS: {modis_collection_label}")
    if not source_modis061_verified:
        st.error(
            "Data yang sedang tampil belum terverifikasi berasal dari MODIS/061. "
            "Export ulang GEE Collection 6.1, lalu simpan sebagai riau_fire_monthly_stats_modis061.csv "
            "dan jalankan ulang script ML."
        )
    if metadata_missing:
        st.warning(
            "Metadata run model belum ada. Setelah re-export GEE dari MODIS/061, "
            "jalankan ulang 2_Colab_ML_fire_prediction.py agar metadata, model, dan estimasi sinkron."
        )
    if raw_is_newer:
        st.warning(
            "CSV GEE mentah lebih baru dari riau_fire_processed.csv. "
            "Jalankan ulang 2_Colab_ML_fire_prediction.py agar model dan estimasi Streamlit ikut memakai data terbaru."
        )

df_filt = df[
    (df["year"].between(year_range[0], year_range[1]))
    & (df["month"].isin(selected_months))
    & (df["fire_area_ha"] >= min_fire)
].copy()

st.markdown(
    """
<div class="page-title">
    <h1>Sistem Deteksi Dini Kebakaran Hutan - Provinsi Riau</h1>
    <p>Perbandingan model machine learning, SHAP/importance, peta risiko, dan estimasi bulanan 2026.</p>
</div>
""",
    unsafe_allow_html=True,
)


def safe_metric_value(value, fmt="{:,.0f}"):
    if pd.isna(value):
        return "N/A"
    return fmt.format(value)


def annual_summary(data):
    if data.empty:
        return pd.DataFrame()
    out = data.groupby("year").agg(
        total_ha=("fire_area_ha", "sum"),
        model_input_ha=("fire_area_model_ha", "sum"),
        max_ha=("fire_area_ha", "max"),
        fire_months=("has_fire_event", "sum"),
        mean_fri=("fire_risk_index", "mean"),
        sensor_suspect=("fri_recomputed_after_imputation", "sum"),
    ).reset_index()
    return out


def render_map(row, mode_label, tile="OpenStreetMap"):
    if row is None:
        st.info("Tidak ada data untuk periode yang dipilih.")
        return

    if mode_label == "Estimasi 2026":
        area_value = float(row.get("pred_fire_ha", 0))
        fri_value = float(row.get("fire_risk_index", 0))
        risk_label = row.get("pred_risk_label", "Rendah")
        seed_text = str(row.get("date_label", "2026"))
    else:
        area_value = float(row.get("fire_area_ha", 0))
        fri_value = float(row.get("fire_risk_index", 0))
        risk_label = row.get("risk_label", "Rendah")
        seed_text = str(row.get("date_label", "hist"))

    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    # OpenStreetMap = tidak perlu API key; CartoDB/Mapbox butuh token.
    fmap = folium.Map(location=[0.55, 101.95], zoom_start=7, tiles=tile)
    bounds = [[-1.15, 100.0], [2.35, 104.2]]
    folium.Rectangle(
        bounds=bounds,
        color="#455a64",
        weight=1,
        fill=True,
        fill_opacity=0.03,
        tooltip="Perkiraan cakupan Provinsi Riau",
    ).add_to(fmap)

    intensity_base = max(fri_value, min(1.0, np.log1p(area_value) / 12))
    point_count_base = 1 if area_value <= 0 and fri_value < 0.35 else max(2, int(intensity_base * 18))

    for kab, (lat, lon) in HOTSPOT_CENTERS.items():
        weight = rng.uniform(0.5, 1.4)
        n_points = max(1, int(point_count_base * weight / 3))
        for _ in range(n_points):
            jitter_lat = lat + rng.normal(0, 0.18)
            jitter_lon = lon + rng.normal(0, 0.18)
            intensity = float(np.clip(intensity_base * rng.uniform(0.65, 1.15), 0.05, 1.0))
            color = "#c62828" if intensity >= 0.65 else "#ef6c00" if intensity >= 0.45 else "#f9a825"
            folium.CircleMarker(
                location=[jitter_lat, jitter_lon],
                radius=4 + intensity * 11,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.68,
                weight=0.8,
                tooltip=f"{kab} | FRI {fri_value:.3f} | Risiko {risk_label}",
            ).add_to(fmap)

    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 12px; border: 1px solid #ccd3dc;
                border-radius: 8px; font-size: 12px;">
        <b>{mode_label}</b><br>
        Area: {area_value:,.0f} ha<br>
        FRI: {fri_value:.3f}<br>
        Risiko: {risk_label}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    st_folium(fmap, height=540, use_container_width=True)


if menu == "Overview":
    total_fire = df_filt["fire_area_ha"].sum() if not df_filt.empty else np.nan
    max_fire = df_filt["fire_area_ha"].max() if not df_filt.empty else np.nan
    fire_months = int(df_filt["has_fire_event"].sum()) if not df_filt.empty else 0
    avg_fri = df_filt["fire_risk_index"].mean() if not df_filt.empty else np.nan
    suspect_months = int(df_filt["fri_recomputed_after_imputation"].sum()) if not df_filt.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total area observasi", safe_metric_value(total_fire), "ha")
    c2.metric("Puncak bulanan", safe_metric_value(max_fire), "ha")
    c3.metric("Bulan kebakaran", f"{fire_months}")
    c4.metric("Rata-rata FRI", safe_metric_value(avg_fri, "{:.3f}"))
    c5.metric("Bulan sensor diimputasi", f"{suspect_months}")

    if suspect_months > 0:
        st.warning(
            "Sebagian bulan memiliki LST/NDVI bernilai no-data atau tidak realistis. "
            "Dashboard menampilkan flag kualitas dan model memakai imputasi klimatologi bulanan."
        )

    fig_ts = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Area terbakar observasi", "Fire Risk Index"],
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
    )
    bar_colors = [
        "#c62828" if v > 10000 else "#ef6c00" if v > 1000 else "#f9a825" if v > 100 else "#90a4ae"
        for v in df_filt["fire_area_ha"]
    ]
    fig_ts.add_trace(go.Bar(x=df_filt["date_label"], y=df_filt["fire_area_ha"], marker_color=bar_colors, name="Area"), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=df_filt["date_label"], y=df_filt["fire_risk_index"], fill="tozeroy", line=dict(color="#c62828"), name="FRI"), row=2, col=1)
    fig_ts.add_hline(y=0.5, line_dash="dash", line_color="#ef6c00", row=2, col=1)
    fig_ts.add_hline(y=0.7, line_dash="dash", line_color="#c62828", row=2, col=1)
    fig_ts.update_layout(height=560, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_ts, use_container_width=True)

    annual = annual_summary(df_filt)
    st.subheader("Ringkasan Tahunan")
    if not annual.empty:
        fig_annual = px.bar(
            annual,
            x="year",
            y=["total_ha", "model_input_ha"],
            barmode="group",
            labels={"value": "Area (ha)", "year": "Tahun", "variable": "Seri"},
            color_discrete_sequence=["#c62828", "#1976d2"],
        )
        fig_annual.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_annual, use_container_width=True)
        st.dataframe(annual.round(2), use_container_width=True, height=280)

elif menu == "Peta":
    st.subheader("Peta Risiko dan Fire Mask")
    map_mode = st.radio("Mode peta", ["Historis", "Estimasi 2026"], horizontal=True)

    left, right = st.columns([1, 3])
    with left:
        if map_mode == "Historis":
            sel_year = st.selectbox("Tahun", sorted(df["year"].unique()), index=0)
            sel_month = st.selectbox("Bulan", list(range(1, 13)), format_func=lambda x: MONTH_NAMES_FULL[x - 1])
            selected = df[(df["year"].eq(sel_year)) & (df["month"].eq(sel_month))]
            row = selected.iloc[0] if len(selected) else None
            if row is not None:
                st.metric("Area observasi", f"{row['fire_area_ha']:,.0f} ha")
                st.metric("FRI", f"{row['fire_risk_index']:.3f}", row["risk_label"])
                st.metric("Curah hujan", f"{row['total_rain_mm']:.1f} mm")
                st.metric("LST", f"{row['mean_lst_celsius']:.1f} C")
        else:
            sel_month = st.selectbox("Bulan 2026", list(range(1, 13)), format_func=lambda x: MONTH_NAMES_FULL[x - 1])
            selected = df26[df26["month"].eq(sel_month)]
            row = selected.iloc[0] if len(selected) else None
            if row is not None:
                st.metric("Estimasi area", f"{row['pred_fire_ha']:,.0f} ha")
                st.metric("FRI", f"{row['fire_risk_index']:.3f}", row["pred_risk_label"])
                st.metric("Curah hujan", f"{row['total_rain']:.1f} mm")
                st.metric("LST", f"{row['mean_lst']:.1f} C")

    with right:
        render_map(row, map_mode, tile=map_tile)

elif menu == "Analisis Variabel":
    st.subheader("Variabel yang Berpengaruh")
    col1, col2 = st.columns(2)
    with col1:
        fig_s1 = px.scatter(
            df_filt,
            x="total_rain_mm",
            y="fire_area_ha",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            size="fire_risk_index",
            hover_data=["date_label", "mean_lst_celsius", "mean_ndvi"],
            labels={"total_rain_mm": "Curah hujan (mm)", "fire_area_ha": "Area observasi (ha)"},
            title="Curah Hujan vs Area Terbakar",
        )
        fig_s1.update_layout(height=430, template="plotly_white")
        st.plotly_chart(fig_s1, use_container_width=True)
    with col2:
        fig_s2 = px.scatter(
            df_filt,
            x="mean_lst_celsius",
            y="fire_risk_index",
            color="fire_area_ha",
            color_continuous_scale="Reds",
            size="trigger_score",
            hover_data=["date_label", "total_rain_mm"],
            labels={"mean_lst_celsius": "LST (C)", "fire_risk_index": "FRI"},
            title="Suhu Permukaan vs Fire Risk Index",
        )
        fig_s2.update_layout(height=430, template="plotly_white")
        st.plotly_chart(fig_s2, use_container_width=True)

    monthly = df_filt.groupby("month").agg(
        avg_fire=("fire_area_ha", "mean"),
        avg_fri=("fire_risk_index", "mean"),
        avg_rain=("total_rain_mm", "mean"),
        avg_ndvi=("mean_ndvi", "mean"),
    ).reindex(range(1, 13)).reset_index()
    monthly["month_name"] = monthly["month"].apply(lambda x: MONTH_NAMES_SHORT[int(x) - 1])
    fig_month = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Area rata-rata", "FRI rata-rata", "Hujan rata-rata", "NDVI rata-rata"],
    )
    for r, c, y_col, color in [
        (1, 1, "avg_fire", "#c62828"),
        (1, 2, "avg_fri", "#ef6c00"),
        (2, 1, "avg_rain", "#1976d2"),
        (2, 2, "avg_ndvi", "#2e7d32"),
    ]:
        fig_month.add_trace(go.Bar(x=monthly["month_name"], y=monthly[y_col], marker_color=color, showlegend=False), row=r, col=c)
    fig_month.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Feature Importance Model Terbaik")
    top_features = feature_importance.head(15).sort_values("importance", ascending=True)
    fig_fi = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
        title=f"Top fitur ({feature_importance['method'].iloc[0]})",
    )
    fig_fi.update_layout(height=480, template="plotly_white", yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    method = str(feature_importance["method"].iloc[0])
    if "shap" in method.lower():
        st.markdown(
            '<div class="note-box">SHAP membaca seberapa besar setiap fitur menggeser prediksi model. '
            'Bar terbesar berarti fitur itu paling sering memberi kontribusi absolut terbesar pada keputusan model.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note-box">SHAP belum tersedia atau belum dihitung. Grafik memakai permutation importance/korelasi fallback.</div>',
            unsafe_allow_html=True,
        )

    corr_cols = [
        "fire_area_ha",
        "fire_risk_index",
        "total_rain_mm",
        "mean_lst_celsius",
        "mean_ndvi",
        "mean_wind_ms",
        "trigger_score",
    ]
    corr = df_filt[corr_cols].corr().round(2)
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr_cols,
            y=corr_cols,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig_corr.update_layout(title="Korelasi Antar Variabel", height=470, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "Model & Kualitas Data":
    st.subheader("Perbandingan Model Machine Learning")
    if not run_metadata.empty:
        st.caption("Metadata run terakhir")
        st.dataframe(run_metadata, use_container_width=True, height=75)

    if model_comparison.empty:
        st.info("File model_comparison_results.csv belum ada. Jalankan 2_Colab_ML_fire_prediction.py untuk melihat hasil tuning model.")
    else:
        metric_cols = ["holdout_f1_weighted", "holdout_accuracy", "holdout_balanced_accuracy"]
        fig_cmp = px.bar(
            model_comparison,
            x="model_name",
            y="holdout_f1_weighted",
            color="holdout_accuracy",
            color_continuous_scale="Viridis",
            title="Skor holdout F1 weighted per model",
            labels={"model_name": "Model", "holdout_f1_weighted": "F1 weighted"},
        )
        fig_cmp.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.dataframe(model_comparison, use_container_width=True, height=300)
        best = model_comparison.iloc[0]
        st.success(f"Model terbaik yang digunakan: {best['model_name']} dengan F1 holdout {best['holdout_f1_weighted']:.3f}.")

    st.subheader("Analisis Data 2023-2025")
    q = quality_summary.copy()
    fig_q = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Area observasi vs area input model", "Jumlah bulan bermasalah"],
        vertical_spacing=0.12,
    )
    fig_q.add_trace(go.Bar(x=q["year"], y=q["total_fire_ha"], name="Area observasi", marker_color="#c62828"), row=1, col=1)
    if "model_input_fire_ha" in q.columns:
        fig_q.add_trace(go.Bar(x=q["year"], y=q["model_input_fire_ha"], name="Area input model", marker_color="#1976d2"), row=1, col=1)
    for col, color, label in [
        ("lst_suspect_months", "#ef6c00", "LST suspect"),
        ("ndvi_suspect_months", "#2e7d32", "NDVI suspect"),
        ("fire_area_missing_suspect_months", "#6a1b9a", "Area missing suspect"),
    ]:
        if col in q.columns:
            fig_q.add_trace(go.Bar(x=q["year"], y=q[col], name=label, marker_color=color), row=2, col=1)
    fig_q.update_layout(height=620, template="plotly_white", barmode="group")
    st.plotly_chart(fig_q, use_container_width=True)

    focus = q[q["year"].between(2023, 2025)]
    st.dataframe(focus.round(3), use_container_width=True, height=180)
    st.markdown(
        """
<div class="note-box">
Pada CSV lokal, 2023 masih memiliki sebagian area kebakaran, sedangkan 2024-2025 bernilai 0 untuk area dan banyak bulan memiliki LST/NDVI = 0. 
LST 0 C dan NDVI 0 berbulan-bulan tidak realistis untuk Riau, sehingga lebih masuk akal dibaca sebagai no-data/masked pixel dari proses export, bukan kondisi lapangan aman.
Solusi yang disarankan: re-export dari GEE dengan count pixel valid per layer, jangan isi masked value menjadi 0, dan gabungkan data pembanding seperti SIPONGI/KLHK atau hotspot VIIRS untuk validasi 2023-2025.
</div>
""",
        unsafe_allow_html=True,
    )

elif menu == "Estimasi 2026":
    st.subheader("Estimasi Bulanan Tahun 2026")
    c1, c2, c3, c4, c5 = st.columns(5)
    total_26 = df26["pred_fire_ha"].sum()
    peak_26 = df26.loc[df26["pred_fire_ha"].idxmax()]
    high_months = int((df26["pred_risk_cat"] >= 2).sum())
    avg_fri_26 = df26["fire_risk_index"].mean()
    model_name = df26["risk_model_name"].iloc[0] if "risk_model_name" in df26.columns else "model"
    c1.metric("Total estimasi", f"{total_26:,.0f} ha")
    c2.metric("Puncak risiko", peak_26["month_name"])
    c3.metric("Bulan risiko tinggi", f"{high_months}")
    c4.metric("Rata-rata FRI", f"{avg_fri_26:.3f}")
    c5.metric("Model risiko", str(model_name)[:22])

    fig_26 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Area estimasi", "Kategori risiko", "Probabilitas risiko", "Skor pemicu"],
    )
    fig_26.add_trace(go.Bar(x=df26["month_name"], y=df26["pred_fire_ha"], marker_color=df26["risk_color"], name="Area"), row=1, col=1)
    fig_26.add_trace(go.Scatter(x=df26["month_name"], y=df26["pred_risk_cat"], mode="lines+markers", marker=dict(color=df26["risk_color"], size=10), name="Risiko"), row=1, col=2)
    fig_26.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=RISK_LABELS, row=1, col=2)
    for col, color, label in [
        ("prob_rendah", "#2e7d32", "Rendah"),
        ("prob_sedang", "#f9a825", "Sedang"),
        ("prob_tinggi", "#ef6c00", "Tinggi"),
        ("prob_sgt_tinggi", "#c62828", "Sangat Tinggi"),
    ]:
        if col in df26.columns:
            fig_26.add_trace(go.Bar(x=df26["month_name"], y=df26[col], marker_color=color, name=label), row=2, col=1)
    fig_26.add_trace(go.Bar(x=df26["month_name"], y=df26["trigger_score"], marker_color="#1976d2", name="Trigger"), row=2, col=2)
    fig_26.update_layout(height=640, template="plotly_white", barmode="stack")
    st.plotly_chart(fig_26, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Tabel Estimasi")
        show_cols = [
            "month_name",
            "pred_fire_ha",
            "pred_risk_label",
            "fire_risk_index",
            "trigger_score",
            "mean_lst",
            "total_rain",
            "mean_ndvi",
        ]
        st.dataframe(df26[show_cols].round(3), use_container_width=True, height=430)
    with right:
        st.subheader("Peta Bulan Terpilih")
        month_26 = st.selectbox("Bulan", list(range(1, 13)), format_func=lambda x: MONTH_NAMES_FULL[x - 1], key="map26")
        row_26 = df26[df26["month"].eq(month_26)].iloc[0]
        render_map(row_26, "Estimasi 2026", tile=map_tile)

elif menu == "Tabel Data":
    st.subheader("Tabel Data")
    tab_hist, tab_est, tab_quality, tab_features = st.tabs(
        ["Historis", "Estimasi 2026", "Kualitas Data", "Feature Importance"]
    )
    with tab_hist:
        cols = [
            "date_label",
            "year",
            "month",
            "fire_area_ha",
            "fire_area_model_ha",
            "risk_label",
            "fire_risk_index",
            "total_rain_mm",
            "mean_lst_celsius",
            "mean_ndvi",
            "mean_wind_ms",
            "trigger_score",
            "fri_recomputed_after_imputation",
            "fire_area_missing_suspect",
        ]
        st.dataframe(df_filt[cols], use_container_width=True, height=520)
        st.download_button(
            "Download historis CSV",
            df_filt[cols].to_csv(index=False).encode("utf-8"),
            "riau_fire_historis.csv",
            "text/csv",
        )
    with tab_est:
        st.dataframe(df26, use_container_width=True, height=520)
        st.download_button(
            "Download estimasi 2026 CSV",
            df26.to_csv(index=False).encode("utf-8"),
            "riau_fire_estimasi_2026.csv",
            "text/csv",
        )
    with tab_quality:
        st.dataframe(quality_summary, use_container_width=True, height=420)
    with tab_features:
        st.dataframe(feature_importance, use_container_width=True, height=420)

st.markdown("---")
st.caption(
    f"Data: GEE export {modis_collection_label} / fallback lokal. Model: classifier terbaik hasil tuning + LSTM atau fallback regresi. "
    "Estimasi adalah proyeksi berbasis pola historis dan perlu divalidasi dengan data cuaca serta hotspot aktual."
)
