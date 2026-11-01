# ============================================================
# GEE EXPORT – MODIS Collection 6.1 (MODIS/061)
# Provinsi Riau, statistik bulanan 2015-2025
#
# Jalankan di terminal:
#   python 1_export_gee_modis061.py
#
# Persyaratan:
#   pip install earthengine-api geemap
#   earthengine authenticate   (sekali saja, pakai akun Google)
# ============================================================

import calendar
import sys
from pathlib import Path

import pandas as pd

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api belum terinstall.")
    print("  pip install earthengine-api")
    sys.exit(1)

# ── Autentikasi ──────────────────────────────────────────────────────────────
print("Menginisialisasi Google Earth Engine…")
try:
    ee.Initialize(project=None)
except Exception:
    print("GEE belum terautentikasi. Jalankan: earthengine authenticate")
    print("Mencoba autentikasi interaktif…")
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as exc:
        print(f"Gagal autentikasi: {exc}")
        sys.exit(1)

print("GEE berhasil diinisialisasi.\n")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
START_YEAR = 2015
END_YEAR   = 2025
OUTPUT_CSV = Path(__file__).parent / "riau_fire_monthly_stats_modis061.csv"

# Geometri Provinsi Riau (bounding box konservatif)
RIAU = ee.Geometry.Rectangle([100.0, -1.15, 104.2, 2.35])

SCALE_FIRE = 500    # MCD64A1 native 500 m
SCALE_LST  = 1000   # MOD11A2  native 1 km
SCALE_NDVI = 1000   # MOD13A2  native 1 km
SCALE_RAIN = 5000   # CHIRPS   ~5 km
SCALE_WIND = 11132  # ERA5     ~0.1°

# ── Fungsi bantu ──────────────────────────────────────────────────────────────

def safe_get(image, band, scale, geometry=RIAU, reducer=ee.Reducer.mean()):
    """Ambil nilai tunggal dari image, return None jika gagal."""
    try:
        val = (
            image.select(band)
            .reduceRegion(reducer=reducer, geometry=geometry, scale=scale, maxPixels=1e9, bestEffort=True)
            .get(band)
            .getInfo()
        )
        return float(val) if val is not None else None
    except Exception:
        return None


def month_fire_area_ha(year, month):
    """Area terbakar (ha) dari MCD64A1 – MODIS/061."""
    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{end_day:02d}"

    burned = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(start, end)
        .filterBounds(RIAU)
        .select("BurnDate")
    )
    count = burned.size().getInfo()
    if count == 0:
        return 0.0

    # Setiap pixel terbakar = BurnDate > 0, luas pixel 500m x 500m = 25 ha
    burned_mask = (
        burned.max()
        .gt(0)
        .selfMask()
    )
    pixel_area = ee.Image.pixelArea().divide(10000)  # m² → ha
    area = (
        burned_mask.multiply(pixel_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=RIAU,
            scale=SCALE_FIRE,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("BurnDate")
        .getInfo()
    )
    return float(area) if area else 0.0


def month_lst_celsius(year, month):
    """Mean LST siang hari (°C) dari MOD11A2 – MODIS/061."""
    start = f"{year}-{month:02d}-01"
    days = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{days:02d}"

    col = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .filterBounds(RIAU)
        .select("LST_Day_1km")
    )
    if col.size().getInfo() == 0:
        return None

    # LST unit: Kelvin × 0.02
    lst_k = col.mean().multiply(0.02)
    lst_c = lst_k.subtract(273.15)
    val = (
        lst_c.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=RIAU,
            scale=SCALE_LST,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("LST_Day_1km")
        .getInfo()
    )
    return float(val) if val else None


def month_ndvi(year, month):
    """Mean NDVI dari MOD13A2 – MODIS/061."""
    start = f"{year}-{month:02d}-01"
    days = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{days:02d}"

    col = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate(start, end)
        .filterBounds(RIAU)
        .select("NDVI")
    )
    if col.size().getInfo() == 0:
        return None

    # NDVI unit: × 0.0001
    ndvi = col.mean().multiply(0.0001)
    val = (
        ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=RIAU,
            scale=SCALE_NDVI,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("NDVI")
        .getInfo()
    )
    return float(val) if val else None


def month_rain_mm(year, month):
    """Total curah hujan (mm) dari CHIRPS Daily."""
    start = f"{year}-{month:02d}-01"
    days = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{days:02d}"

    col = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(RIAU)
        .select("precipitation")
    )
    if col.size().getInfo() == 0:
        return None

    total = col.sum()
    val = (
        total.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=RIAU,
            scale=SCALE_RAIN,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("precipitation")
        .getInfo()
    )
    return float(val) if val else None


def month_wind_ms(year, month):
    """Mean kecepatan angin 10m (m/s) dari ERA5-Land Monthly."""
    # ERA5-Land monthly: 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
    start = f"{year}-{month:02d}-01"
    days = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{days:02d}"

    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start, end)
        .filterBounds(RIAU)
    )
    if col.size().getInfo() == 0:
        return None

    img = col.first()
    u10 = safe_get(img, "u_component_of_wind_10m", SCALE_WIND)
    v10 = safe_get(img, "v_component_of_wind_10m", SCALE_WIND)
    if u10 is None or v10 is None:
        return None
    return float((u10**2 + v10**2) ** 0.5)


def compute_fire_risk_index(lst, ndvi, rain, wind):
    import numpy as np
    lst   = lst   if lst   is not None else 30.0
    ndvi  = ndvi  if ndvi  is not None else 0.60
    rain  = rain  if rain  is not None else 220.0
    wind  = wind  if wind  is not None else 2.5
    return float(
        min(max(0.0,
            ((lst - 20) / 20) * 0.35
            + (1 - ndvi) * 0.35
            + (1 - rain / 300) * 0.20
            + (wind / 10) * 0.10
        ), 1.0)
    )


# ── Main export loop ──────────────────────────────────────────────────────────
records = []
total = (END_YEAR - START_YEAR + 1) * 12
done = 0

print(f"Mengekspor {total} bulan data dari GEE (MODIS/061)…")
print(f"Output: {OUTPUT_CSV}\n")

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        done += 1
        label = f"{year}-{month:02d}"
        print(f"[{done:3d}/{total}] {label} … ", end="", flush=True)

        fire_ha = month_fire_area_ha(year, month)
        lst     = month_lst_celsius(year, month)
        ndvi    = month_ndvi(year, month)
        rain    = month_rain_mm(year, month)
        wind    = month_wind_ms(year, month)
        fri     = compute_fire_risk_index(lst, ndvi, rain, wind)

        records.append({
            "date_label"               : label,
            "year"                     : year,
            "month"                    : month,
            "date"                     : f"{label}-01",
            "fire_area_ha"             : round(fire_ha, 3),
            "mean_lst_celsius"         : round(lst,     3) if lst     is not None else None,
            "mean_ndvi"                : round(ndvi,    5) if ndvi    is not None else None,
            "total_rain_mm"            : round(rain,    2) if rain    is not None else None,
            "mean_wind_ms"             : round(wind,    3) if wind    is not None else None,
            "fire_risk_index"          : round(fri,     5),
            "has_fire_event"           : int(fire_ha > 100),
            # ── metadata verifikasi MODIS/061 ──
            "modis_collection_version" : "MODIS/061",
            "source_mcd64a1"           : "MODIS/061/MCD64A1",
            "source_lst"               : "MODIS/061/MOD11A2",
            "source_ndvi"              : "MODIS/061/MOD13A2",
            "source_rain"              : "UCSB-CHG/CHIRPS/DAILY",
            "source_wind"              : "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        })

        status = f"fire={fire_ha:,.0f}ha  lst={lst:.1f}°C  ndvi={ndvi:.3f}  rain={rain:.0f}mm  wind={wind:.1f}m/s  fri={fri:.3f}" \
            if all(v is not None for v in [lst, ndvi, rain, wind]) \
            else "sebagian data null (akan diimputasi di ML script)"
        print(status)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Selesai! {len(df)} baris disimpan ke:\n   {OUTPUT_CSV}")
print(f"\nKolom: {list(df.columns)}")
print(f"Verifikasi MODIS/061: {df['modis_collection_version'].unique()}")
print("\nLangkah selanjutnya:")
print("  python 2_Colab_ML_fire_prediction.py")

