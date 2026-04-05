from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGDATA_DIR.mkdir(parents=True, exist_ok=True)

# Coffee reference location in Brazil:
# CEPEA/ESALQ arabica index is delivered in São Paulo (capital)
TARGET_NAME = "Sao Paulo, SP, Brazil"
TARGET_LAT = -23.5505
TARGET_LON = -46.6333

# NetCDF inputs expected in data/ only if rebuild is needed
TMAX_FILE = DATA_DIR / "Tmax.nc"
TMIN_FILE = DATA_DIR / "Tmin.nc"
RAIN_FILE = DATA_DIR / "rainfall.nc"

# Existing / output climate CSVs
OUTPUT_CSV_DATA = DATA_DIR / "coffee_climate_sao_paulo.csv"
OUTPUT_CSV_LOGDATA = LOGDATA_DIR / "coffee_climate_sao_paulo.csv"

# Derived event CSVs expected by the feature builder
DROUGHT_CSV_DATA = DATA_DIR / "brazil_drought.csv"
DROUGHT_CSV_LOGDATA = LOGDATA_DIR / "brazil_drought.csv"
FROST_CSV_DATA = DATA_DIR / "brazil_frost.csv"
FROST_CSV_LOGDATA = LOGDATA_DIR / "brazil_frost.csv"

# Event heuristics
DROUGHT_INDEX_THRESHOLD = 1.0
FROST_TEMP_C_THRESHOLD = 2.0


# ============================================================
# HELPERS
# ============================================================

def find_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"Could not find any of these coordinates/variables: {candidates}")


def find_time_name(ds):
    return find_coord_name(ds, ["time", "Time", "date", "Date", "datetime"])


def find_lat_name(ds):
    return find_coord_name(ds, ["lat", "latitude", "Latitude", "LAT", "nav_lat"])


def find_lon_name(ds):
    return find_coord_name(ds, ["lon", "longitude", "Longitude", "LON", "nav_lon"])


def guess_data_var(ds):
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("No data variables found in dataset.")
    return data_vars[0]


def normalize_longitude(lon_value, ds_lon_values):
    lon_arr = np.asarray(ds_lon_values)
    lon_min = np.nanmin(lon_arr)
    lon_max = np.nanmax(lon_arr)

    if lon_min >= 0 and lon_max > 180 and lon_value < 0:
        return lon_value % 360
    return lon_value


def extract_series(nc_file: Path, value_name: str) -> pd.DataFrame:
    if not nc_file.exists():
        raise FileNotFoundError(f"Missing NetCDF file: {nc_file}")

    try:
        ds = xr.open_dataset(nc_file, engine="netcdf4")
    except Exception:
        ds = xr.open_dataset(nc_file, engine="h5netcdf")

    lat_name = find_lat_name(ds)
    lon_name = find_lon_name(ds)
    time_name = find_time_name(ds)
    data_var = guess_data_var(ds)

    target_lon = normalize_longitude(TARGET_LON, ds[lon_name].values)

    da = ds[data_var].sel(
        {lat_name: TARGET_LAT, lon_name: target_lon},
        method="nearest"
    )

    df = da.to_dataframe(name=value_name).reset_index()
    df[time_name] = pd.to_datetime(df[time_name], errors="coerce")
    df = df.rename(columns={time_name: "Date"})
    df = df[["Date", value_name]].copy()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return df


def add_pct_change(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_change_pct"] = df[col].pct_change() * 100
    return df


def finalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_temperature_units(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    # Heuristic: values like 290 imply Kelvin.
    if float(s.median()) > 100:
        return s - 273.15
    return s


def normalize_rainfall_units(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s
    # Heuristic: many gridded products store rainfall as meters/day.
    if float(non_na.quantile(0.95)) < 1.0:
        return s * 1000.0
    return s


def load_existing_or_build_climate() -> pd.DataFrame:
    if OUTPUT_CSV_DATA.exists():
        print("Climate CSV already exists in data/.")
        df = pd.read_csv(OUTPUT_CSV_DATA)
        df = finalize_date_index(df)
        df.to_csv(OUTPUT_CSV_LOGDATA, index=False)
        print(f"Copied to: {OUTPUT_CSV_LOGDATA}")
        return df

    if OUTPUT_CSV_LOGDATA.exists():
        print("Climate CSV already exists in logdata/.")
        df = pd.read_csv(OUTPUT_CSV_LOGDATA)
        df = finalize_date_index(df)
        df.to_csv(OUTPUT_CSV_DATA, index=False)
        print(f"Copied to: {OUTPUT_CSV_DATA}")
        return df

    missing = [p for p in [TMAX_FILE, TMIN_FILE, RAIN_FILE] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "No existing climate CSV was found, and the following NetCDF inputs are missing: "
            f"{missing_str}"
        )

    tmax_df = extract_series(TMAX_FILE, "tmax")
    tmin_df = extract_series(TMIN_FILE, "tmin")
    rain_df = extract_series(RAIN_FILE, "rainfall")

    df = tmax_df.merge(tmin_df, on="Date", how="outer")
    df = df.merge(rain_df, on="Date", how="outer")
    df = finalize_date_index(df)

    return df


def prepare_climate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = finalize_date_index(df)
    out = coerce_numeric(out, ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"])

    if "tmax" in out.columns:
        out["tmax"] = normalize_temperature_units(out["tmax"])
    if "tmin" in out.columns:
        out["tmin"] = normalize_temperature_units(out["tmin"])
    if "rainfall" in out.columns:
        out["rainfall"] = normalize_rainfall_units(out["rainfall"])

    for col in ["tmax", "tmin"]:
        if col in out.columns:
            out = add_pct_change(out, col)

    return out


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - mean) / std


def build_drought_csv(climate_df: pd.DataFrame) -> pd.DataFrame:
    df = climate_df[["Date"]].copy()

    rainfall = pd.to_numeric(climate_df.get("rainfall"), errors="coerce")
    tmax = pd.to_numeric(climate_df.get("tmax"), errors="coerce")
    tmin = pd.to_numeric(climate_df.get("tmin"), errors="coerce")

    df["rainfall_30d"] = rainfall.rolling(30, min_periods=10).sum()
    df["rainfall_90d"] = rainfall.rolling(90, min_periods=30).sum()
    df["tavg"] = (tmax + tmin) / 2.0
    df["tavg_30d"] = df["tavg"].rolling(30, min_periods=10).mean()

    # Monthly baseline reduces seasonality distortion.
    month = pd.to_datetime(df["Date"]).dt.month
    monthly_rain_30_baseline = df.groupby(month)["rainfall_30d"].transform("median")
    monthly_tavg_30_baseline = df.groupby(month)["tavg_30d"].transform("median")

    df["rainfall_30d_pct_of_monthly_median"] = df["rainfall_30d"] / monthly_rain_30_baseline.replace(0, np.nan)
    df["heat_anomaly"] = df["tavg_30d"] - monthly_tavg_30_baseline
    df["dryness_anomaly"] = monthly_rain_30_baseline - df["rainfall_30d"]

    heat_score = zscore(df["heat_anomaly"]).clip(lower=0)
    dry_score = zscore(df["dryness_anomaly"]).clip(lower=0)
    df["drought_index"] = (0.6 * dry_score + 0.4 * heat_score).fillna(0.0)

    severe_dry = df["rainfall_30d_pct_of_monthly_median"] < 0.75
    elevated_index = df["drought_index"] >= DROUGHT_INDEX_THRESHOLD
    df["drought_flag"] = (severe_dry | elevated_index).astype(float)

    return finalize_date_index(df)


def build_frost_csv(climate_df: pd.DataFrame) -> pd.DataFrame:
    df = climate_df[["Date"]].copy()
    tmin = pd.to_numeric(climate_df.get("tmin"), errors="coerce")

    df["tmin_c"] = tmin
    df["frost_severity"] = (FROST_TEMP_C_THRESHOLD - df["tmin_c"]).clip(lower=0)
    df["frost_flag"] = (df["tmin_c"] <= FROST_TEMP_C_THRESHOLD).astype(float)
    df["cold_spell_3d_min"] = df["tmin_c"].rolling(3, min_periods=1).min()

    # A softer continuous feature that is still useful even when frost events are rare.
    df["cold_stress_index"] = zscore(-df["tmin_c"]).clip(lower=0).fillna(0.0)

    return finalize_date_index(df)


def save_dual(df: pd.DataFrame, data_path: Path, logdata_path: Path, label: str) -> None:
    df.to_csv(data_path, index=False)
    df.to_csv(logdata_path, index=False)
    print(f"Saved {label} CSV to: {data_path}")
    print(f"Saved {label} CSV to: {logdata_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using target location: {TARGET_NAME} ({TARGET_LAT}, {TARGET_LON})")

    climate_raw = load_existing_or_build_climate()
    climate_df = prepare_climate_dataframe(climate_raw)

    drought_df = build_drought_csv(climate_df)
    frost_df = build_frost_csv(climate_df)

    save_dual(climate_df, OUTPUT_CSV_DATA, OUTPUT_CSV_LOGDATA, "climate")
    save_dual(drought_df, DROUGHT_CSV_DATA, DROUGHT_CSV_LOGDATA, "drought")
    save_dual(frost_df, FROST_CSV_DATA, FROST_CSV_LOGDATA, "frost")

    print("\nClimate sample:")
    print(climate_df.tail(5).to_string(index=False))

    print("\nDrought sample:")
    print(drought_df.tail(5).to_string(index=False))

    print("\nFrost sample:")
    print(frost_df.tail(5).to_string(index=False))

    print("\nNotes:")
    print("- tmax/tmin are normalized to Celsius when the source appears to be Kelvin.")
    print("- rainfall is normalized to mm/day when the source appears to be meters/day.")
    print("- drought.csv and frost.csv are derived event datasets built from the climate series.")
    print("- These event labels are heuristic proxies unless you replace them with a dedicated event source.")


if __name__ == "__main__":
    main()
