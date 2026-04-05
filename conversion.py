from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

# Coffee reference location in Brazil:
# CEPEA/ESALQ arabica index is delivered in São Paulo (capital)
TARGET_NAME = "Sao Paulo, SP, Brazil"
TARGET_LAT = -23.5505
TARGET_LON = -46.6333

# Replace these with your actual file paths
TMAX_FILE = "Tmax.nc"
TMIN_FILE = "Tmin.nc"
RAIN_FILE = "rainfall.nc"

OUTPUT_CSV = "coffee_climate_sao_paulo.csv"


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
    """
    If the dataset has one main data variable, use it.
    Otherwise prefer the first non-coordinate variable.
    """
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("No data variables found in dataset.")
    return data_vars[0]


def normalize_longitude(lon_value, ds_lon_values):
    """
    If the dataset longitudes are 0..360 and the target lon is -180..180,
    convert target lon to 0..360.
    """
    lon_arr = np.asarray(ds_lon_values)
    lon_min = np.nanmin(lon_arr)
    lon_max = np.nanmax(lon_arr)

    if lon_min >= 0 and lon_max > 180 and lon_value < 0:
        return lon_value % 360
    return lon_value


def extract_series(nc_file, value_name):
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
    df[time_name] = pd.to_datetime(df[time_name])
    df = df.rename(columns={time_name: "Date"})
    df = df[["Date", value_name]].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_pct_change(df, col):
    df[f"{col}_change_pct"] = df[col].pct_change() * 100
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    # Extract nearest-grid-cell series
    tmax_df = extract_series(TMAX_FILE, "tmax")
    tmin_df = extract_series(TMIN_FILE, "tmin")
    rain_df = extract_series(RAIN_FILE, "rainfall")

    # Merge on Date
    df = tmax_df.merge(tmin_df, on="Date", how="outer")
    df = df.merge(rain_df, on="Date", how="outer")

    df = df.sort_values("Date").reset_index(drop=True)

    # Add daily % change columns for temperature only
    for col in ["tmax", "tmin"]:
        if col in df.columns:
            df = add_pct_change(df, col)

    # Optional: reorder columns
    desired_order = [
        "Date",
        "tmax",
        "tmax_change_pct",
        "tmin",
        "tmin_change_pct",
        "rainfall",
    ]
    df = df[[col for col in desired_order if col in df.columns]]

    # Optional: round numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Reference location: {TARGET_NAME}")
    print(f"Latitude:  {TARGET_LAT}")
    print(f"Longitude: {TARGET_LON}")
    print(f"Saved CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()