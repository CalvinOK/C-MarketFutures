from __future__ import annotations

"""Build a merged coffee futures modelling dataset.

This script combines the strongest ideas from the existing project files:
- robust daily local market/climate/inventory pipeline from the macro models
- optional richer weekly API/model-panel inputs from data(7).py outputs
- conservative as-of joins to avoid lookahead leakage

Outputs:
    outputs/coffee_model_dataset_merged.csv

The dataset is intentionally mostly *raw / lightly transformed*.
Feature engineering and target creation belong in the training script so that
horizon-specific logic stays in one place.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "coffee_model_dataset_merged.csv"

COFFEE_FILE = LOGDATA_DIR / "CoffeeCData_log_returns.csv"
SOY_FILE = LOGDATA_DIR / "US Soybeans Futures Historical Data_log_returns.csv"
SUGAR_FILE = LOGDATA_DIR / "US Sugar #11 Futures Historical Data_log_returns.csv"
FX_FILE = LOGDATA_DIR / "USD_BRLT Historical Data_log_returns.csv"

CLIMATE_CANDIDATES = [
    LOGDATA_DIR / "coffee_climate_sao_paulo.csv",
    DATA_DIR / "coffee_climate_sao_paulo.csv",
]
ENSO_CANDIDATES = [
    LOGDATA_DIR / "enso.csv",
    DATA_DIR / "enso.csv",
    DATA_DIR / "nino34.long.anom.csv",
    LOGDATA_DIR / "nino34.long.anom.csv",
]
DROUGHT_CANDIDATES = [LOGDATA_DIR / "brazil_drought.csv", DATA_DIR / "brazil_drought.csv"]
FROST_CANDIDATES = [LOGDATA_DIR / "brazil_frost.csv", DATA_DIR / "brazil_frost.csv"]
INVENTORY_CANDIDATES = [
    DATA_DIR / "standardized_inventory.csv",
    LOGDATA_DIR / "standardized_inventory.csv",
    DATA_DIR / "inventory.csv",
    LOGDATA_DIR / "inventory.csv",
]

API_WEEKLY_PANEL_CANDIDATES = [
    DATA_DIR / "kc_model_panel_weekly_asof.csv",
    DATA_DIR / "kc_model_panel_weekly_overlap_only.csv",
    DATA_DIR / "kc_weekly_with_fx_cot_weather_overlap_only.csv",
    DATA_DIR / "kc_weekly_with_fx_cot_weather.csv",
    DATA_DIR / "kc_weekly_with_fx_cot_overlap_only.csv",
    DATA_DIR / "kc_weekly_with_fx_cot.csv",
    DATA_DIR / "kc_weekly_with_fx.csv",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.where(out != -99.99, np.nan)


def normalize_date_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out[date_col] = out[date_col].dt.normalize()
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    return out


def standardize_date_name(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for candidate in ["Date", "date", "DATE", "friday_week", "timestamp"]:
        if candidate in df.columns:
            out = df.copy()
            if candidate != "Date":
                out = out.rename(columns={candidate: "Date"})
            return normalize_date_column(out, "Date"), "Date"
    raise ValueError(f"Could not find a date column in columns={list(df.columns)}")


def clean_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("%", "pct")
        .replace("#", "num")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


# ---------------------------------------------------------------------------
# Local daily loaders
# ---------------------------------------------------------------------------

def load_market_file(path: Path, price_name: str, return_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = normalize_date_column(df, "Date")

    price_col = None
    for candidate in ["Price", "price", "Close", "close"]:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        raise ValueError(f"{path.name} missing price column")

    df[price_name] = safe_numeric(df[price_col])
    if return_name in df.columns:
        df[return_name] = safe_numeric(df[return_name])
    elif "log_return" in df.columns:
        df[return_name] = safe_numeric(df["log_return"])
    else:
        df[return_name] = np.log(df[price_name]).diff()

    return df[["Date", price_name, return_name]].copy()


def load_climate_file() -> pd.DataFrame | None:
    path = first_existing(CLIMATE_CANDIDATES)
    if path is None:
        return None
    df = pd.read_csv(path)
    df, _ = standardize_date_name(df)
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "tempmax" and "tmax" not in df.columns:
            rename_map[col] = "tmax"
        elif lc == "tempmin" and "tmin" not in df.columns:
            rename_map[col] = "tmin"
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in [c for c in ["tmax", "tmin", "rainfall"] if c in df.columns]:
        df[col] = safe_numeric(df[col])
    if "tmax" in df.columns and "tmin" in df.columns:
        df["tavg"] = (df["tmax"] + df["tmin"]) / 2.0
        df["trange"] = df["tmax"] - df["tmin"]
    keep = [c for c in ["Date", "tmax", "tmin", "tavg", "trange", "rainfall"] if c in df.columns]
    return df[keep].copy()


def load_enso_file() -> pd.DataFrame | None:
    path = first_existing(ENSO_CANDIDATES)
    if path is None:
        return None
    df = pd.read_csv(path)
    df, _ = standardize_date_name(df)

    value_col = None
    for candidate in ["enso_index", "anomaly", "value", "ANOM", "NINO3.4", "nino34"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        non_date = [c for c in df.columns if c != "Date"]
        if len(non_date) == 1:
            value_col = non_date[0]
        else:
            return None

    out = df[["Date", value_col]].copy().rename(columns={value_col: "enso_index"})
    out["enso_index"] = safe_numeric(out["enso_index"])
    return out


def load_binary_event_file(candidates: list[Path], value_col_name: str, flag_col_name: str) -> pd.DataFrame | None:
    path = first_existing(candidates)
    if path is None:
        return None
    df = pd.read_csv(path)
    df, _ = standardize_date_name(df)

    value_col = None
    for candidate in [value_col_name, "value", "severity", "index"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        non_date = [c for c in df.columns if c != "Date"]
        if len(non_date) == 1:
            value_col = non_date[0]
        else:
            return None

    out = df[["Date", value_col]].copy().rename(columns={value_col: value_col_name})
    out[value_col_name] = safe_numeric(out[value_col_name])
    out[flag_col_name] = (out[value_col_name].fillna(0.0) > 0).astype(float)
    return out


def load_inventory_file() -> pd.DataFrame | None:
    path = first_existing(INVENTORY_CANDIDATES)
    if path is None:
        return None
    df = pd.read_csv(path)
    df, _ = standardize_date_name(df)
    df.columns = ["Date" if c == "Date" else clean_name(c) for c in df.columns]

    if "report_date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"report_date": "Date"})
        df = normalize_date_column(df, "Date")

    # Aggregate long format inventory reports if needed.
    if {"section", "bags"}.issubset(df.columns):
        section = df["section"].astype(str).str.upper().str.strip()
        bags = safe_numeric(df["bags"])
        tmp = pd.DataFrame({"Date": df["Date"], "section": section, "bags": bags})
        cert = tmp.loc[tmp["section"].str.contains("TOTAL BAGS CERTIFIED", na=False)].groupby("Date", as_index=False)["bags"].sum()
        trans = tmp.loc[tmp["section"].str.contains("TRANSITION BAGS CERTIFIED", na=False)].groupby("Date", as_index=False)["bags"].sum()
        out = cert.rename(columns={"bags": "inventory_certified_bags"})
        out = out.merge(trans.rename(columns={"bags": "inventory_transition_bags"}), on="Date", how="outer")
    else:
        rename = {}
        for candidate in df.columns:
            if candidate in {"certified_bags", "inventory_certified_bags", "total_bags_certified"}:
                rename[candidate] = "inventory_certified_bags"
            elif candidate in {"transition_bags", "inventory_transition_bags", "transition_bags_certified"}:
                rename[candidate] = "inventory_transition_bags"
            elif candidate in {"total_bags", "inventory_total_bags"}:
                rename[candidate] = "inventory_total_bags"
        out = df.rename(columns=rename)

    for col in [c for c in ["inventory_certified_bags", "inventory_transition_bags", "inventory_total_bags"] if c in out.columns]:
        out[col] = safe_numeric(out[col])

    if "inventory_total_bags" not in out.columns:
        cert = out.get("inventory_certified_bags", pd.Series(np.nan, index=out.index))
        trans = out.get("inventory_transition_bags", pd.Series(0.0, index=out.index)).fillna(0.0)
        out["inventory_total_bags"] = cert + trans

    out["inventory_available_flag"] = out[[c for c in ["inventory_certified_bags", "inventory_total_bags"] if c in out.columns]].notna().any(axis=1).astype(float)
    keep = [c for c in ["Date", "inventory_certified_bags", "inventory_transition_bags", "inventory_total_bags", "inventory_available_flag"] if c in out.columns]
    out = out[keep].copy()
    out = normalize_date_column(out, "Date")
    return out


# ---------------------------------------------------------------------------
# Weekly API/model-panel loader
# ---------------------------------------------------------------------------

def load_api_weekly_panel() -> pd.DataFrame | None:
    path = first_existing(API_WEEKLY_PANEL_CANDIDATES)
    if path is None:
        return None

    df = pd.read_csv(path)
    df, _ = standardize_date_name(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # Standardize names.
    df.columns = ["Date" if c == "Date" else clean_name(c) for c in df.columns]

    # Prefix non-date cols to prevent collisions with local daily fields.
    rename = {}
    for c in df.columns:
        if c == "Date":
            continue
        rename[c] = c if c.startswith("api_") else f"api_{c}"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c != "Date":
            df[c] = safe_numeric(df[c]) if df[c].dtype != object else df[c]

    return df


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_daily_base() -> pd.DataFrame:
    coffee = load_market_file(COFFEE_FILE, "coffee_c", "coffee_c_log_return")
    soy = load_market_file(SOY_FILE, "soybeans", "soybeans_log_return") if SOY_FILE.exists() else None
    sugar = load_market_file(SUGAR_FILE, "sugar", "sugar_log_return") if SUGAR_FILE.exists() else None
    fx = load_market_file(FX_FILE, "usd_brl", "usd_brl_log_return") if FX_FILE.exists() else None

    df = coffee.copy()
    for other in [soy, sugar, fx]:
        if other is not None:
            df = df.merge(other, on="Date", how="left")

    # Forward-fill exogenous holidays onto coffee trading dates.
    exog_cols = [c for c in df.columns if c not in {"Date", "coffee_c", "coffee_c_log_return"}]
    if exog_cols:
        df[exog_cols] = df[exog_cols].ffill(limit=5)

    climate = load_climate_file()
    if climate is not None:
        df = df.merge(climate, on="Date", how="left")
        fill_cols = [c for c in ["tmax", "tmin", "tavg", "trange", "rainfall"] if c in df.columns]
        if fill_cols:
            df[fill_cols] = df[fill_cols].ffill(limit=5)

    enso = load_enso_file()
    if enso is not None:
        full_daily = pd.date_range(enso["Date"].min(), enso["Date"].max(), freq="D")
        enso_daily = enso.set_index("Date").reindex(full_daily).ffill(limit=31).rename_axis("Date").reset_index()
        df = df.merge(enso_daily, on="Date", how="left")

    drought = load_binary_event_file(DROUGHT_CANDIDATES, "drought_index", "drought_flag")
    if drought is not None:
        df = df.merge(drought, on="Date", how="left")
        df[["drought_index", "drought_flag"]] = df[["drought_index", "drought_flag"]].ffill(limit=14)

    frost = load_binary_event_file(FROST_CANDIDATES, "frost_severity", "frost_flag")
    if frost is not None:
        df = df.merge(frost, on="Date", how="left")
        df[["frost_severity", "frost_flag"]] = df[["frost_severity", "frost_flag"]].ffill(limit=14)

    inventory = load_inventory_file()
    if inventory is not None:
        inventory = inventory.sort_values("Date")
        df = df.merge(inventory, on="Date", how="left")
        inv_cols = [c for c in inventory.columns if c != "Date"]
        df[inv_cols] = df[inv_cols].ffill(limit=10)
        df["inventory_available_flag"] = df["inventory_available_flag"].fillna(0.0)
    else:
        df["inventory_available_flag"] = 0.0

    return normalize_date_column(df, "Date")


def merge_api_panel_asof(df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> pd.DataFrame:
    # Conservative as-of merge: each coffee date only sees latest published weekly row.
    daily = df_daily.sort_values("Date").copy()
    weekly = df_weekly.sort_values("Date").copy()

    out = pd.merge_asof(daily, weekly, on="Date", direction="backward")

    api_cols = [c for c in weekly.columns if c != "Date"]
    if api_cols:
        out[api_cols] = out[api_cols].ffill(limit=7)
        obs_flag = out[api_cols].notna().any(axis=1).astype(float)
        out["api_panel_available_flag"] = obs_flag
        last_obs_date = out["Date"].where(obs_flag == 1).ffill()
        out["days_since_api_panel"] = (out["Date"] - last_obs_date).dt.days.fillna(999).astype(float)
    else:
        out["api_panel_available_flag"] = 0.0
        out["days_since_api_panel"] = 999.0

    return out


def add_light_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "coffee_c" in out.columns:
        out["coffee_log_price"] = np.log(out["coffee_c"].where(out["coffee_c"] > 0))

    for col in ["tmax", "tmin", "rainfall", "drought_index", "frost_severity", "enso_index"]:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(float)

    if "api_spec_net" in out.columns:
        out["api_spec_net_missing_flag"] = out["api_spec_net"].isna().astype(float)
    if "api_curve_slope_1_6" in out.columns:
        out["api_curve_missing_flag"] = out["api_curve_slope_1_6"].isna().astype(float)

    return out


def build_dataset() -> pd.DataFrame:
    df = merge_daily_base()
    weekly = load_api_weekly_panel()
    if weekly is not None:
        df = merge_api_panel_asof(df, weekly)
    else:
        df["api_panel_available_flag"] = 0.0
        df["days_since_api_panel"] = 999.0

    df = add_light_derived_columns(df)
    df = normalize_date_column(df, "Date")
    return df


def main() -> None:
    df = build_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved merged dataset:")
    print(f"  {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns):,}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    show_cols = [
        "coffee_c", "coffee_c_log_return", "soybeans", "sugar", "usd_brl",
        "tmax", "rainfall", "enso_index", "inventory_certified_bags",
        "api_price", "api_brl_per_usd", "api_spec_net", "api_curve_slope_1_6",
        "api_panel_available_flag",
    ]
    existing = [c for c in show_cols if c in df.columns]
    if existing:
        print("\nNon-null counts for key columns:")
        print(df[existing].notna().sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
