from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
LOGDATA_DIR = BASE_DIR / "logdata"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGDATA_DIR.mkdir(parents=True, exist_ok=True)

COFFEE_FILE = LOGDATA_DIR / "CoffeeCData_log_returns.csv"
SOY_FILE = LOGDATA_DIR / "US Soybeans Futures Historical Data_log_returns.csv"
SUGAR_FILE = LOGDATA_DIR / "US Sugar #11 Futures Historical Data_log_returns.csv"
FX_FILE = LOGDATA_DIR / "USD_BRLT Historical Data_log_returns.csv"

CLIMATE_CANDIDATES = [
    LOGDATA_DIR / "coffee_climate_sao_paulo.csv",
    DATA_DIR / "coffee_climate_sao_paulo.csv",
]

# Prefer already-cleaned ENSO, then raw NOAA, in either data/ or logdata/
ENSO_CANDIDATES = [
    LOGDATA_DIR / "enso.csv",
    DATA_DIR / "enso.csv",
    DATA_DIR / "nino34.long.anom.csv",
    LOGDATA_DIR / "nino34.long.anom.csv",
]

# Optional event data can live in either folder
DROUGHT_CANDIDATES = [
    LOGDATA_DIR / "brazil_drought.csv",
    DATA_DIR / "brazil_drought.csv",
]

FROST_CANDIDATES = [
    LOGDATA_DIR / "brazil_frost.csv",
    DATA_DIR / "brazil_frost.csv",
]

CANONICAL_ENSO_FILE = LOGDATA_DIR / "enso.csv"
CANONICAL_DROUGHT_FILE = LOGDATA_DIR / "brazil_drought.csv"
CANONICAL_FROST_FILE = LOGDATA_DIR / "brazil_frost.csv"

# ICE certified stock / inventory data
INVENTORY_CANDIDATES = [
    DATA_DIR / "standardized_inventory.csv",
    LOGDATA_DIR / "standardized_inventory.csv",
    DATA_DIR / "inventory.csv",
    LOGDATA_DIR / "inventory.csv",
]

OUTPUT_FILE = OUTPUT_DIR / "coffee_feature_dataset_macro.csv"


# ============================================================
# HELPERS
# ============================================================


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["Date"] = out["Date"].dt.normalize()
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


def normalize_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
    return out


def pick_value_column(df: pd.DataFrame, preferred: list[str]) -> str | None:
    lower_lookup = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in lower_lookup:
            return lower_lookup[name.lower()]

    non_date_cols = [c for c in df.columns if c.lower() != "date"]
    if len(non_date_cols) == 1:
        return non_date_cols[0]
    return None


def canonicalize_binary(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    return (numeric.fillna(0.0) > 0).astype(float)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = safe_numeric(numerator)
    den = safe_numeric(denominator).replace(0, np.nan)
    return num / den


def load_market_log_file(file_path: Path, value_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")

    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        raise ValueError(f"{file_path.name} must contain a Date column")
    if "Price" not in df.columns:
        raise ValueError(f"{file_path.name} must contain a Price column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = safe_numeric(df["Price"])

    log_return_col = None
    for candidate in ["log_return", f"{value_name}_log_return", f"{value_name}_return_log"]:
        if candidate in df.columns:
            log_return_col = candidate
            break

    if log_return_col is None:
        df = df.sort_values("Date").reset_index(drop=True)
        df["log_return"] = np.log(df["Price"]).diff()
        log_return_col = "log_return"

    df[log_return_col] = safe_numeric(df[log_return_col])

    out = (
        df[["Date", "Price", log_return_col]]
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
        .rename(columns={"Price": value_name, log_return_col: f"{value_name}_log_return"})
    )
    return out


def load_climate_file() -> pd.DataFrame:
    climate_path = first_existing(CLIMATE_CANDIDATES)
    if climate_path is None:
        raise FileNotFoundError(
            "Missing climate file. Expected one of: "
            + ", ".join(str(p) for p in CLIMATE_CANDIDATES)
        )

    df = pd.read_csv(climate_path)
    if "Date" not in df.columns:
        raise ValueError(f"{climate_path.name} must contain a Date column")

    df = normalize_date_column(df)
    df = normalize_numeric_columns(df, ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"])
    return df


def load_enso_file() -> tuple[pd.DataFrame | None, str | None]:
    for path in ENSO_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "Date" not in df.columns:
            continue

        value_col = pick_value_column(df, ["enso_index", "anomaly", "nino34", "value"])
        if value_col is None:
            continue

        out = df[["Date", value_col]].copy()
        out = normalize_date_column(out)
        out[value_col] = safe_numeric(out[value_col]).replace(-99.99, np.nan)
        out = out.rename(columns={value_col: "enso_index"})
        out.to_csv(CANONICAL_ENSO_FILE, index=False)
        return out, None

    return None, "Missing optional ENSO file in data/ or logdata/"


def load_optional_event_file(
    candidates: list[Path],
    canonical_name: str,
    alias_columns: list[str],
    *,
    binary: bool = False,
) -> tuple[pd.DataFrame | None, str | None]:
    path = first_existing(candidates)
    if path is None:
        return None, f"Missing optional file: one of {[p.name for p in candidates]}"

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        return None, f"{path.name} must contain a Date column"

    value_col = pick_value_column(df, alias_columns)
    if value_col is None:
        return None, (
            f"{path.name} must contain one of {alias_columns} or exactly one non-Date value column"
        )

    out = df[["Date", value_col]].copy()
    out = normalize_date_column(out)
    if binary:
        out[value_col] = canonicalize_binary(out[value_col])
    else:
        out[value_col] = safe_numeric(out[value_col])

    out = out.rename(columns={value_col: canonical_name})

    canonical_path = LOGDATA_DIR / path.name
    out.to_csv(canonical_path, index=False)
    return out, None


def load_inventory_file() -> tuple[pd.DataFrame | None, str | None]:
    """
    Load the standardized_inventory.csv and pivot it into a daily time-series.

    The raw file has columns: report_date, section, country, warehouse, bags, source_file
    We aggregate at the report_date level to produce:
        - inventory_total_bags          : all certified + transition bags, all countries, warehouse=Total rows only
        - inventory_certified_bags      : section == 'TOTAL BAGS CERTIFIED', warehouse == 'Total'
        - inventory_transition_bags     : section == 'TRANSITION BAGS CERTIFIED', warehouse == 'Total'
        - inventory_brazil_bags         : certified bags for Brazil only
        - inventory_brazil_share        : Brazil bags / total certified bags
        - inventory_certified_log_chg   : log change in certified bags week-over-week
    """
    path = first_existing(INVENTORY_CANDIDATES)
    if path is None:
        return None, (
            "Missing optional inventory file. Expected one of: "
            + ", ".join(str(p) for p in INVENTORY_CANDIDATES)
        )

    df = pd.read_csv(path)

    required = {"report_date", "section", "country", "warehouse", "bags"}
    if not required.issubset(df.columns):
        return None, (
            f"{path.name} is missing columns {required - set(df.columns)}. "
            "Expected: report_date, section, country, warehouse, bags."
        )

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["bags"] = safe_numeric(df["bags"])
    df = df.dropna(subset=["report_date", "bags"])

    # ── aggregate certified stock (Total warehouse rows only) ──────────────────
    certified_mask = (
        df["section"].str.upper().str.contains("TOTAL BAGS CERTIFIED", na=False)
        & (df["warehouse"].str.strip().str.upper() == "TOTAL")
    )
    certified = (
        df[certified_mask]
        .groupby("report_date", as_index=False)["bags"]
        .sum()
        .rename(columns={"bags": "inventory_certified_bags", "report_date": "Date"})
    )

    # ── aggregate transition stock ─────────────────────────────────────────────
    transition_mask = (
        df["section"].str.upper().str.contains("TRANSITION BAGS CERTIFIED", na=False)
        & (df["warehouse"].str.strip().str.upper() == "TOTAL")
    )
    transition = (
        df[transition_mask]
        .groupby("report_date", as_index=False)["bags"]
        .sum()
        .rename(columns={"bags": "inventory_transition_bags", "report_date": "Date"})
    )

    # ── Brazil-only certified bags ─────────────────────────────────────────────
    brazil_mask = (
        certified_mask
        & (df["country"].str.strip().str.upper() == "BRAZIL")
    )
    brazil = (
        df[brazil_mask]
        .groupby("report_date", as_index=False)["bags"]
        .sum()
        .rename(columns={"bags": "inventory_brazil_bags", "report_date": "Date"})
    )

    # ── merge into a single date-indexed frame ─────────────────────────────────
    out = certified.merge(transition, on="Date", how="outer")
    out = out.merge(brazil, on="Date", how="outer")

    out = normalize_date_column(out)
    out["inventory_certified_bags"] = safe_numeric(out["inventory_certified_bags"])
    out["inventory_transition_bags"] = safe_numeric(out["inventory_transition_bags"]).fillna(0.0)
    out["inventory_brazil_bags"] = safe_numeric(out["inventory_brazil_bags"]).fillna(0.0)

    out["inventory_total_bags"] = out["inventory_certified_bags"] + out["inventory_transition_bags"]

    cert = out["inventory_certified_bags"].replace(0, np.nan)
    out["inventory_brazil_share"] = out["inventory_brazil_bags"] / cert
    out["inventory_certified_log_chg"] = np.log(cert).diff()

    return out, None


def add_inventory_features(df: pd.DataFrame) -> None:
    """
    Derive rolling and regime features from the inventory time-series columns
    already merged into df.  All features are built defensively — if a column
    is absent the function simply skips it.
    """
    if "inventory_certified_bags" not in df.columns:
        return

    cert = safe_numeric(df["inventory_certified_bags"])
    total = safe_numeric(df.get("inventory_total_bags", cert))
    brazil_share = safe_numeric(df.get("inventory_brazil_share"))
    log_chg = safe_numeric(df.get("inventory_certified_log_chg"))

    # rolling levels
    for window in [4, 13, 26]:          # ~1 month, ~1 quarter, ~6 months (report is weekly)
        days = window * 5               # approximate business-day equivalent for rolling
        df[f"inventory_certified_roll_mean_{window}w"] = cert.rolling(days, min_periods=2).mean()
        df[f"inventory_certified_roll_std_{window}w"] = cert.rolling(days, min_periods=2).std()

    # level-vs-rolling-mean ratios (supply surplus/deficit signal)
    roll_mean_13 = safe_numeric(df.get("inventory_certified_roll_mean_13w")).replace(0, np.nan)
    roll_mean_26 = safe_numeric(df.get("inventory_certified_roll_mean_26w")).replace(0, np.nan)
    df["inventory_cert_vs_13w_mean"] = cert / roll_mean_13 - 1.0
    df["inventory_cert_vs_26w_mean"] = cert / roll_mean_26 - 1.0

    # regime flags
    df["inventory_low_supply_flag"] = (
        safe_numeric(df.get("inventory_cert_vs_26w_mean", pd.Series(np.nan, index=df.index))) < -0.05
    ).astype(float)
    df["inventory_high_supply_flag"] = (
        safe_numeric(df.get("inventory_cert_vs_26w_mean", pd.Series(np.nan, index=df.index))) > 0.05
    ).astype(float)

    # momentum of certified stock (rate of drawdown/build)
    if log_chg is not None:
        df["inventory_log_chg_roll_4w"] = log_chg.rolling(20, min_periods=2).sum()   # ~4 week accumulation
        df["inventory_log_chg_roll_13w"] = log_chg.rolling(65, min_periods=2).sum()  # ~13 week accumulation
        df["inventory_drawdown_flag"] = (safe_numeric(df["inventory_log_chg_roll_4w"]) < 0).astype(float)

    # Brazil share features
    if brazil_share is not None:
        df["inventory_brazil_share_roll_13w"] = brazil_share.rolling(65, min_periods=2).mean()
        df["inventory_brazil_dominant_flag"] = (brazil_share > 0.50).astype(float)

    # total (certified + transition) relative to 26w baseline
    df["inventory_total_roll_mean_26w"] = total.rolling(130, min_periods=2).mean()
    total_base = safe_numeric(df["inventory_total_roll_mean_26w"]).replace(0, np.nan)
    df["inventory_total_vs_26w_mean"] = total / total_base - 1.0

    # lags on the key supply signal (weekly reports -> lag in 5-day steps)
    for lag in [5, 10, 20, 60]:
        if "inventory_certified_bags" in df.columns:
            df[f"inventory_certified_bags_lag_{lag}"] = cert.shift(lag)
        if "inventory_certified_log_chg" in df.columns:
            df[f"inventory_certified_log_chg_lag_{lag}"] = log_chg.shift(lag)
        if "inventory_cert_vs_26w_mean" in df.columns:
            df[f"inventory_cert_vs_26w_mean_lag_{lag}"] = safe_numeric(df["inventory_cert_vs_26w_mean"]).shift(lag)


def add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = safe_numeric(df[col]).shift(lag)


def add_rolling_stats(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    series = safe_numeric(df[col])
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = series.rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = series.rolling(w).std()
        df[f"{col}_roll_min_{w}"] = series.rolling(w).min()
        df[f"{col}_roll_max_{w}"] = series.rolling(w).max()


def add_price_trend_features(df: pd.DataFrame, price_col: str, prefix: str) -> None:
    price = safe_numeric(df[price_col])

    for w in [5, 10, 20, 50, 100]:
        df[f"{prefix}_sma_{w}"] = price.rolling(w).mean()

    df[f"{prefix}_price_vs_sma_20"] = price / df[f"{prefix}_sma_20"] - 1
    df[f"{prefix}_price_vs_sma_50"] = price / df[f"{prefix}_sma_50"] - 1
    df[f"{prefix}_sma_20_vs_50"] = df[f"{prefix}_sma_20"] / df[f"{prefix}_sma_50"] - 1
    df[f"{prefix}_sma_50_vs_100"] = df[f"{prefix}_sma_50"] / df[f"{prefix}_sma_100"] - 1

    df[f"{prefix}_above_sma_20"] = (price > df[f"{prefix}_sma_20"]).astype(float)
    df[f"{prefix}_above_sma_50"] = (price > df[f"{prefix}_sma_50"]).astype(float)

    df[f"{prefix}_roc_5"] = price / price.shift(5) - 1
    df[f"{prefix}_roc_20"] = price / price.shift(20) - 1
    df[f"{prefix}_roc_60"] = price / price.shift(60) - 1


def add_seasonality_features(df: pd.DataFrame) -> None:
    date = pd.to_datetime(df["Date"])
    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["quarter"] = date.dt.quarter
    df["day_of_week"] = date.dt.dayofweek
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

    df["brazil_harvest_flag"] = df["month"].isin([5, 6, 7, 8, 9]).astype(float)
    df["brazil_flowering_flag"] = df["month"].isin([9, 10, 11]).astype(float)
    df["brazil_offseason_flag"] = df["month"].isin([12, 1, 2, 3]).astype(float)


def add_relative_value_features(df: pd.DataFrame) -> None:
    for col in ["coffee_c", "sugar", "soybeans", "usd_brl"]:
        df[col] = safe_numeric(df[col])

    df["coffee_sugar_ratio"] = safe_divide(df["coffee_c"], df["sugar"])
    df["coffee_soy_ratio"] = safe_divide(df["coffee_c"], df["soybeans"])
    df["coffee_fx_ratio"] = safe_divide(df["coffee_c"], df["usd_brl"])

    df["coffee_minus_sugar"] = df["coffee_c"] - df["sugar"]
    df["coffee_minus_soybeans"] = df["coffee_c"] - df["soybeans"]

    df["coffee_minus_sugar_log_return"] = df["coffee_c_log_return"] - df["sugar_log_return"]
    df["coffee_minus_soybeans_log_return"] = df["coffee_c_log_return"] - df["soybeans_log_return"]
    df["coffee_minus_fx_log_return"] = df["coffee_c_log_return"] - df["usd_brl_log_return"]


def add_volatility_features(df: pd.DataFrame) -> None:
    for base in ["coffee_c_log_return", "sugar_log_return", "soybeans_log_return", "usd_brl_log_return"]:
        if base in df.columns:
            add_rolling_stats(df, base, [5, 10, 20, 60])

    if "coffee_c_log_return" in df.columns:
        coffee_ret = safe_numeric(df["coffee_c_log_return"])
        df["coffee_abs_return_5d"] = coffee_ret.abs().rolling(5).mean()
        df["coffee_abs_return_20d"] = coffee_ret.abs().rolling(20).mean()
        df["coffee_abs_return_60d"] = coffee_ret.abs().rolling(60).mean()
        df["coffee_vol_regime_ratio_20_60"] = safe_divide(
            df["coffee_c_log_return_roll_std_20"],
            df["coffee_c_log_return_roll_std_60"],
        )
        df["coffee_high_vol_regime"] = (
            safe_numeric(df["coffee_vol_regime_ratio_20_60"]) > 1.25
        ).astype(float)


def add_climate_equation_features(df: pd.DataFrame) -> None:
    if "tmax" in df.columns and "tmin" in df.columns:
        df["tavg"] = (safe_numeric(df["tmax"]) + safe_numeric(df["tmin"])) / 2.0
        df["trange"] = safe_numeric(df["tmax"]) - safe_numeric(df["tmin"])

        add_lags(df, "tavg", [1, 3, 5])
        add_lags(df, "trange", [1, 3, 5])

        df["tavg_roll_mean_5"] = safe_numeric(df["tavg"]).rolling(5).mean()
        df["tavg_roll_std_5"] = safe_numeric(df["tavg"]).rolling(5).std()
        df["trange_roll_mean_5"] = safe_numeric(df["trange"]).rolling(5).mean()
        df["trange_roll_std_5"] = safe_numeric(df["trange"]).rolling(5).std()

        tmax = safe_numeric(df["tmax"])
        tmin = safe_numeric(df["tmin"])
        df["heat_stress_daily"] = (tmax > 30.0).astype(float)
        df["severe_heat_daily"] = (tmax > 32.0).astype(float)
        df["cold_stress_daily"] = (tmin < 5.0).astype(float)
        df["frost_severity_temp"] = np.maximum(0.0, 2.0 - tmin)
        df["heat_stress_30d"] = df["heat_stress_daily"].rolling(30).mean()
        df["heat_stress_90d"] = df["heat_stress_daily"].rolling(90).mean()
        df["severe_heat_90d"] = df["severe_heat_daily"].rolling(90).mean()
        df["cold_stress_30d"] = df["cold_stress_daily"].rolling(30).mean()
        df["frost_severity_temp_30d"] = safe_numeric(df["frost_severity_temp"]).rolling(30).sum()
        df["frost_severity_temp_90d"] = safe_numeric(df["frost_severity_temp"]).rolling(90).sum()

    if "rainfall" in df.columns:
        rain = safe_numeric(df["rainfall"])
        df["rainfall_rolling_7"] = rain.rolling(7).sum()
        df["rainfall_rolling_30"] = rain.rolling(30).sum()
        df["rainfall_rolling_90"] = rain.rolling(90).sum()
        df["rainfall_rolling_180"] = rain.rolling(180).sum()
        df["rainfall_mean_30"] = rain.rolling(30).mean()
        df["rainfall_mean_90"] = rain.rolling(90).mean()
        df["rainfall_30_vs_90_regime"] = safe_divide(df["rainfall_mean_30"], df["rainfall_mean_90"]) - 1.0
        df["rainfall_90_vs_180_regime"] = safe_divide(
            df["rainfall_rolling_90"],
            df["rainfall_rolling_180"] / 2.0,
        ) - 1.0
        add_lags(df, "rainfall", [1, 3, 5])


def add_core_lags(df: pd.DataFrame) -> None:
    for base in ["coffee_c_log_return", "sugar_log_return", "soybeans_log_return", "usd_brl_log_return"]:
        if base in df.columns:
            add_lags(df, base, [1, 2, 3, 5, 10, 20])

    for base in ["tmax", "tmin", "tmax_change_pct", "tmin_change_pct"]:
        if base in df.columns:
            add_lags(df, base, [1, 3, 5])


def add_macro_state_features(df: pd.DataFrame) -> None:
    coffee_price = safe_numeric(df["coffee_c"])
    fx_price = safe_numeric(df["usd_brl"])

    for window in [20, 60, 120]:
        df[f"coffee_trend_{window}d"] = coffee_price.pct_change(window)
        df[f"usd_brl_trend_{window}d"] = fx_price.pct_change(window)

    df["coffee_drawdown_60d"] = safe_divide(
        coffee_price,
        coffee_price.rolling(60).max(),
    ) - 1.0
    df["coffee_drawdown_120d"] = safe_divide(
        coffee_price,
        coffee_price.rolling(120).max(),
    ) - 1.0
    df["coffee_distance_from_120d_low"] = safe_divide(
        coffee_price,
        coffee_price.rolling(120).min(),
    ) - 1.0

    df["coffee_macro_uptrend_flag"] = (safe_numeric(df["coffee_trend_120d"]) > 0).astype(float)
    df["coffee_macro_downtrend_flag"] = (safe_numeric(df["coffee_trend_120d"]) < 0).astype(float)
    df["fx_strengthening_flag"] = (safe_numeric(df["usd_brl_trend_60d"]) < 0).astype(float)
    df["fx_weakening_flag"] = (safe_numeric(df["usd_brl_trend_60d"]) > 0).astype(float)

    if "enso_index" in df.columns:
        enso = safe_numeric(df["enso_index"])
        df["enso_roll_mean_30"] = enso.rolling(30).mean()
        df["enso_roll_mean_90"] = enso.rolling(90).mean()
        df["enso_positive_regime"] = (df["enso_roll_mean_90"] > 0.5).astype(float)
        df["enso_negative_regime"] = (df["enso_roll_mean_90"] < -0.5).astype(float)

    if "drought_index" in df.columns:
        drought = safe_numeric(df["drought_index"])
        df["drought_roll_mean_30"] = drought.rolling(30).mean()
        df["drought_roll_mean_90"] = drought.rolling(90).mean()
        df["drought_persistence_90d"] = (drought > 0).rolling(90).mean()

    if "frost_flag" in df.columns:
        frost_flag = safe_numeric(df["frost_flag"]).fillna(0.0)
        df["frost_rolling_7"] = frost_flag.rolling(7).sum()
        df["frost_rolling_30"] = frost_flag.rolling(30).sum()
        if "frost_severity_temp" in df.columns:
            df["frost_macro_30d"] = safe_numeric(df["frost_severity_temp"]).rolling(30).sum()
        else:
            df["frost_macro_30d"] = frost_flag.rolling(30).sum()

    if "heat_stress_90d" in df.columns and "brazil_flowering_flag" in df.columns:
        df["flowering_heat_risk"] = df["heat_stress_90d"] * df["brazil_flowering_flag"]
    if "rainfall_30_vs_90_regime" in df.columns and "brazil_flowering_flag" in df.columns:
        df["flowering_rain_regime"] = df["rainfall_30_vs_90_regime"] * df["brazil_flowering_flag"]
    if "rainfall_rolling_180" in df.columns and "brazil_harvest_flag" in df.columns:
        df["harvest_rain_load"] = df["rainfall_rolling_180"] * df["brazil_harvest_flag"]


def cumulative_target(series: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(0.0, index=series.index)
    numeric_series = safe_numeric(series)
    for i in range(1, horizon + 1):
        out = out + numeric_series.shift(-i)
    return out


# ============================================================
# MAIN DATASET BUILD
# ============================================================


def build_feature_dataset() -> tuple[pd.DataFrame, list[str]]:
    missing_optional_messages: list[str] = []

    coffee = load_market_log_file(COFFEE_FILE, "coffee_c")
    soy = load_market_log_file(SOY_FILE, "soybeans")
    sugar = load_market_log_file(SUGAR_FILE, "sugar")
    fx = load_market_log_file(FX_FILE, "usd_brl")
    climate = load_climate_file()

    df = coffee.merge(soy, on="Date", how="left")
    df = df.merge(sugar, on="Date", how="left")
    df = df.merge(fx, on="Date", how="left")
    df = df.merge(climate, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    market_cols = [
        c for c in df.columns
        if c not in ["Date", "tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]
    ]
    df[market_cols] = df[market_cols].ffill()

    climate_cols = [c for c in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"] if c in df.columns]
    if climate_cols:
        df[climate_cols] = df[climate_cols].ffill(limit=5)

    enso_df, enso_err = load_enso_file()
    if enso_err:
        missing_optional_messages.append(enso_err)
    else:
        df = df.merge(enso_df, on="Date", how="left")
        df["enso_index"] = safe_numeric(df["enso_index"]).ffill(limit=31)
        df["enso_missing_flag"] = df["enso_index"].isna().astype(float)

    drought_df, drought_err = load_optional_event_file(
        DROUGHT_CANDIDATES,
        "drought_index",
        ["drought_index", "drought_flag", "severity", "value"],
        binary=False,
    )
    if drought_err:
        missing_optional_messages.append(drought_err)
    else:
        df = df.merge(drought_df, on="Date", how="left")
        df["drought_index"] = safe_numeric(df["drought_index"]).ffill(limit=31)
        df["drought_flag"] = (safe_numeric(df["drought_index"]).fillna(0.0) > 0).astype(float)

    frost_df, frost_err = load_optional_event_file(
        FROST_CANDIDATES,
        "frost_flag",
        ["frost_flag", "frost", "freeze_flag", "event_flag", "value"],
        binary=True,
    )
    if frost_err:
        missing_optional_messages.append(frost_err)
    else:
        df = df.merge(frost_df, on="Date", how="left")
        df["frost_flag"] = canonicalize_binary(df["frost_flag"])

    # ── inventory (ICE certified stock) ───────────────────────────────────────
    inventory_df, inventory_err = load_inventory_file()
    if inventory_err:
        missing_optional_messages.append(inventory_err)
    else:
        df = df.merge(inventory_df, on="Date", how="left")
        # Inventory reports are weekly; forward-fill up to 7 business days
        inv_cols = [c for c in df.columns if c.startswith("inventory_")]
        if inv_cols:
            df[inv_cols] = df[inv_cols].ffill(limit=7)

    add_volatility_features(df)
    add_price_trend_features(df, "coffee_c", "coffee")
    add_price_trend_features(df, "sugar", "sugar")
    add_price_trend_features(df, "soybeans", "soybeans")
    add_price_trend_features(df, "usd_brl", "usd_brl")
    add_seasonality_features(df)
    add_relative_value_features(df)
    add_core_lags(df)
    add_climate_equation_features(df)
    add_macro_state_features(df)
    add_inventory_features(df)

    if "enso_index" in df.columns:
        df["enso_positive"] = (safe_numeric(df["enso_index"]) > 0).astype(float)
        add_lags(df, "enso_index", [1, 5, 20])

    if "drought_index" in df.columns:
        add_lags(df, "drought_index", [1, 5, 20])
        if "drought_flag" in df.columns:
            add_lags(df, "drought_flag", [1, 5, 20])

    if "frost_flag" in df.columns:
        add_lags(df, "frost_flag", [1, 5, 20])

    for horizon in [1, 5, 20, 60]:
        df[f"coffee_target_log_return_{horizon}d"] = cumulative_target(df["coffee_c_log_return"], horizon)

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df, missing_optional_messages


def main() -> None:
    print("Building enriched macro feature dataset...")
    print(f"Using market inputs from: {LOGDATA_DIR}")
    print(f"Using climate candidates: {[str(p) for p in CLIMATE_CANDIDATES]}")
    print(f"Using drought candidates: {[str(p) for p in DROUGHT_CANDIDATES]}")
    print(f"Using frost candidates: {[str(p) for p in FROST_CANDIDATES]}")
    print(f"Using inventory candidates: {[str(p) for p in INVENTORY_CANDIDATES]}")

    df, missing_messages = build_feature_dataset()

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nAdded macro-state features:")
    print("- coffee_trend_20d / 60d / 120d")
    print("- usd_brl_trend_20d / 60d / 120d")
    print("- rainfall_rolling_90 / 180 and rainfall regime ratios")
    print("- heat stress / frost severity / drought persistence features")
    print("- coffee volatility regime and drawdown features")
    print("- 60-day target for slower macro moves")
    print("- ICE certified stock level, momentum, regime, and Brazil-share features")

    if missing_messages:
        print("\nOptional files not used:")
        for msg in missing_messages:
            print(f"- {msg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
