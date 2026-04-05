from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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
ENSO_CANDIDATES = [
    LOGDATA_DIR / "enso.csv",
    DATA_DIR / "enso.csv",
    DATA_DIR / "nino34.long.anom.csv",
    LOGDATA_DIR / "nino34.long.anom.csv",
]
DROUGHT_CANDIDATES = [
    LOGDATA_DIR / "brazil_drought.csv",
    DATA_DIR / "brazil_drought.csv",
]
FROST_CANDIDATES = [
    LOGDATA_DIR / "brazil_frost.csv",
    DATA_DIR / "brazil_frost.csv",
]

# ICE certified stock / inventory data
INVENTORY_CANDIDATES = [
    DATA_DIR / "standardized_inventory.csv",
    LOGDATA_DIR / "standardized_inventory.csv",
    DATA_DIR / "inventory.csv",
    LOGDATA_DIR / "inventory.csv",
]

HISTORY_YEARS = 2
PRESENT_DATE = pd.Timestamp("2026-04-03")
FORECAST_WEEKS = 52
BUSINESS_DAYS_PER_WEEK = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 200
PREDICTION_SHRINK = 0.50

COFFEE_LAGS = [1, 2, 3, 5, 10, 20]
EXOG_LAGS = [1, 2, 3, 5, 10]
CLIMATE_LAGS = [1, 3, 5, 10]
EVENT_LAGS = [1, 5, 10, 20]
ROLL_WINDOWS = [5, 10, 20]
MACRO_TREND_WINDOWS = [20, 60, 120]
MACRO_ACCUM_WINDOWS = [30, 90, 180]
VOL_WINDOWS = [20, 60]

MERGED_FILE = OUTPUT_DIR / "model_merged_data_log_returns.csv"
METRICS_FILE = OUTPUT_DIR / "model_metrics_by_horizon_log_returns.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance_h1_log_returns.csv"
FORECAST_FILE = OUTPUT_DIR / "coffee_forecast_output_log_returns.csv"
PLOT_FILE = OUTPUT_DIR / "coffee_projection_plot_log_returns.png"


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


def next_business_days(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last_date + pd.offsets.BDay(1), periods=periods)


def train_test_split_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, label: str, horizon_weeks: int) -> dict:
    return {
        "model": label,
        "horizon_weeks": horizon_weeks,
        "n_obs": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def standardize_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return df.reset_index(drop=True)


def load_log_return_market_file(file_path: Path, value_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")

    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        raise ValueError(f"{file_path} is missing a Date column")
    if "Price" not in df.columns:
        raise ValueError(f"{file_path} is missing a Price column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = safe_numeric(df["Price"])
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)

    log_return_col = None
    for candidate in ["log_return", f"{value_name}_log_return", f"{value_name}_return_log"]:
        if candidate in df.columns:
            log_return_col = candidate
            break

    if log_return_col is None:
        valid_price = df["Price"].where(df["Price"] > 0)
        df["log_return"] = np.log(valid_price).diff()
        log_return_col = "log_return"

    df[log_return_col] = safe_numeric(df[log_return_col])

    out = df[["Date", "Price", log_return_col]].copy()
    out = out.rename(columns={"Price": value_name, log_return_col: f"{value_name}_log_return"})
    return out.reset_index(drop=True)


def load_climate_file() -> pd.DataFrame:
    climate_path = first_existing(CLIMATE_CANDIDATES)
    if climate_path is None:
        raise FileNotFoundError(
            "Missing climate file. Expected one of: " + ", ".join(str(p) for p in CLIMATE_CANDIDATES)
        )

    df = pd.read_csv(climate_path)
    if "Date" not in df.columns:
        raise ValueError("Climate file must contain a Date column")

    df = standardize_date(df)
    for col in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df


def load_enso_file() -> tuple[pd.DataFrame | None, str | None]:
    for path in ENSO_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "Date" not in df.columns:
            continue

        if "enso_index" in df.columns:
            value_col = "enso_index"
        else:
            non_date_cols = [c for c in df.columns if c != "Date"]
            if not non_date_cols:
                continue
            value_col = non_date_cols[0]

        out = df[["Date", value_col]].copy()
        out = standardize_date(out)
        out[value_col] = safe_numeric(out[value_col]).replace(-99.99, np.nan)
        out = out.rename(columns={value_col: "enso_index"})
        out["enso_missing_flag"] = out["enso_index"].isna().astype(float)
        return out[["Date", "enso_index", "enso_missing_flag"]], None

    return None, "Missing optional ENSO file in data/ or logdata/"


def load_optional_file(candidates: list[Path], expected_cols: list[str], aliases: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    path = first_existing(candidates)
    if path is None:
        return None, f"Missing optional file: one of {[p.name for p in candidates]}"

    aliases = aliases or {}
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        return None, f"{path.name} is missing a Date column"

    rename_map: dict[str, str] = {}
    current_cols = set(df.columns)
    for expected in expected_cols:
        if expected in current_cols:
            continue
        found = None
        for alt in aliases.get(expected, []):
            if alt in current_cols:
                found = alt
                break
        if found is not None:
            rename_map[found] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return None, f"{path.name} is missing columns: {missing}"

    out = df[["Date"] + expected_cols].copy()
    out = standardize_date(out)
    for col in expected_cols:
        out[col] = safe_numeric(out[col])

    return out, None


def load_inventory_file() -> tuple[pd.DataFrame | None, str | None]:
    """
    Load standardized_inventory.csv and pivot it into a daily time-series.

    Columns expected: report_date, section, country, warehouse, bags
    Produces:
        inventory_certified_bags      – 'TOTAL BAGS CERTIFIED', warehouse='Total', all countries
        inventory_transition_bags     – 'TRANSITION BAGS CERTIFIED', warehouse='Total'
        inventory_total_bags          – certified + transition
        inventory_brazil_bags         – certified bags for Brazil
        inventory_brazil_share        – Brazil bags / total certified
        inventory_certified_log_chg   – log change vs prior report
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

    out = certified.merge(transition, on="Date", how="outer")
    out = out.merge(brazil, on="Date", how="outer")
    out = standardize_date(out)

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
    Build rolling, regime, and lag features from the inventory columns
    already merged into df.  Skips gracefully if columns are absent.

    Rolling windows and ratio features are computed only on rows with real
    inventory data (inventory_available_flag == 1).  Historical rows where
    inventory is imputed as 0 are treated as NaN during rolling computation
    so they do not distort the baseline means.  All resulting NaNs are filled
    with 0 so training rows are preserved; the model uses inventory_available_flag
    to learn when these features are meaningful.
    """
    if "inventory_certified_bags" not in df.columns:
        return

    available = df.get("inventory_available_flag", pd.Series(1.0, index=df.index))

    # Mask imputed-zero rows so rolling windows use only real data
    cert = safe_numeric(df["inventory_certified_bags"]).where(available == 1)
    total = safe_numeric(df.get("inventory_total_bags", cert)).where(available == 1)
    brazil_share = safe_numeric(df.get("inventory_brazil_share")).where(available == 1) if "inventory_brazil_share" in df.columns else None
    log_chg = safe_numeric(df.get("inventory_certified_log_chg")).where(available == 1) if "inventory_certified_log_chg" in df.columns else None

    # rolling level stats (windows in approximate business days)
    for window, days in [(4, 20), (13, 65), (26, 130)]:
        df[f"inventory_certified_roll_mean_{window}w"] = cert.rolling(days, min_periods=2).mean().fillna(0.0)
        df[f"inventory_certified_roll_std_{window}w"] = cert.rolling(days, min_periods=2).std().fillna(0.0)

    # ratio of current level vs rolling baseline (surplus / deficit signal)
    roll_13 = df["inventory_certified_roll_mean_13w"].replace(0, np.nan)
    roll_26 = df["inventory_certified_roll_mean_26w"].replace(0, np.nan)
    df["inventory_cert_vs_13w_mean"] = (cert / roll_13 - 1.0).fillna(0.0)
    df["inventory_cert_vs_26w_mean"] = (cert / roll_26 - 1.0).fillna(0.0)

    # binary supply regime flags (0 where no inventory)
    cert_vs_26 = df["inventory_cert_vs_26w_mean"]
    df["inventory_low_supply_flag"] = ((cert_vs_26 < -0.05) & (available == 1)).astype(float)
    df["inventory_high_supply_flag"] = ((cert_vs_26 > 0.05) & (available == 1)).astype(float)

    # momentum: rolling accumulation of weekly log-change
    if log_chg is not None:
        df["inventory_log_chg_roll_4w"] = log_chg.rolling(20, min_periods=2).sum().fillna(0.0)
        df["inventory_log_chg_roll_13w"] = log_chg.rolling(65, min_periods=2).sum().fillna(0.0)
        df["inventory_drawdown_flag"] = ((df["inventory_log_chg_roll_4w"] < 0) & (available == 1)).astype(float)

    # Brazil-concentration features
    if brazil_share is not None:
        df["inventory_brazil_share_roll_13w"] = brazil_share.rolling(65, min_periods=2).mean().fillna(0.0)
        df["inventory_brazil_dominant_flag"] = ((brazil_share > 0.50) & (available == 1)).astype(float)

    # total supply relative to 26-week baseline
    total_base_26 = total.rolling(130, min_periods=2).mean()
    df["inventory_total_roll_mean_26w"] = total_base_26.fillna(0.0)
    df["inventory_total_vs_26w_mean"] = (total / total_base_26.replace(0, np.nan) - 1.0).fillna(0.0)

    # lags on key supply signals (shift the masked series; fill NaN with 0)
    for lag in [5, 10, 20, 60]:
        df[f"inventory_certified_bags_lag_{lag}"] = cert.shift(lag).fillna(0.0)
        if log_chg is not None:
            df[f"inventory_certified_log_chg_lag_{lag}"] = log_chg.shift(lag).fillna(0.0)
        df[f"inventory_cert_vs_26w_mean_lag_{lag}"] = cert_vs_26.shift(lag).fillna(0.0)


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    series = safe_numeric(df[col])
    for lag in lags:
        if lag > 0:
            df[f"{col}_lag_{lag}"] = series.shift(lag)


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    series = safe_numeric(df[col])
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = series.rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = series.rolling(w).std()


def add_calendar_features(df: pd.DataFrame) -> None:
    date = pd.to_datetime(df["Date"])
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


def cumulative_forward_log_return(series: pd.Series, horizon_days: int) -> pd.Series:
    acc = pd.Series(0.0, index=series.index)
    for i in range(1, horizon_days + 1):
        acc = acc + series.shift(-i)
    return acc


def add_macro_state_features(df: pd.DataFrame) -> None:
    coffee_price = safe_numeric(df.get("coffee_c"))
    coffee_ret = safe_numeric(df.get("coffee_c_log_return"))
    fx_price = safe_numeric(df.get("usd_brl"))
    rainfall = safe_numeric(df.get("rainfall"))
    tmax = safe_numeric(df.get("tmax"))
    tmin = safe_numeric(df.get("tmin"))
    enso = safe_numeric(df.get("enso_index"))
    drought = safe_numeric(df.get("drought_index"))
    frost = safe_numeric(df.get("frost_severity"))

    for window in MACRO_TREND_WINDOWS:
        if coffee_price is not None:
            df[f"coffee_trend_{window}d"] = coffee_price.pct_change(window)
            rolling_peak = coffee_price.rolling(window, min_periods=max(5, window // 4)).max()
            df[f"coffee_drawdown_{window}d"] = coffee_price / rolling_peak - 1.0
        if fx_price is not None:
            df[f"usd_brl_trend_{window}d"] = fx_price.pct_change(window)

    if "coffee_trend_120d" in df.columns:
        df["coffee_macro_uptrend_flag"] = (safe_numeric(df["coffee_trend_120d"]) > 0).astype(float)
        df["coffee_macro_downtrend_flag"] = (safe_numeric(df["coffee_trend_120d"]) < 0).astype(float)
    if "usd_brl_trend_120d" in df.columns:
        df["fx_brl_strengthening_flag"] = (safe_numeric(df["usd_brl_trend_120d"]) < 0).astype(float)
        df["fx_brl_weakening_flag"] = (safe_numeric(df["usd_brl_trend_120d"]) > 0).astype(float)

    if rainfall is not None:
        for window in MACRO_ACCUM_WINDOWS:
            df[f"rainfall_rolling_{window}"] = rainfall.rolling(window, min_periods=max(5, window // 3)).sum()
        if "rainfall_rolling_30" in df.columns and "rainfall_rolling_90" in df.columns:
            base = safe_numeric(df["rainfall_rolling_90"]).replace(0, np.nan)
            df["rainfall_regime_ratio_30_90"] = safe_numeric(df["rainfall_rolling_30"]) / base
        if "rainfall_rolling_90" in df.columns and "rainfall_rolling_180" in df.columns:
            base = safe_numeric(df["rainfall_rolling_180"]).replace(0, np.nan)
            df["rainfall_regime_ratio_90_180"] = safe_numeric(df["rainfall_rolling_90"]) / base

    if tmax is not None:
        heat_flag = (tmax > 30.0).astype(float)
        for window in [30, 90]:
            df[f"heat_stress_{window}d"] = heat_flag.rolling(window, min_periods=max(5, window // 3)).mean()

    if tmin is not None:
        frost_temp_severity = np.maximum(0.0, 2.0 - tmin)
        for window in [30, 90]:
            df[f"frost_severity_temp_{window}d"] = frost_temp_severity.rolling(window, min_periods=max(5, window // 3)).sum()

    if drought is not None:
        for window in [30, 90]:
            df[f"drought_roll_mean_{window}"] = drought.rolling(window, min_periods=max(5, window // 3)).mean()
        if "drought_roll_mean_90" in df.columns:
            df["drought_persistence_90d"] = (safe_numeric(df["drought_roll_mean_90"]) > 0).astype(float)

    if enso is not None:
        for window in [30, 90, 180]:
            df[f"enso_roll_mean_{window}"] = enso.rolling(window, min_periods=max(5, window // 3)).mean()
        if "enso_roll_mean_90" in df.columns:
            df["enso_el_nino_flag"] = (safe_numeric(df["enso_roll_mean_90"]) > 0.5).astype(float)
            df["enso_la_nina_flag"] = (safe_numeric(df["enso_roll_mean_90"]) < -0.5).astype(float)

    if coffee_ret is not None:
        abs_ret = coffee_ret.abs()
        for window in [5, 20, 60]:
            df[f"coffee_abs_return_{window}d"] = abs_ret.rolling(window, min_periods=max(3, window // 3)).mean()
        for window in VOL_WINDOWS:
            df[f"coffee_vol_{window}d"] = coffee_ret.rolling(window, min_periods=max(5, window // 3)).std()
        if "coffee_vol_20d" in df.columns and "coffee_vol_60d" in df.columns:
            base = safe_numeric(df["coffee_vol_60d"]).replace(0, np.nan)
            df["coffee_vol_regime_ratio_20_60"] = safe_numeric(df["coffee_vol_20d"]) / base
            df["coffee_high_vol_regime"] = (safe_numeric(df["coffee_vol_regime_ratio_20_60"]) > 1.2).astype(float)

    if frost is not None:
        for window in [30, 90]:
            df[f"frost_roll_sum_{window}"] = frost.rolling(window, min_periods=max(5, window // 3)).sum()

    if all(col in df.columns for col in ["brazil_flowering_flag", "rainfall_rolling_90"]):
        df["flowering_rainfall_90_interaction"] = safe_numeric(df["brazil_flowering_flag"]) * safe_numeric(df["rainfall_rolling_90"])
    if all(col in df.columns for col in ["brazil_harvest_flag", "coffee_trend_60d"]):
        df["harvest_trend_60_interaction"] = safe_numeric(df["brazil_harvest_flag"]) * safe_numeric(df["coffee_trend_60d"])


# ============================================================
# DATASET
# ============================================================

def build_merged_dataset() -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []

    coffee = load_log_return_market_file(COFFEE_FILE, "coffee_c")
    soy = load_log_return_market_file(SOY_FILE, "soybeans")
    sugar = load_log_return_market_file(SUGAR_FILE, "sugar")
    fx = load_log_return_market_file(FX_FILE, "usd_brl")

    df = coffee.merge(soy, on="Date", how="left")
    df = df.merge(sugar, on="Date", how="left")
    df = df.merge(fx, on="Date", how="left")
    df = standardize_date(df)

    market_cols = [c for c in df.columns if c != "Date"]
    df[market_cols] = df[market_cols].ffill()

    climate = load_climate_file()
    df = df.merge(climate, on="Date", how="left")
    climate_cols = [c for c in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"] if c in df.columns]
    if climate_cols:
        df[climate_cols] = df[climate_cols].ffill(limit=7)
        if {"tmax", "tmin"}.issubset(df.columns):
            df["tavg"] = (safe_numeric(df["tmax"]) + safe_numeric(df["tmin"])) / 2.0
            df["trange"] = safe_numeric(df["tmax"]) - safe_numeric(df["tmin"])
        if "rainfall" in df.columns:
            df["rainfall_roll_mean_7"] = safe_numeric(df["rainfall"]).rolling(7).mean()
            df["rainfall_roll_mean_30"] = safe_numeric(df["rainfall"]).rolling(30).mean()

    enso_df, enso_err = load_enso_file()
    if enso_df is not None:
        df = df.merge(enso_df, on="Date", how="left")
        df["enso_index"] = safe_numeric(df["enso_index"]).ffill(limit=31)
        df["enso_missing_flag"] = safe_numeric(df["enso_missing_flag"]).fillna(1.0)
        df["enso_positive"] = (safe_numeric(df["enso_index"]) > 0).astype(float)
    else:
        notes.append(enso_err or "ENSO unavailable")

    drought_df, drought_err = load_optional_file(
        DROUGHT_CANDIDATES,
        ["drought_index"],
        aliases={"drought_index": ["drought_severity", "dryness_index", "drought_score"]},
    )
    if drought_df is not None:
        df = df.merge(drought_df, on="Date", how="left")
        df["drought_index"] = safe_numeric(df["drought_index"]).ffill(limit=14)
        df["drought_flag"] = (safe_numeric(df["drought_index"]) > 0).astype(float)
    else:
        notes.append(drought_err or "Drought unavailable")

    frost_df, frost_err = load_optional_file(
        FROST_CANDIDATES,
        ["frost_flag"],
        aliases={"frost_flag": ["frost", "freeze_flag", "cold_event_flag"]},
    )
    if frost_df is not None:
        df = df.merge(frost_df, on="Date", how="left")
        df["frost_flag"] = safe_numeric(df["frost_flag"]).fillna(0.0)
        if "frost_severity" not in df.columns and "tmin" in df.columns:
            df["frost_severity"] = np.maximum(0.0, 2.0 - safe_numeric(df["tmin"])) * df["frost_flag"]
    else:
        notes.append(frost_err or "Frost unavailable")

    if "frost_severity" not in df.columns and "frost_flag" in df.columns:
        df["frost_severity"] = safe_numeric(df["frost_flag"])

    # ── ICE certified inventory stock ──────────────────────────────────────────
    inventory_df, inventory_err = load_inventory_file()
    if inventory_df is not None:
        df = df.merge(inventory_df, on="Date", how="left")
        # inventory reports are weekly; forward-fill up to 7 business days
        inv_cols = [c for c in df.columns if c.startswith("inventory_")]
        if inv_cols:
            df[inv_cols] = df[inv_cols].ffill(limit=7)
        # Mark rows with real inventory data vs. imputed (historical rows before coverage)
        df["inventory_available_flag"] = (~df["inventory_certified_bags"].isna()).astype(float)
        # Impute missing inventory with 0 so training rows are not dropped
        raw_inv_cols = [c for c in inv_cols if c in df.columns]
        df[raw_inv_cols] = df[raw_inv_cols].fillna(0.0)
    else:
        notes.append(inventory_err or "Inventory unavailable")

    add_lag_features(df, "coffee_c_log_return", COFFEE_LAGS)
    add_rolling_features(df, "coffee_c_log_return", ROLL_WINDOWS)

    for col in ["soybeans_log_return", "sugar_log_return", "usd_brl_log_return"]:
        if col in df.columns:
            add_lag_features(df, col, EXOG_LAGS)

    for col in [
        "tmax", "tmin", "tavg", "trange", "rainfall", "tmax_change_pct", "tmin_change_pct",
        "rainfall_roll_mean_7", "rainfall_roll_mean_30",
    ]:
        if col in df.columns:
            add_lag_features(df, col, CLIMATE_LAGS)

    for col in ["enso_index", "enso_positive", "enso_missing_flag", "drought_index", "drought_flag", "frost_flag", "frost_severity"]:
        if col in df.columns:
            add_lag_features(df, col, EVENT_LAGS)

    df["coffee_minus_sugar_log_return"] = safe_numeric(df["coffee_c_log_return"]) - safe_numeric(df.get("sugar_log_return"))
    df["coffee_minus_soybeans_log_return"] = safe_numeric(df["coffee_c_log_return"]) - safe_numeric(df.get("soybeans_log_return"))
    df["coffee_minus_fx_log_return"] = safe_numeric(df["coffee_c_log_return"]) - safe_numeric(df.get("usd_brl_log_return"))

    add_calendar_features(df)
    add_macro_state_features(df)
    add_inventory_features(df)
    df = standardize_date(df)
    df.to_csv(MERGED_FILE, index=False)
    return df, notes


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []

    cols += [f"coffee_c_log_return_lag_{lag}" for lag in COFFEE_LAGS if f"coffee_c_log_return_lag_{lag}" in df.columns]
    cols += [f"coffee_c_log_return_roll_mean_{w}" for w in ROLL_WINDOWS if f"coffee_c_log_return_roll_mean_{w}" in df.columns]
    cols += [f"coffee_c_log_return_roll_std_{w}" for w in ROLL_WINDOWS if f"coffee_c_log_return_roll_std_{w}" in df.columns]

    for base in ["soybeans_log_return", "sugar_log_return", "usd_brl_log_return"]:
        cols += [f"{base}_lag_{lag}" for lag in EXOG_LAGS if f"{base}_lag_{lag}" in df.columns]

    for base in [
        "tmax", "tmin", "tavg", "trange", "rainfall", "tmax_change_pct", "tmin_change_pct",
        "rainfall_roll_mean_7", "rainfall_roll_mean_30",
    ]:
        cols += [f"{base}_lag_{lag}" for lag in CLIMATE_LAGS if f"{base}_lag_{lag}" in df.columns]

    for base in ["enso_index", "enso_positive", "enso_missing_flag", "drought_index", "drought_flag", "frost_flag", "frost_severity"]:
        cols += [f"{base}_lag_{lag}" for lag in EVENT_LAGS if f"{base}_lag_{lag}" in df.columns]

    macro_cols = [
        "coffee_minus_sugar_log_return",
        "coffee_minus_soybeans_log_return",
        "coffee_minus_fx_log_return",
        "month_sin", "month_cos", "week_sin", "week_cos",
        "brazil_harvest_flag", "brazil_flowering_flag",
        "quarter", "day_of_week",
        "coffee_trend_20d", "coffee_trend_60d", "coffee_trend_120d",
        "coffee_drawdown_60d", "coffee_drawdown_120d",
        "coffee_macro_uptrend_flag", "coffee_macro_downtrend_flag",
        "usd_brl_trend_20d", "usd_brl_trend_60d", "usd_brl_trend_120d",
        "fx_brl_strengthening_flag", "fx_brl_weakening_flag",
        "rainfall_rolling_30", "rainfall_rolling_90", "rainfall_rolling_180",
        "rainfall_regime_ratio_30_90", "rainfall_regime_ratio_90_180",
        "heat_stress_30d", "heat_stress_90d",
        "frost_severity_temp_30d", "frost_severity_temp_90d",
        "frost_roll_sum_30", "frost_roll_sum_90",
        "drought_roll_mean_30", "drought_roll_mean_90", "drought_persistence_90d",
        "enso_roll_mean_30", "enso_roll_mean_90", "enso_roll_mean_180",
        "enso_el_nino_flag", "enso_la_nina_flag",
        "coffee_abs_return_5d", "coffee_abs_return_20d", "coffee_abs_return_60d",
        "coffee_vol_20d", "coffee_vol_60d", "coffee_vol_regime_ratio_20_60", "coffee_high_vol_regime",
        "flowering_rainfall_90_interaction", "harvest_trend_60_interaction",
    ]
    cols += [c for c in macro_cols if c in df.columns]

    # ── inventory / ICE certified stock features ──────────────────────────────
    inventory_lag_cols = [
        f"inventory_certified_bags_lag_{lag}" for lag in [5, 10, 20, 60]
    ] + [
        f"inventory_certified_log_chg_lag_{lag}" for lag in [5, 10, 20, 60]
    ] + [
        f"inventory_cert_vs_26w_mean_lag_{lag}" for lag in [5, 10, 20, 60]
    ]
    inventory_derived_cols = [
        "inventory_available_flag",
        "inventory_certified_roll_mean_4w",
        "inventory_certified_roll_mean_13w",
        "inventory_certified_roll_mean_26w",
        "inventory_certified_roll_std_13w",
        "inventory_cert_vs_13w_mean",
        "inventory_cert_vs_26w_mean",
        "inventory_low_supply_flag",
        "inventory_high_supply_flag",
        "inventory_log_chg_roll_4w",
        "inventory_log_chg_roll_13w",
        "inventory_drawdown_flag",
        "inventory_brazil_share",
        "inventory_brazil_share_roll_13w",
        "inventory_brazil_dominant_flag",
        "inventory_total_vs_26w_mean",
    ]
    cols += [c for c in inventory_lag_cols + inventory_derived_cols if c in df.columns]

    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# ============================================================
# MODELING
# ============================================================

def fit_direct_models(df: pd.DataFrame) -> tuple[dict[int, XGBRegressor], dict[int, tuple[float, float]], pd.DataFrame]:
    features = get_feature_columns(df)
    if not features:
        raise RuntimeError("No usable feature columns were built.")

    models: dict[int, XGBRegressor] = {}
    clips: dict[int, tuple[float, float]] = {}
    metrics_rows: list[dict] = []
    h1_importance = None

    for horizon_weeks in range(1, FORECAST_WEEKS + 1):
        horizon_days = horizon_weeks * BUSINESS_DAYS_PER_WEEK
        target_col = f"target_log_return_{horizon_weeks}w"

        work = df.copy()
        work[target_col] = cumulative_forward_log_return(work["coffee_c_log_return"], horizon_days)

        model_df = work.dropna(subset=features + [target_col, "coffee_c"]).copy()
        if len(model_df) < MIN_TRAIN_ROWS:
            continue

        train_df, test_df = train_test_split_time(model_df, TEST_SIZE)
        if len(train_df) < MIN_TRAIN_ROWS or len(test_df) < 20:
            continue

        X_train = train_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = train_df[target_col].astype(float)
        X_test = test_df[features].apply(pd.to_numeric, errors="coerce")
        y_test = test_df[target_col].astype(float)

        sample_weight = np.ones(len(train_df), dtype=float)
        if "coffee_high_vol_regime" in train_df.columns:
            sample_weight *= 1.0 + 0.5 * train_df["coffee_high_vol_regime"].fillna(0.0).to_numpy(dtype=float)
        if "coffee_macro_uptrend_flag" in train_df.columns and "coffee_macro_downtrend_flag" in train_df.columns:
            macro_regime = (train_df["coffee_macro_uptrend_flag"].fillna(0.0) + train_df["coffee_macro_downtrend_flag"].fillna(0.0)).to_numpy(dtype=float)
            sample_weight *= 1.0 + 0.25 * macro_regime
        sample_weight *= 1.0 + 0.01 * horizon_weeks

        model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            min_child_weight=3,
            reg_alpha=0.0,
            reg_lambda=1.0,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        pred_test = model.predict(X_test)
        zero_pred = np.zeros(len(y_test))

        metrics_rows.append(evaluate_predictions(y_test, pred_test, "xgboost", horizon_weeks))
        metrics_rows.append(evaluate_predictions(y_test, zero_pred, "zero_return_baseline", horizon_weeks))

        lo = float(np.nanpercentile(y_train, 5))
        hi = float(np.nanpercentile(y_train, 95))
        clips[horizon_weeks] = (lo, hi)
        models[horizon_weeks] = model

        if horizon_weeks == 1:
            h1_importance = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_FILE, index=False)

    if h1_importance is not None:
        h1_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    return models, clips, metrics_df


def forecast_from_latest(
    df: pd.DataFrame,
    models: dict[int, XGBRegressor],
    clips: dict[int, tuple[float, float]],
    anchor_date: pd.Timestamp = PRESENT_DATE,
) -> pd.DataFrame:
    features = get_feature_columns(df)

    all_history = df[["Date", "coffee_c"]].dropna().copy()
    all_history["Date"] = pd.to_datetime(all_history["Date"])
    all_history = all_history.sort_values("Date").reset_index(drop=True)

    usable = df.dropna(subset=features + ["coffee_c"]).copy()
    usable["Date"] = pd.to_datetime(usable["Date"])
    usable = usable.sort_values("Date").reset_index(drop=True)

    anchor_date = pd.to_datetime(anchor_date).normalize()

    history_on_or_before_anchor = all_history.loc[all_history["Date"] <= anchor_date].copy()
    if history_on_or_before_anchor.empty:
        min_date = all_history["Date"].min()
        max_date = all_history["Date"].max()
        raise ValueError(
            f"No observed historical coffee price exists on or before anchor date {anchor_date.date()}. "
            f"Available historical dates span {min_date.date()} to {max_date.date()}."
        )

    anchor_row = history_on_or_before_anchor.iloc[-1]
    anchor_price_date = pd.to_datetime(anchor_row["Date"])
    anchor_price = float(anchor_row["coffee_c"])

    usable_on_or_before_anchor = usable.loc[usable["Date"] <= anchor_date].copy()
    if usable_on_or_before_anchor.empty:
        min_date = usable["Date"].min()
        max_date = usable["Date"].max()
        raise ValueError(
            f"No model-ready historical row exists on or before anchor date {anchor_date.date()}. "
            f"Available model-ready dates span {min_date.date()} to {max_date.date()}."
        )

    latest = usable_on_or_before_anchor.iloc[-1]
    model_asof_date = pd.to_datetime(latest["Date"])

    if anchor_price_date != anchor_date:
        print(
            f"Warning: no observed coffee price exists exactly on {anchor_date.date()}. "
            f"Using latest available observed price from {anchor_price_date.date()} as the anchor price."
        )

    if model_asof_date != anchor_date:
        print(
            f"Warning: no model-ready row exists exactly on {anchor_date.date()}. "
            f"Using latest available model-ready row from {model_asof_date.date()} for features."
        )

    raw_predictions: dict[int, float] = {}
    for horizon_weeks, model in models.items():
        X_latest = latest[features].to_frame().T.apply(pd.to_numeric, errors="coerce")
        pred_log_return = float(model.predict(X_latest)[0])
        lo, hi = clips[horizon_weeks]
        pred_log_return = float(np.clip(pred_log_return, lo, hi))
        pred_log_return *= PREDICTION_SHRINK
        raw_predictions[horizon_weeks] = pred_log_return

    macro_anchor = {}
    for horizon in [12, 26, 52]:
        if horizon in raw_predictions:
            macro_anchor[horizon] = raw_predictions[horizon]

    smoothed_predictions: dict[int, float] = {}
    for horizon_weeks, pred_log_return in sorted(raw_predictions.items()):
        long_candidates = [v for k, v in macro_anchor.items() if k >= horizon_weeks]
        if long_candidates:
            macro_blend = float(np.mean(long_candidates))
            blend_weight = min(0.60, horizon_weeks / 52.0)
            pred_log_return = (1.0 - blend_weight) * pred_log_return + blend_weight * macro_blend
        smoothed_predictions[horizon_weeks] = pred_log_return

    rows = []
    for horizon_weeks, pred_log_return in smoothed_predictions.items():
        projected_price = anchor_price * np.exp(pred_log_return)
        forecast_date = next_business_days(anchor_date, horizon_weeks * BUSINESS_DAYS_PER_WEEK)[-1]

        rows.append({
            "anchor_date": anchor_date,
            "anchor_price_date": anchor_price_date,
            "model_asof_date": model_asof_date,
            "forecast_date": forecast_date,
            "horizon_weeks": horizon_weeks,
            "anchor_price": anchor_price,
            "predicted_cumulative_log_return": pred_log_return,
            "projected_price": projected_price,
            "projected_pct_change": projected_price / anchor_price - 1.0,
        })

    out = pd.DataFrame(rows).sort_values("horizon_weeks").reset_index(drop=True)
    out.to_csv(FORECAST_FILE, index=False)
    return out


def make_projection_plot(df: pd.DataFrame, forecast_df: pd.DataFrame, anchor_date: pd.Timestamp = PRESENT_DATE) -> None:
    hist = df[["Date", "coffee_c"]].dropna().copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)

    fc = forecast_df.copy()
    fc["anchor_date"] = pd.to_datetime(fc["anchor_date"])
    fc["anchor_price_date"] = pd.to_datetime(fc["anchor_price_date"])
    fc["model_asof_date"] = pd.to_datetime(fc["model_asof_date"])
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    fc = fc.sort_values("forecast_date").reset_index(drop=True)

    anchor_date = pd.to_datetime(anchor_date).normalize()
    history_start = anchor_date - pd.DateOffset(years=HISTORY_YEARS)
    one_year_end = anchor_date + pd.DateOffset(years=1)

    anchor_price = float(fc["anchor_price"].iloc[0])

    hist = hist.loc[(hist["Date"] >= history_start) & (hist["Date"] <= anchor_date)].copy()

    anchor_hist = hist.loc[hist["Date"] == anchor_date].copy()
    if anchor_hist.empty:
        hist = pd.concat(
            [
                hist,
                pd.DataFrame({"Date": [anchor_date], "coffee_c": [anchor_price]}),
            ],
            ignore_index=True,
        ).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    fc_plot = pd.concat(
        [
            pd.DataFrame({"plot_date": [anchor_date], "plot_price": [anchor_price]}),
            fc[["forecast_date", "projected_price"]].rename(
                columns={"forecast_date": "plot_date", "projected_price": "plot_price"}
            ),
        ],
        ignore_index=True,
    )
    fc_plot = fc_plot.loc[(fc_plot["plot_date"] >= anchor_date) & (fc_plot["plot_date"] <= one_year_end)].copy()

    plt.figure(figsize=(14, 7))
    plt.plot(hist["Date"], hist["coffee_c"], label="Historical coffee price")
    plt.plot(fc_plot["plot_date"], fc_plot["plot_price"], marker="o", label="Projected coffee price")
    plt.axvline(anchor_date, linestyle="--", linewidth=1, label=f"Anchor date ({anchor_date.date()})")
    plt.xlabel("Date")
    plt.ylabel("Coffee price")
    plt.title("Coffee price: 2 years historical + 1 year projected from Apr 3, 2026")
    plt.xlim(history_start, one_year_end)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Building modeling dataset...")
    df, notes = build_merged_dataset()

    print("Training direct multi-horizon XGBoost models...")
    models, clips, metrics_df = fit_direct_models(df)
    if not models:
        raise RuntimeError("No models were trained. Check data availability and feature coverage.")

    print(f"Generating forecast from anchor date {PRESENT_DATE.date()}...")
    forecast_df = forecast_from_latest(df, models, clips, anchor_date=PRESENT_DATE)

    anchor_price_date = pd.to_datetime(forecast_df["anchor_price_date"].iloc[0]).date()
    model_asof_date = pd.to_datetime(forecast_df["model_asof_date"].iloc[0]).date()
    if anchor_price_date != PRESENT_DATE.date():
        print(f"Projection used the latest observed coffee price from {anchor_price_date} as the anchor price for {PRESENT_DATE.date()}.")
    if model_asof_date != PRESENT_DATE.date():
        print(f"Projection used the latest model-ready row from {model_asof_date} for features and anchored the forecast at {PRESENT_DATE.date()}.")

    print("Saving plot...")
    make_projection_plot(df, forecast_df, anchor_date=PRESENT_DATE)

    print(f"\nSaved merged dataset: {MERGED_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    if FEATURE_IMPORTANCE_FILE.exists():
        print(f"Saved H=1 feature importance: {FEATURE_IMPORTANCE_FILE}")
    print(f"Saved forecast: {FORECAST_FILE}")
    print(f"Saved plot: {PLOT_FILE}")

    if notes:
        print("\nOptional inputs not used:")
        for note in notes:
            print(f"- {note}")

    best_rows = metrics_df[metrics_df["model"] == "xgboost"].sort_values("horizon_weeks").head(5)
    if not best_rows.empty:
        print("\nFirst few model metrics:")
        print(best_rows.to_string(index=False))


if __name__ == "__main__":
    main()
