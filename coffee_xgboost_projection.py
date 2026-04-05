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

    cols += [
        c for c in [
            "coffee_minus_sugar_log_return",
            "coffee_minus_soybeans_log_return",
            "coffee_minus_fx_log_return",
            "month_sin", "month_cos", "week_sin", "week_cos",
            "brazil_harvest_flag", "brazil_flowering_flag",
            "quarter", "day_of_week",
        ] if c in df.columns
    ]

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
        model.fit(X_train, y_train)

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

    rows = []
    for horizon_weeks, model in models.items():
        X_latest = latest[features].to_frame().T.apply(pd.to_numeric, errors="coerce")
        pred_log_return = float(model.predict(X_latest)[0])

        lo, hi = clips[horizon_weeks]
        pred_log_return = float(np.clip(pred_log_return, lo, hi))
        pred_log_return *= PREDICTION_SHRINK

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
