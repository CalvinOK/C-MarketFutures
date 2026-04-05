
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
LOGDATA_DIR = BASE_DIR / "logdata"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COFFEE_FILE = LOGDATA_DIR / "CoffeeCData_log_returns.csv"
SOY_FILE = LOGDATA_DIR / "US Soybeans Futures Historical Data_log_returns.csv"
SUGAR_FILE = LOGDATA_DIR / "US Sugar #11 Futures Historical Data_log_returns.csv"
FX_FILE = LOGDATA_DIR / "USD_BRLT Historical Data_log_returns.csv"
CLIMATE_FILE = LOGDATA_DIR / "coffee_climate_sao_paulo.csv"

USE_CLIMATE = False
HISTORY_PLOT_DAYS = 365
FORECAST_WEEKS = 52
BUSINESS_DAYS_PER_WEEK = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Keep lags short and realistic
COFFEE_LAGS = [1, 2, 3, 5, 10, 20]
EXOG_LAGS = [0, 1, 2, 3, 5]
ROLL_WINDOWS = [5, 10, 20]
CLIMATE_LAGS = [1, 3, 5]

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


def load_log_return_market_file(file_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        raise ValueError(f"{file_path} is missing a Date column")
    if "Price" not in df.columns:
        raise ValueError(f"{file_path} is missing a Price column")

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
        .reset_index(drop=True)
        .rename(columns={"Price": value_name, log_return_col: f"{value_name}_log_return"})
    )
    return out


def load_climate_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df.sort_values("Date").reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int], include_zero: bool = False) -> None:
    for lag in lags:
        if lag == 0 and include_zero:
            df[f"{col}_lag_0"] = safe_numeric(df[col])
        elif lag > 0:
            df[f"{col}_lag_{lag}"] = safe_numeric(df[col]).shift(lag)


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    series = safe_numeric(df[col])
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = series.rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = series.rolling(w).std()


def cumulative_forward_log_return(series: pd.Series, horizon_days: int) -> pd.Series:
    # cumulative future log return over the next horizon_days business days
    acc = pd.Series(0.0, index=series.index)
    for i in range(1, horizon_days + 1):
        acc = acc + series.shift(-i)
    return acc


# ============================================================
# DATASET
# ============================================================

def build_merged_dataset() -> pd.DataFrame:
    coffee = load_log_return_market_file(COFFEE_FILE, "coffee_c")
    soy = load_log_return_market_file(SOY_FILE, "soybeans")
    sugar = load_log_return_market_file(SUGAR_FILE, "sugar")
    fx = load_log_return_market_file(FX_FILE, "usd_brl")

    df = coffee.merge(soy, on="Date", how="left")
    df = df.merge(sugar, on="Date", how="left")
    df = df.merge(fx, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep coffee trading dates and fill exogenous market observations onto them.
    exog_cols = [c for c in df.columns if c not in ["Date", "coffee_c", "coffee_c_log_return"]]
    df[exog_cols] = df[exog_cols].ffill()

    if USE_CLIMATE and CLIMATE_FILE.exists():
        climate = load_climate_file(CLIMATE_FILE)
        df = df.merge(climate, on="Date", how="left")
        climate_cols = [c for c in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"] if c in df.columns]
        if climate_cols:
            df[climate_cols] = df[climate_cols].ffill(limit=5)

    # Feature engineering from actual observed history only
    add_lag_features(df, "coffee_c_log_return", COFFEE_LAGS)
    add_rolling_features(df, "coffee_c_log_return", ROLL_WINDOWS)

    for col in ["soybeans_log_return", "sugar_log_return", "usd_brl_log_return"]:
        if col in df.columns:
            add_lag_features(df, col, EXOG_LAGS, include_zero=True)

    if USE_CLIMATE:
        for col in [c for c in ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"] if c in df.columns]:
            add_lag_features(df, col, CLIMATE_LAGS)

    df.to_csv(MERGED_FILE, index=False)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    cols += [f"coffee_c_log_return_lag_{lag}" for lag in COFFEE_LAGS if f"coffee_c_log_return_lag_{lag}" in df.columns]
    cols += [f"coffee_c_log_return_roll_mean_{w}" for w in ROLL_WINDOWS if f"coffee_c_log_return_roll_mean_{w}" in df.columns]
    cols += [f"coffee_c_log_return_roll_std_{w}" for w in ROLL_WINDOWS if f"coffee_c_log_return_roll_std_{w}" in df.columns]

    for base in ["soybeans_log_return", "sugar_log_return", "usd_brl_log_return"]:
        cols += [f"{base}_lag_{lag}" for lag in EXOG_LAGS if f"{base}_lag_{lag}" in df.columns]

    if USE_CLIMATE:
        for base in [c for c in ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"] if c in df.columns]:
            cols += [f"{base}_lag_{lag}" for lag in CLIMATE_LAGS if f"{base}_lag_{lag}" in df.columns]

    return cols


# ============================================================
# MODELING
# ============================================================

def fit_direct_models(df: pd.DataFrame) -> dict:
    features = get_feature_columns(df)

    models: dict[int, XGBRegressor] = {}
    clips: dict[int, tuple[float, float]] = {}
    metrics_rows: list[dict] = []

    # Save H=1 feature importance as a representative file
    h1_importance = None

    for horizon_weeks in range(1, FORECAST_WEEKS + 1):
        horizon_days = horizon_weeks * BUSINESS_DAYS_PER_WEEK
        target_col = f"target_log_return_{horizon_weeks}w"

        work = df.copy()
        work[target_col] = cumulative_forward_log_return(work["coffee_c_log_return"], horizon_days)

        model_df = work.dropna(subset=features + [target_col, "coffee_c"]).copy()
        if len(model_df) < 200:
            # skip impossible horizons if too little data
            continue

        train_df, test_df = train_test_split_time(model_df, TEST_SIZE)

        X_train = train_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = train_df[target_col]
        X_test = test_df[features].apply(pd.to_numeric, errors="coerce")
        y_test = test_df[target_col]

        model = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=8,
            random_state=RANDOM_STATE + horizon_weeks,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        baseline = np.zeros_like(y_test.to_numpy())

        metrics_rows.append(evaluate_predictions(y_test, y_pred, "direct_xgboost", horizon_weeks))
        metrics_rows.append(evaluate_predictions(y_test, baseline, "zero_return_baseline", horizon_weeks))

        q05 = float(y_train.quantile(0.05))
        q95 = float(y_train.quantile(0.95))
        clips[horizon_weeks] = (q05, q95)
        models[horizon_weeks] = model

        if horizon_weeks == 1:
            h1_importance = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(METRICS_FILE, index=False)

    if h1_importance is not None:
        h1_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    return {
        "models": models,
        "features": features,
        "clips": clips,
        "metrics": metrics,
        "h1_importance": h1_importance,
    }


# ============================================================
# FORECASTING
# ============================================================

def forecast_direct(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    features = model_bundle["features"]
    models = model_bundle["models"]
    clips = model_bundle["clips"]

    latest = df.dropna(subset=features + ["coffee_c"]).sort_values("Date").iloc[-1].copy()
    X_latest = pd.DataFrame([latest[features]]).apply(pd.to_numeric, errors="coerce")

    last_date = pd.to_datetime(latest["Date"])
    last_price = float(latest["coffee_c"])

    forecast_rows: list[dict] = []
    future_dates = next_business_days(last_date, FORECAST_WEEKS * BUSINESS_DAYS_PER_WEEK)

    for horizon_weeks, model in models.items():
        pred = float(model.predict(X_latest)[0])

        # shrink toward zero and clip to realistic training range
        lower, upper = clips[horizon_weeks]
        pred = 0.50 * pred
        pred = float(np.clip(pred, lower, upper))

        horizon_days = horizon_weeks * BUSINESS_DAYS_PER_WEEK
        target_date = future_dates[horizon_days - 1]
        projected_price = float(last_price * np.exp(pred))

        forecast_rows.append({
            "forecast_origin_date": last_date,
            "horizon_weeks": horizon_weeks,
            "horizon_days": horizon_days,
            "projected_date": target_date,
            "predicted_cumulative_log_return": pred,
            "projected_price": projected_price,
        })

    forecast_df = pd.DataFrame(forecast_rows).sort_values("projected_date").reset_index(drop=True)
    forecast_df.to_csv(FORECAST_FILE, index=False)
    return forecast_df


# ============================================================
# PLOTTING
# ============================================================

def plot_history_and_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    hist = df[["Date", "coffee_c"]].dropna().sort_values("Date")
    if hist.empty:
        raise ValueError("No historical coffee price data to plot.")

    cutoff = hist["Date"].max() - pd.Timedelta(days=HISTORY_PLOT_DAYS)
    hist_plot = hist.loc[hist["Date"] >= cutoff].copy()

    plt.figure(figsize=(14, 7))
    plt.plot(
        hist_plot["Date"],
        hist_plot["coffee_c"],
        color="black",
        linewidth=1.8,
        label="Historical coffee price"
    )
    plt.plot(
        forecast_df["projected_date"],
        forecast_df["projected_price"],
        color="blue",
        linewidth=2.0,
        label="Projected coffee price"
    )

    if not forecast_df.empty:
        plt.axvline(hist["Date"].max(), linestyle="--", color="gray", alpha=0.7)

    plt.title("Coffee C Futures: Last 1 Year Historical (black) and 1 Year Projection (blue)")
    plt.xlabel("Date")
    plt.ylabel("Coffee C Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Building merged dataset...")
    df = build_merged_dataset()

    print("Training direct multi-horizon models...")
    model_bundle = fit_direct_models(df)

    metrics = model_bundle["metrics"]
    summary = (
        metrics.pivot(index="horizon_weeks", columns="model", values="r2")
        if not metrics.empty else pd.DataFrame()
    )

    print("\nR² by horizon:")
    if not summary.empty:
        print(summary.head(12).to_string())
    else:
        print("No metrics available.")

    if model_bundle["h1_importance"] is not None:
        print("\nTop H=1 week features:")
        print(model_bundle["h1_importance"].head(15).to_string(index=False))

    print("\nForecasting from latest actual data only...")
    forecast_df = forecast_direct(df, model_bundle)

    plot_history_and_forecast(df, forecast_df)

    print("\nSaved files:")
    print(f"- {MERGED_FILE}")
    print(f"- {METRICS_FILE}")
    print(f"- {FEATURE_IMPORTANCE_FILE}")
    print(f"- {FORECAST_FILE}")
    print(f"- {PLOT_FILE}")

    if not forecast_df.empty:
        print("\nForecast head:")
        print(forecast_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
