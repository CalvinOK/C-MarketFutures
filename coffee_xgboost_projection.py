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

COFFEE_FILE = BASE_DIR / "CoffeeCData.csv"
SOY_FILE = BASE_DIR / "US Soybeans Futures Historical Data.csv"
SUGAR_FILE = BASE_DIR / "US Sugar #11 Futures Historical Data.csv"
FX_FILE = BASE_DIR / "USD_BRLT Historical Data.csv"
CLIMATE_FILE = BASE_DIR / "coffee_climate_sao_paulo.csv"

USE_CLIMATE = False  # default off because climate file ends much earlier than futures/FX data
FORECAST_HORIZON = 60  # business days to project forward
TEST_SIZE = 0.20
RANDOM_STATE = 42

# This model uses only historical exogenous lags, so it can forecast until the
# smallest lag in this set is exhausted. Example: if the smallest exogenous lag
# is 10, the model can use actual known exogenous data for 10 future business days.
BRIDGE_EXOG_LAGS = {
    "sugar_return_pct": [28],
    "usd_brl_return_pct": [17],
    "soybeans_return_pct": [10],
}

# Coffee autoregressive features used by both models
COFFEE_LAGS = [1, 2, 3, 5, 10, 20]
ROLL_WINDOWS = [5, 10, 20]

# Climate lags only matter if USE_CLIMATE=True and enough overlap exists
CLIMATE_LAGS = [1, 3, 5]

# Output files
MERGED_FILE = BASE_DIR / "model_merged_data.csv"
FORECAST_FILE = BASE_DIR / "coffee_forecast_output.csv"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "primary_feature_importance.csv"
PLOT_FILE = BASE_DIR / "coffee_projection_plot.png"

# ============================================================
# HELPERS
# ============================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def load_market_file(file_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = clean_numeric_series(df["Price"])

    if "Change %" in df.columns:
        df[f"{value_name}_return_pct"] = clean_numeric_series(df["Change %"])
    else:
        df = df.sort_values("Date")
        df[f"{value_name}_return_pct"] = df["Price"].pct_change() * 100

    df = df.rename(columns={"Price": value_name})
    keep = ["Date", value_name, f"{value_name}_return_pct"]
    df = df[keep].dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_climate_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()


def next_business_days(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last_date + pd.offsets.BDay(1), periods=periods)


def train_test_split_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict:
    return {
        "model": label,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_merged_dataset() -> pd.DataFrame:
    coffee = load_market_file(COFFEE_FILE, "coffee_c")
    soy = load_market_file(SOY_FILE, "soybeans")
    sugar = load_market_file(SUGAR_FILE, "sugar")
    fx = load_market_file(FX_FILE, "usd_brl")

    df = coffee.merge(soy, on="Date", how="left")
    df = df.merge(sugar, on="Date", how="left")
    df = df.merge(fx, on="Date", how="left")

    # Keep only coffee trading dates; forward-fill exogenous holidays onto coffee dates.
    exog_cols = [c for c in df.columns if c != "Date"]
    df = df.sort_values("Date").reset_index(drop=True)
    df[exog_cols] = df[exog_cols].ffill()

    if USE_CLIMATE and CLIMATE_FILE.exists():
        climate = load_climate_file(CLIMATE_FILE)
        df = df.merge(climate, on="Date", how="left")
        climate_cols = [c for c in ["tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"] if c in df.columns]
        # Light fill only within observed climate history; after climate end date values stay NaN.
        if climate_cols:
            df[climate_cols] = df[climate_cols].ffill(limit=5)

    # Target is next-trading-day coffee return
    df["target_return_pct"] = df["coffee_c_return_pct"].shift(-1)

    # Coffee autoregressive features
    add_lag_features(df, "coffee_c_return_pct", COFFEE_LAGS)
    add_rolling_features(df, "coffee_c_return_pct", ROLL_WINDOWS)

    # Bridge exogenous features using only lagged values, so we can forecast multiple days
    # using actual known exogenous data before switching to the fallback model.
    for col, lags in BRIDGE_EXOG_LAGS.items():
        if col in df.columns:
            add_lag_features(df, col, lags)

    # Optional climate features
    if USE_CLIMATE:
        climate_bases = [c for c in ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"] if c in df.columns]
        for col in climate_bases:
            add_lag_features(df, col, CLIMATE_LAGS)

    df.to_csv(MERGED_FILE, index=False)
    return df


def get_primary_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    cols += [f"coffee_c_return_pct_lag_{lag}" for lag in COFFEE_LAGS if f"coffee_c_return_pct_lag_{lag}" in df.columns]
    cols += [f"coffee_c_return_pct_roll_mean_{w}" for w in ROLL_WINDOWS if f"coffee_c_return_pct_roll_mean_{w}" in df.columns]
    cols += [f"coffee_c_return_pct_roll_std_{w}" for w in ROLL_WINDOWS if f"coffee_c_return_pct_roll_std_{w}" in df.columns]

    for base, lags in BRIDGE_EXOG_LAGS.items():
        cols += [f"{base}_lag_{lag}" for lag in lags if f"{base}_lag_{lag}" in df.columns]

    if USE_CLIMATE:
        climate_bases = [c for c in ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"] if c in df.columns]
        for base in climate_bases:
            cols += [f"{base}_lag_{lag}" for lag in CLIMATE_LAGS if f"{base}_lag_{lag}" in df.columns]

    return cols


def get_fallback_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    cols += [f"coffee_c_return_pct_lag_{lag}" for lag in COFFEE_LAGS if f"coffee_c_return_pct_lag_{lag}" in df.columns]
    cols += [f"coffee_c_return_pct_roll_mean_{w}" for w in ROLL_WINDOWS if f"coffee_c_return_pct_roll_mean_{w}" in df.columns]
    cols += [f"coffee_c_return_pct_roll_std_{w}" for w in ROLL_WINDOWS if f"coffee_c_return_pct_roll_std_{w}" in df.columns]
    return cols


def fit_models(df: pd.DataFrame):
    primary_features = get_primary_feature_columns(df)
    fallback_features = get_fallback_feature_columns(df)

    primary_df = df.dropna(subset=primary_features + ["target_return_pct", "coffee_c"]).copy()
    fallback_df = df.dropna(subset=fallback_features + ["target_return_pct", "coffee_c"]).copy()

    train_primary, test_primary = train_test_split_time(primary_df, TEST_SIZE)
    train_fallback, test_fallback = train_test_split_time(fallback_df, TEST_SIZE)

    X_train_p = train_primary[primary_features]
    y_train_p = train_primary["target_return_pct"]
    X_test_p = test_primary[primary_features]
    y_test_p = test_primary["target_return_pct"]

    X_train_f = train_fallback[fallback_features]
    y_train_f = train_fallback["target_return_pct"]
    X_test_f = test_fallback[fallback_features]
    y_test_f = test_fallback["target_return_pct"]

    primary_model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
    )
    fallback_model = XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
    )

    primary_model.fit(X_train_p, y_train_p)
    fallback_model.fit(X_train_f, y_train_f)

    primary_pred = primary_model.predict(X_test_p)
    fallback_pred = fallback_model.predict(X_test_f)

    metrics = pd.DataFrame([
        evaluate_predictions(y_test_p, primary_pred, "primary_bridge_model"),
        evaluate_predictions(y_test_f, fallback_pred, "fallback_autoregressive_model"),
    ])

    importances = pd.DataFrame({
        "feature": primary_features,
        "importance": primary_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    return {
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "primary_features": primary_features,
        "fallback_features": fallback_features,
        "metrics": metrics,
        "importances": importances,
    }


def build_future_row(history: pd.DataFrame, future_exog_actual: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    row = {"Date": date}

    # Initialize all series we may need as NaN first.
    for col in [
        "coffee_c", "coffee_c_return_pct", "soybeans", "soybeans_return_pct",
        "sugar", "sugar_return_pct", "usd_brl", "usd_brl_return_pct",
        "tmax", "tmax_change_pct", "tmin", "tmin_change_pct", "rainfall"
    ]:
        row[col] = np.nan

    # Pull any actual exogenous values that are still inferable from the historical window.
    # We do not forecast exogenous variables here; instead we use actual lagged exogenous data
    # as far out as those lags allow, then switch to the fallback model.
    if date in set(future_exog_actual["Date"]):
        matched = future_exog_actual.loc[future_exog_actual["Date"] == date]
        if not matched.empty:
            for c in matched.columns:
                if c != "Date":
                    row[c] = matched.iloc[0][c]

    return pd.Series(row)


def recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Recompute lag and rolling features after appending a new predicted row.
    for lag in COFFEE_LAGS:
        df[f"coffee_c_return_pct_lag_{lag}"] = df["coffee_c_return_pct"].shift(lag)
    for w in ROLL_WINDOWS:
        df[f"coffee_c_return_pct_roll_mean_{w}"] = df["coffee_c_return_pct"].rolling(w).mean()
        df[f"coffee_c_return_pct_roll_std_{w}"] = df["coffee_c_return_pct"].rolling(w).std()

    for base, lags in BRIDGE_EXOG_LAGS.items():
        if base in df.columns:
            for lag in lags:
                df[f"{base}_lag_{lag}"] = df[base].shift(lag)

    if USE_CLIMATE:
        climate_bases = [c for c in ["tmax", "tmin", "rainfall", "tmax_change_pct", "tmin_change_pct"] if c in df.columns]
        for base in climate_bases:
            for lag in CLIMATE_LAGS:
                df[f"{base}_lag_{lag}"] = df[base].shift(lag)
    return df


def forecast_future(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    primary_model = model_bundle["primary_model"]
    fallback_model = model_bundle["fallback_model"]
    primary_features = model_bundle["primary_features"]
    fallback_features = model_bundle["fallback_features"]

    history = df.copy().sort_values("Date").reset_index(drop=True)
    history = recompute_features(history)

    future_dates = next_business_days(history["Date"].max(), FORECAST_HORIZON)

    # Only actual historical exogenous values are available. The bridge model can keep using
    # these values until the smallest exogenous lag is exhausted.
    min_bridge_lag = min(min(v) for v in BRIDGE_EXOG_LAGS.values()) if BRIDGE_EXOG_LAGS else 0

    forecast_rows = []
    last_actual_price = float(history["coffee_c"].dropna().iloc[-1])

    # Use the existing historical data frame as the state that grows with predictions.
    state = history.copy()

    for step_ahead, date in enumerate(future_dates, start=1):
        new_row = build_future_row(state, history[[c for c in history.columns if c in [
            "Date", "soybeans", "soybeans_return_pct", "sugar", "sugar_return_pct",
            "usd_brl", "usd_brl_return_pct", "tmax", "tmax_change_pct", "tmin",
            "tmin_change_pct", "rainfall"
        ]]], date)

        state = pd.concat([state, pd.DataFrame([new_row])], ignore_index=True)
        state = recompute_features(state)

        model_type = "primary_bridge_model" if step_ahead <= min_bridge_lag else "fallback_autoregressive_model"
        features = primary_features if model_type == "primary_bridge_model" else fallback_features
        model = primary_model if model_type == "primary_bridge_model" else fallback_model

        X_row = state.loc[[state.index[-1]], features]

        # If the primary model row still has NaNs because not enough lags exist, fall back early.
        if X_row.isna().any(axis=1).iloc[0]:
            model_type = "fallback_autoregressive_model"
            features = fallback_features
            model = fallback_model
            X_row = state.loc[[state.index[-1]], features]

        pred_return = float(model.predict(X_row)[0])

        prev_price = float(state.loc[state.index[-2], "coffee_c"])
        pred_price = prev_price * (1 + pred_return / 100.0)

        state.loc[state.index[-1], "coffee_c_return_pct"] = pred_return
        state.loc[state.index[-1], "coffee_c"] = pred_price
        state = recompute_features(state)

        forecast_rows.append({
            "Date": date,
            "predicted_return_pct": pred_return,
            "predicted_price": pred_price,
            "model_used": model_type,
            "step_ahead": step_ahead,
        })

        last_actual_price = pred_price

    return pd.DataFrame(forecast_rows)


def plot_history_and_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    hist = df[["Date", "coffee_c"]].dropna().sort_values("Date")

    plt.figure(figsize=(14, 7))
    plt.plot(hist["Date"], hist["coffee_c"], color="black", linewidth=1.5, label="Historical coffee price")
    plt.plot(forecast_df["Date"], forecast_df["predicted_price"], color="blue", linewidth=2.0, label="Projected coffee price")

    if not forecast_df.empty:
        plt.axvline(forecast_df["Date"].min(), linestyle="--", color="gray", alpha=0.7)

    plt.title("Coffee C Futures: Historical (black) and Projected (blue)")
    plt.xlabel("Date")
    plt.ylabel("Coffee C Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()


def main():
    print("Building merged dataset...")
    df = build_merged_dataset()

    print("Training models...")
    model_bundle = fit_models(df)
    print("\nModel metrics:")
    print(model_bundle["metrics"].to_string(index=False))

    print("\nTop primary-model features:")
    print(model_bundle["importances"].head(15).to_string(index=False))

    print("\nForecasting future...")
    forecast_df = forecast_future(df, model_bundle)
    forecast_df.to_csv(FORECAST_FILE, index=False)

    plot_history_and_forecast(df, forecast_df)

    print("\nSaved files:")
    print(f"- {MERGED_FILE}")
    print(f"- {FEATURE_IMPORTANCE_FILE}")
    print(f"- {FORECAST_FILE}")
    print(f"- {PLOT_FILE}")

    if not forecast_df.empty:
        print("\nForecast head:")
        print(forecast_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
