from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from old.data_sources import (
    DEFAULT_REGIONS,
    DataSourceError,
    combine_weighted_regions,
    fetch_alpha_vantage_economic_series,
    fetch_alpha_vantage_fx_monthly,
    fetch_open_meteo_climate_monthly,
    fetch_open_meteo_historical_monthly,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coffee price forecast with XGBoost and API-updated exogenous factors.")
    p.add_argument("--csv", default="CoffeeCData.csv", help="Path to CoffeeCData CSV file.")
    p.add_argument("--outdir", default="outputs", help="Directory for output files.")
    p.add_argument("--horizon-months", type=int, default=60, help="Forecast horizon in months. Default 60 = 5 years.")
    p.add_argument("--climate-model", default="MRI_AGCM3_2_S", help="Open-Meteo climate model name.")
    p.add_argument(
        "--signal-profile",
        choices=["core", "full"],
        default="core",
        help="Alpha Vantage signal set to use. core is faster; full includes all market series.",
    )
    return p.parse_args()


def parse_volume(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in {"-", "", "nan", "None"}:
        return np.nan
    mult = 1.0
    if s.endswith("K"):
        mult = 1e3
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1e9
        s = s[:-1]
    return float(s) * mult


def parse_percent(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("%", "").replace(",", "")
    if s in {"", "-", "nan", "None"}:
        return np.nan
    return float(s) / 100.0


def load_coffee_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = {"Date", "Price", "Open", "High", "Low", "Vol.", "Change %"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Price", "Open", "High", "Low"]:
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(",", ""), errors="coerce")
    out["Vol."] = out["Vol."].apply(parse_volume)
    out["Change %"] = out["Change %"].apply(parse_percent)
    out = out.dropna(subset=["Date", "Price"]).sort_values("Date")
    out = out.set_index("Date")
    return out


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.DataFrame(index=df.resample("ME").last().index)
    monthly["price_nominal"] = df["Price"].resample("ME").last()
    monthly["open"] = df["Open"].resample("ME").first()
    monthly["high"] = df["High"].resample("ME").max()
    monthly["low"] = df["Low"].resample("ME").min()
    monthly["volume"] = df["Vol."].resample("ME").sum(min_count=1)
    monthly["change_pct_avg"] = df["Change %"].resample("ME").mean()
    return monthly.dropna(subset=["price_nominal"])


def seasonal_trend_projection(series: pd.Series, future_index: pd.DatetimeIndex, lookback_months: int = 60) -> pd.Series:
    s = series.dropna().copy()
    if s.empty:
        return pd.Series(index=future_index, dtype=float)

    s = s.iloc[-lookback_months:] if len(s) > lookback_months else s
    t = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    slope, intercept = np.polyfit(t, y, 1) if len(s) >= 2 else (0.0, y[-1])

    seasonal = s.groupby(s.index.month).mean()
    seasonal -= seasonal.mean()

    values = []
    for i, dt in enumerate(future_index, start=1):
        trend = intercept + slope * (len(s) - 1 + i)
        month_adj = seasonal.get(dt.month, 0.0)
        values.append(trend + month_adj)

    proj = pd.Series(values, index=future_index)
    return proj.ffill().bfill()


def fetch_external_factors(
    start_date: str,
    end_date: str,
    future_start_date: str,
    future_end_date: str,
    climate_model: str,
    signal_profile: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _maybe_fetch(fetch_fn, *args):
        try:
            return fetch_fn(*args)
        except DataSourceError:
            return None

    # Historical market and macro factors
    cpi = _maybe_fetch(fetch_alpha_vantage_economic_series, "CPI")
    sugar = _maybe_fetch(fetch_alpha_vantage_economic_series, "SUGAR")
    coffee_global = _maybe_fetch(fetch_alpha_vantage_economic_series, "COFFEE")
    brlusd = _maybe_fetch(fetch_alpha_vantage_fx_monthly, "BRL", "USD")
    copusd = _maybe_fetch(fetch_alpha_vantage_fx_monthly, "COP", "USD")

    market_frames = [frame for frame in [cpi, sugar, coffee_global, brlusd, copusd] if frame is not None]
    if signal_profile == "full":
        wheat = _maybe_fetch(fetch_alpha_vantage_economic_series, "WHEAT")
        cotton = _maybe_fetch(fetch_alpha_vantage_economic_series, "COTTON")
        wti = _maybe_fetch(fetch_alpha_vantage_economic_series, "WTI")
        market_frames.extend([frame for frame in [wheat, cotton, wti] if frame is not None])

    market = pd.concat(market_frames, axis=1).sort_index() if market_frames else pd.DataFrame()

    # Historical weather in major coffee regions
    hist_regions = [fetch_open_meteo_historical_monthly(r, start_date, end_date) for r in DEFAULT_REGIONS]
    weather_hist = combine_weighted_regions(hist_regions, DEFAULT_REGIONS)

    # Future climate
    future_regions = [
        fetch_open_meteo_climate_monthly(r, start_date=future_start_date, end_date=future_end_date, model=climate_model)
        for r in DEFAULT_REGIONS
    ]
    weather_future = combine_weighted_regions(future_regions, DEFAULT_REGIONS)

    hist = pd.concat([market, weather_hist], axis=1).sort_index()
    future = weather_future.sort_index()
    return hist, future


def adjust_for_inflation(monthly: pd.DataFrame, cpi: pd.Series | None) -> pd.DataFrame:
    merged = monthly.copy()
    if cpi is None or cpi.dropna().empty:
        merged["price_real"] = merged["price_nominal"]
        return merged

    base_cpi = cpi.dropna().iloc[-1]
    # If the cpi column is already present in the DataFrame (joined upstream), use it
    # directly rather than re-joining, which would raise a duplicate-column error.
    if "cpi" not in merged.columns:
        merged = merged.join(cpi.rename("cpi"), how="left")
    merged["cpi"] = merged["cpi"].interpolate().ffill().bfill()
    merged["price_real"] = merged["price_nominal"] * (base_cpi / merged["cpi"])
    merged[["open", "high", "low"]] = merged[["open", "high", "low"]].multiply(base_cpi / merged["cpi"], axis=0)
    return merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_index()
    data["month"] = data.index.month.astype("int16")
    data["quarter"] = data.index.quarter.astype("int16")
    data["year"] = data.index.year.astype("int16")

    for lag in [1, 2, 3, 6, 12]:
        data[f"price_real_lag_{lag}"] = data["price_real"].shift(lag)
        data[f"change_lag_{lag}"] = data["change_pct_avg"].shift(lag)
        data[f"vol_lag_{lag}"] = data["volume"].shift(lag)

    for window in [3, 6, 12]:
        data[f"price_real_rollmean_{window}"] = data["price_real"].shift(1).rolling(window).mean()
        data[f"price_real_rollstd_{window}"] = data["price_real"].shift(1).rolling(window).std()
        data[f"volume_rollmean_{window}"] = data["volume"].shift(1).rolling(window).mean()

    # Weather lags
    weather_cols = [c for c in data.columns if c.startswith("weather_")]
    market_cols = [
        c for c in data.columns if c in {"sugar", "wheat", "cotton", "coffee", "wti", "fx_brlusd", "fx_copusd", "cpi"}
    ]

    for col in weather_cols + market_cols:
        for lag in [1, 3, 6, 12]:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        yoy = data[col].pct_change(12)
        # pct_change can produce ±inf when the denominator is zero; XGBoost cannot
        # handle inf values, so replace them with NaN and let interpolation fill later.
        data[f"{col}_yoy"] = yoy.replace([np.inf, -np.inf], np.nan)

    # Targets are monthly real price levels.
    return data


def train_xgboost(feature_df: pd.DataFrame) -> Tuple[XGBRegressor, List[str], Dict[str, float], pd.DataFrame]:
    model_df = feature_df.dropna(subset=["price_real"]).copy()
    feature_cols = [c for c in model_df.columns if c not in {"price_nominal", "price_real"}]
    model_df = model_df.dropna(subset=feature_cols)

    if len(model_df) < 48:
        raise ValueError("Not enough monthly observations after feature engineering. Need at least ~48 months.")

    val_months = min(24, max(12, len(model_df) // 5))
    train_df = model_df.iloc[:-val_months]
    val_df = model_df.iloc[-val_months:]

    X_train, y_train = train_df[feature_cols], train_df["price_real"]
    X_val, y_val = val_df[feature_cols], val_df["price_real"]

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred = model.predict(X_val)
    metrics = {
        "validation_mae": float(mean_absolute_error(y_val, pred)),
        "validation_rmse": float(math.sqrt(mean_squared_error(y_val, pred))),
        "training_months": int(len(train_df)),
        "validation_months": int(len(val_df)),
    }

    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return model, feature_cols, metrics, fi


def project_future_exogenous(full_df: pd.DataFrame, weather_future: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future_index = pd.date_range(full_df.index.max() + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    future = pd.DataFrame(index=future_index)

    # Known / projected weather
    for col in weather_future.columns:
        future[col] = weather_future.reindex(future_index)[col]
        if future[col].isna().all():
            future[col] = seasonal_trend_projection(full_df[col], future_index)
        else:
            future[col] = future[col].interpolate().ffill().bfill()

    # Unknown market factors -> projected from recent monthly history.
    # Use .get() so that columns absent due to API failures return an empty Series
    # rather than raising a KeyError.
    for col in ["cpi", "sugar", "wheat", "cotton", "coffee", "wti", "fx_brlusd", "fx_copusd"]:
        future[col] = seasonal_trend_projection(full_df.get(col, pd.Series(dtype=float)), future_index)

    # Set placeholders for endogenous items, filled recursively later
    future["price_nominal"] = np.nan
    future["price_real"] = np.nan
    future["open"] = np.nan
    future["high"] = np.nan
    future["low"] = np.nan
    future["volume"] = seasonal_trend_projection(full_df["volume"], future_index)
    future["change_pct_avg"] = 0.0
    return future


def recursive_forecast(model: XGBRegressor, history: pd.DataFrame, weather_future: pd.DataFrame, horizon: int) -> pd.DataFrame:
    combined = history.copy().sort_index()
    future_base = project_future_exogenous(combined, weather_future, horizon)
    combined = pd.concat([combined, future_base], axis=0).sort_index()

    forecast_rows = []
    future_dates = future_base.index.tolist()

    for dt in future_dates:
        features_df = build_features(combined.loc[:dt].copy())
        # Fill NaN values across the whole frame so that lag features for future rows
        # (e.g. open/high/low placeholders) are propagated forward correctly.
        # A single-row slice cannot be forward-filled, so we fill first and then select.
        features_df = features_df.ffill().bfill()
        row = features_df.loc[[dt]].drop(columns=[c for c in ["price_nominal", "price_real"] if c in features_df.columns])
        pred_real = float(model.predict(row)[0])

        prev_real = combined.loc[:dt].iloc[-2]["price_real"]
        change = 0.0 if pd.isna(prev_real) or prev_real == 0 else (pred_real / prev_real) - 1.0

        combined.at[dt, "price_real"] = pred_real
        combined.at[dt, "change_pct_avg"] = change
        combined.at[dt, "price_nominal"] = pred_real * (combined.at[dt, "cpi"] / combined["cpi"].dropna().iloc[-1])
        combined.at[dt, "open"] = combined.at[dt, "price_nominal"]
        combined.at[dt, "high"] = combined.at[dt, "price_nominal"]
        combined.at[dt, "low"] = combined.at[dt, "price_nominal"]
        forecast_rows.append({
            "Date": dt,
            "price_real_forecast": pred_real,
            "price_nominal_forecast": combined.at[dt, "price_nominal"],
        })

    forecast = pd.DataFrame(forecast_rows).set_index("Date")
    return combined, forecast


def plot_history_and_forecast(history: pd.DataFrame, forecast: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(13, 7))
    plt.plot(history.index, history["price_nominal"], color="black", linewidth=1.8, label="Historical")
    plt.plot(forecast.index, forecast["price_nominal_forecast"], color="blue", linewidth=2.0, label="Forecast")
    plt.title("Coffee Price History and 5-Year Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_coffee_csv(args.csv)
    monthly = resample_monthly(raw)
    start_date = monthly.index.min().replace(day=1).strftime("%Y-%m-%d")
    weather_end = min(monthly.index.max().normalize(), pd.Timestamp.today().normalize())
    end_date = weather_end.strftime("%Y-%m-%d")
    future_start = (weather_end + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    future_end = (monthly.index.max() + pd.DateOffset(months=args.horizon_months)).strftime("%Y-%m-%d")

    external_hist, weather_future = fetch_external_factors(
        start_date,
        end_date,
        future_start,
        future_end,
        args.climate_model,
        args.signal_profile,
    )
    base = monthly.join(external_hist, how="left").sort_index()
    base = adjust_for_inflation(base, base["cpi"] if "cpi" in base.columns else None)
    base = base.interpolate().ffill().bfill()

    feature_df = build_features(base)
    model, feature_cols, metrics, fi = train_xgboost(feature_df)
    combined, forecast = recursive_forecast(model, base, weather_future, args.horizon_months)

    combined.to_csv(outdir / "coffee_model_dataset.csv", index=True)
    forecast.to_csv(outdir / "coffee_forecast_5y.csv", index=True)
    fi.to_csv(outdir / "feature_importance.csv", index=False)
    with open(outdir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_history_and_forecast(base, forecast, outdir / "coffee_history_forecast.png")

    print("Saved:")
    print(outdir / "coffee_model_dataset.csv")
    print(outdir / "coffee_forecast_5y.csv")
    print(outdir / "feature_importance.csv")
    print(outdir / "model_metrics.json")
    print(outdir / "coffee_history_forecast.png")


if __name__ == "__main__":
    try:
        main()
    except DataSourceError as exc:
        raise SystemExit(f"Data source error: {exc}")
