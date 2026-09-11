from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FORECAST_HORIZONS = tuple(range(1, 32))
MIN_TRAIN_ROWS = 400
BUSINESS_DAYS = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class ForecastResult:
    as_of: str
    current_price: float
    unit: str
    forecast: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    feature_count: int
    rows_used: int


def normalize_history(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Coffee C history is empty")

    date_column = next((name for name in ("market_date", "Date", "date") if name in frame.columns), None)
    price_column = next((name for name in ("Price", "price", "close", "Close") if name in frame.columns), None)
    if not date_column or not price_column:
        raise ValueError("Coffee C history requires Date and Price columns")

    frame = frame[[date_column, price_column]].rename(columns={date_column: "Date", price_column: "price"})
    frame["Date"] = pd.to_datetime(frame["Date"], format="mixed", errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["Date", "price"])
    frame = frame[frame["price"] > 0]
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if len(frame) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough valid Coffee C history: {len(frame)} rows; need at least {MIN_TRAIN_ROWS}")
    return frame


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    log_price = np.log(frame["price"])
    frame["log_return"] = log_price.diff()

    for lag in (1, 2, 3, 5, 10, 20):
        frame[f"return_lag_{lag}"] = frame["log_return"].shift(lag)
    for window in (5, 10, 20, 60):
        frame[f"return_mean_{window}"] = frame["log_return"].rolling(window, min_periods=max(3, window // 2)).mean()
        frame[f"return_std_{window}"] = frame["log_return"].rolling(window, min_periods=max(3, window // 2)).std()
        frame[f"momentum_{window}"] = log_price - log_price.shift(window)
        frame[f"sma_{window}"] = frame["price"].rolling(window, min_periods=max(3, window // 2)).mean()
        frame[f"price_vs_sma_{window}"] = frame["price"] / frame[f"sma_{window}"] - 1.0

    for window in (60, 252):
        rolling_mean = frame["price"].rolling(window, min_periods=max(20, window // 2)).mean()
        rolling_std = frame["price"].rolling(window, min_periods=max(20, window // 2)).std()
        frame[f"zscore_{window}"] = (frame["price"] - rolling_mean) / rolling_std.replace(0, np.nan)
        frame[f"high_position_{window}"] = frame["price"] / frame["price"].rolling(window, min_periods=max(20, window // 2)).max()
        frame[f"low_position_{window}"] = frame["price"] / frame["price"].rolling(window, min_periods=max(20, window // 2)).min()

    day_of_year = frame["Date"].dt.dayofyear
    frame["month_sin"] = np.sin(2 * np.pi * frame["Date"].dt.month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame["Date"].dt.month / 12.0)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["weekday"] = frame["Date"].dt.weekday

    feature_columns = [column for column in frame.columns if column not in {"Date", "price"}]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    return frame


def target_for_horizon(features: pd.DataFrame, horizon: int) -> pd.Series:
    return np.log(features["price"].shift(-horizon)) - np.log(features["price"])


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in {"Date", "price"}]


def _model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(os.getenv("COFFEE_FORECAST_TREES", "160")),
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=1,
        random_state=42,
    )


def _fit_predict(train_x: pd.DataFrame, train_y: pd.Series, predict_x: pd.DataFrame) -> float:
    if len(train_x) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough model training rows: {len(train_x)}")
    model = _model()
    model.fit(train_x, train_y)
    prediction = float(model.predict(predict_x)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("XGBoost returned a non-finite forecast")
    return prediction


def _price_metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "directional_accuracy": float(np.mean(np.sign(predicted - baseline) == np.sign(actual - baseline))),
        "baseline_mae": float(np.mean(np.abs(actual - baseline))),
        "baseline_rmse": float(np.sqrt(np.mean((actual - baseline) ** 2))),
    }


def compare_price_series(investing: pd.DataFrame, databento: pd.DataFrame) -> dict[str, Any]:
    """Compare overlapping OHLC series without assuming identical roll methodology."""
    left = investing.copy()
    right = databento.copy()
    left["Date"] = pd.to_datetime(left["Date"], format="mixed", errors="coerce").dt.date
    right["Date"] = pd.to_datetime(right["Date"], format="mixed", errors="coerce").dt.date
    merged = left.merge(right, on="Date", suffixes=("_investing", "_databento"))
    if merged.empty:
        return {"overlap_rows": 0, "compatible": None, "reason": "No overlapping dates"}

    differences: dict[str, dict[str, float]] = {}
    for field in ("price", "open", "high", "low"):
        left_name = f"{field}_investing"
        right_name = f"{field}_databento"
        if left_name not in merged or right_name not in merged:
            continue
        absolute = (pd.to_numeric(merged[left_name], errors="coerce") - pd.to_numeric(merged[right_name], errors="coerce")).abs().dropna()
        if absolute.empty:
            continue
        differences[field] = {
            "median_absolute_difference": float(absolute.median()),
            "max_absolute_difference": float(absolute.max()),
            "median_percent_difference": float((absolute / pd.to_numeric(merged[left_name], errors="coerce").abs()).replace([np.inf, -np.inf], np.nan).dropna().median() * 100),
        }
    price_stats = differences.get("price", {})
    compatible = bool(price_stats and price_stats["median_percent_difference"] <= 2.0 and price_stats["max_absolute_difference"] <= 25.0)
    return {"overlap_rows": len(merged), "first_overlap": str(merged["Date"].min()), "last_overlap": str(merged["Date"].max()), "differences": differences, "compatible": compatible}


def validate(features: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20, 31)) -> list[dict[str, Any]]:
    columns = feature_columns(features)
    split = int(len(features) * 0.8)
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        target = target_for_horizon(features, horizon)
        valid = features[columns].notna().all(axis=1) & target.notna()
        train_mask = valid & (np.arange(len(features)) < split)
        test_mask = valid & (np.arange(len(features)) >= split)
        if train_mask.sum() < MIN_TRAIN_ROWS or test_mask.sum() < 5:
            continue
        train_x = features.loc[train_mask, columns]
        train_y = target.loc[train_mask]
        test_x = features.loc[test_mask, columns]
        actual_price = features["price"].shift(-horizon).loc[test_mask].to_numpy(dtype=float)
        baseline_price = features.loc[test_mask, "price"].to_numpy(dtype=float)
        model = _model()
        model.fit(train_x, train_y)
        predicted_price = features.loc[test_mask, "price"].to_numpy(dtype=float) * np.exp(model.predict(test_x))
        future_price = actual_price
        usable = np.isfinite(predicted_price) & np.isfinite(future_price) & np.isfinite(baseline_price)
        if usable.sum() == 0:
            continue
        metrics = _price_metrics(future_price[usable], predicted_price[usable], baseline_price[usable])
        results.append({"horizon": horizon, **metrics, "model_beats_baseline_mae": metrics["mae"] < metrics["baseline_mae"]})
    return results


def next_weekdays(as_of: pd.Timestamp, count: int = 31) -> list[str]:
    dates: list[str] = []
    cursor = as_of.date()
    while len(dates) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() in BUSINESS_DAYS:
            dates.append(cursor.isoformat())
    return dates


def train_direct_forecast(rows: list[dict[str, Any]]) -> ForecastResult:
    history = normalize_history(rows)
    features = build_features(history)
    columns = feature_columns(features)
    latest = features.iloc[[-1]]
    future_dates = next_weekdays(history.iloc[-1]["Date"], 31)
    forecast: list[dict[str, Any]] = []
    current_price = float(history.iloc[-1]["price"])
    for horizon in FORECAST_HORIZONS:
        target = target_for_horizon(features, horizon)
        valid = features[columns].notna().all(axis=1) & target.notna()
        predicted_return = _fit_predict(features.loc[valid, columns], target.loc[valid], latest[columns])
        price = current_price * math.exp(predicted_return)
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid predicted price at horizon {horizon}")
        forecast.append({"date": future_dates[horizon - 1], "horizon": horizon, "price": round(price, 6)})

    return ForecastResult(
        as_of=history.iloc[-1]["Date"].date().isoformat(),
        current_price=round(current_price, 6),
        unit="US¢/lb",
        forecast=forecast,
        validation=validate(features),
        feature_count=len(columns),
        rows_used=len(history),
    )
