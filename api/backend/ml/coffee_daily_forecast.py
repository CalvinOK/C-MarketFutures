from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

FORECAST_HORIZONS = tuple(range(1, 32))
MIN_TRAIN_ROWS = 400

@dataclass(frozen=True)
class ForecastResult:
    as_of: str
    current_price: float
    unit: str
    forecast: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    feature_count: int
    rows_used: int

def _date(value: Any) -> date | None:
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try: return datetime.strptime(text, fmt).date()
        except ValueError: pass
    return None

def normalize_history(rows: list[dict[str, Any]]) -> tuple[list[date], np.ndarray]:
    values: dict[date, float] = {}
    for row in rows:
        item_date = _date(row.get("market_date", row.get("Date", row.get("date"))))
        try: price = float(row.get("Price", row.get("price", row.get("close", row.get("Close")))))
        except (TypeError, ValueError): continue
        if item_date and math.isfinite(price) and price > 0: values[item_date] = price
    if len(values) < MIN_TRAIN_ROWS: raise ValueError(f"Not enough valid Coffee C history: {len(values)} rows; need at least {MIN_TRAIN_ROWS}")
    dates = sorted(values)
    return dates, np.asarray([values[item] for item in dates], dtype=float)

def _rolling(values: np.ndarray, window: int, minimum: int, std: bool = False) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(len(values)):
        sample = values[max(0, index - window + 1):index + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) >= minimum: result[index] = np.std(sample, ddof=1) if std and len(sample) > 1 else (np.mean(sample) if not std else 0.0)
    return result

def build_features(history: tuple[list[date], np.ndarray]) -> tuple[np.ndarray, list[str]]:
    dates, prices = history; log_price = np.log(prices); returns = np.full(len(prices), np.nan); returns[1:] = np.diff(log_price)
    columns = [returns]; names = ["log_return"]
    for lag in (1, 2, 3, 5, 10, 20):
        values = np.full(len(prices), np.nan); values[lag:] = returns[:-lag]; columns.append(values); names.append(f"return_lag_{lag}")
    for window in (5, 10, 20, 60):
        mean = _rolling(returns, window, max(3, window // 2)); std = _rolling(returns, window, max(3, window // 2), True); sma = _rolling(prices, window, max(3, window // 2)); momentum = log_price - np.roll(log_price, window); momentum[:window] = np.nan
        columns.extend([mean, std, momentum, sma, prices / sma - 1.0]); names.extend([f"return_mean_{window}", f"return_std_{window}", f"momentum_{window}", f"sma_{window}", f"price_vs_sma_{window}"])
    for window in (60, 252):
        mean = _rolling(prices, window, max(20, window // 2)); std = _rolling(prices, window, max(20, window // 2), True); high = np.full(len(prices), np.nan); low = np.full(len(prices), np.nan)
        for index in range(len(prices)):
            sample = prices[max(0, index - window + 1):index + 1]
            if len(sample) >= max(20, window // 2): high[index] = sample.max(); low[index] = sample.min()
        columns.extend([(prices - mean) / std, prices / high, prices / low]); names.extend([f"zscore_{window}", f"high_position_{window}", f"low_position_{window}"])
    day = np.asarray([item.timetuple().tm_yday for item in dates], dtype=float); month = np.asarray([item.month for item in dates], dtype=float)
    columns.extend([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25), np.asarray([item.weekday() for item in dates], dtype=float)])
    names.extend(["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos", "weekday"])
    matrix = np.column_stack(columns); matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names

def target_for_horizon(prices: np.ndarray, horizon: int) -> np.ndarray:
    target = np.full(len(prices), np.nan); target[:-horizon] = np.log(prices[horizon:]) - np.log(prices[:-horizon]); return target

class _RidgeRegressor:
    def __init__(self, regularization: float = 1.0) -> None:
        self.regularization = regularization
        self.weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "_RidgeRegressor":
        centered = np.nan_to_num(features, nan=0.0)
        design = np.column_stack([np.ones(len(centered)), centered])
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Forecast model has not been fitted")
        centered = np.nan_to_num(features, nan=0.0)
        design = np.column_stack([np.ones(len(centered)), centered])
        return design @ self.weights

def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray) -> float:
    if len(train_x) < MIN_TRAIN_ROWS: raise ValueError(f"Not enough model training rows: {len(train_x)}")
    model = _RidgeRegressor().fit(train_x, train_y); prediction = float(model.predict(predict_x)[0])
    if not math.isfinite(prediction): raise RuntimeError("Forecast model returned a non-finite prediction")
    return prediction

def _metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    return {"mae": float(np.mean(np.abs(actual - predicted))), "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))), "directional_accuracy": float(np.mean(np.sign(predicted - baseline) == np.sign(actual - baseline))), "baseline_mae": float(np.mean(np.abs(actual - baseline))), "baseline_rmse": float(np.sqrt(np.mean((actual - baseline) ** 2)))}

def validate(features: np.ndarray, prices: np.ndarray, horizons: tuple[int, ...] = (1, 5, 10, 20, 31)) -> list[dict[str, Any]]:
    split = int(len(prices) * 0.8); results = []
    for horizon in horizons:
        target = target_for_horizon(prices, horizon); valid = np.isfinite(features).all(axis=1) & np.isfinite(target); train = valid & (np.arange(len(prices)) < split); test = valid & (np.arange(len(prices)) >= split)
        if train.sum() < MIN_TRAIN_ROWS or test.sum() < 5: continue
        model = _RidgeRegressor().fit(features[train], target[train]); indexes = np.flatnonzero(test); baseline = prices[indexes]; actual = prices[indexes + horizon]; predicted = baseline * np.exp(model.predict(features[test])); metrics = _metrics(actual, predicted, baseline); results.append({"horizon": horizon, **metrics, "model_beats_baseline_mae": metrics["mae"] < metrics["baseline_mae"]})
    return results

def _future_dates(as_of: date, count: int = 31) -> list[str]:
    result = []; cursor = as_of
    while len(result) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5: result.append(cursor.isoformat())
    return result

def train_direct_forecast(rows: list[dict[str, Any]]) -> ForecastResult:
    dates, prices = normalize_history(rows); features, names = build_features((dates, prices)); future_dates = _future_dates(dates[-1]); forecast = []
    for horizon in FORECAST_HORIZONS:
        target = target_for_horizon(prices, horizon); valid = np.isfinite(features).all(axis=1) & np.isfinite(target); predicted_return = _fit_predict(features[valid], target[valid], features[-1:]); price = prices[-1] * math.exp(predicted_return)
        if not math.isfinite(price) or price <= 0: raise RuntimeError(f"Invalid predicted price at horizon {horizon}")
        forecast.append({"date": future_dates[horizon - 1], "horizon": horizon, "price": round(price, 6)})
    return ForecastResult(dates[-1].isoformat(), round(float(prices[-1]), 6), "US¢/lb", forecast, validate(features, prices), len(names), len(prices))
