from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
API_ROOT_DIR = BASE_DIR.parent
PROJECT_ROOT = API_ROOT_DIR.parent
WEBSITE_DATA_DIR = PROJECT_ROOT / "website" / "public" / "data"
LEGACY_DATA_DIR = PROJECT_ROOT / "old" / "api" / "data"
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "old" / "api" / "outputs"

CONTRACT_SYMBOLS = ["KCK26", "KCN26", "KCU26", "KCZ26"]
CONTRACT_EXPIRIES = {
    "KCK26": "2026-05-19",
    "KCN26": "2026-07-20",
    "KCU26": "2026-09-18",
    "KCZ26": "2026-12-18",
}

RUNTIME_CACHE: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _candidate_dirs() -> list[Path]:
    return [WEBSITE_DATA_DIR, LEGACY_DATA_DIR, LEGACY_OUTPUT_DIR]


def _first_existing_path(file_name: str, dirs: list[Path] | None = None) -> Path | None:
    for directory in dirs or _candidate_dirs():
        candidate = directory / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_text(file_name: str, dirs: list[Path] | None = None) -> str | None:
    path = _first_existing_path(file_name, dirs)
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def _read_json(file_name: str, dirs: list[Path] | None = None) -> Any | None:
    text = _read_text(file_name, dirs)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10:
        return text[:10]
    return text


def _last_friday(today: datetime | None = None) -> datetime:
    current = (today or datetime.now(UTC)).date()
    offset = (current.weekday() - 4) % 7
    return datetime.combine(current - timedelta(days=offset), datetime.min.time(), tzinfo=UTC)


def _with_runtime_cache(key: str, payload: Any | None = None) -> Any | None:
    if payload is not None:
        RUNTIME_CACHE[key] = {"generated_at": _now_iso(), "payload": payload}
        return payload
    entry = RUNTIME_CACHE.get(key)
    if isinstance(entry, dict):
        return entry.get("payload")
    return None


def _normalize_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["symbol"] = str(normalized.get("symbol", "")).strip()
    normalized["expiry_date"] = _to_iso_date(normalized.get("expiry_date"))
    normalized["last_price"] = _safe_float(normalized.get("last_price"))
    normalized["price_change"] = _safe_float(normalized.get("price_change"))
    normalized["price_change_pct"] = _safe_float(normalized.get("price_change_pct"))
    normalized["volume"] = _safe_int(normalized.get("volume"))
    normalized["open_interest"] = _safe_int(normalized.get("open_interest"))
    normalized["captured_at"] = normalized.get("captured_at") or _now_iso()
    return normalized


def _fallback_contract_rows() -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for symbol in CONTRACT_SYMBOLS:
        yahoo_symbol = f"{symbol}.NYB"
        try:
            ticker = yf.Ticker(yahoo_symbol)
            history = ticker.history(period="5d", auto_adjust=False)
            if history.empty or "Close" not in history:
                continue

            closes = [float(value) for value in history["Close"].dropna().tolist()]
            if not closes:
                continue

            last_price = closes[-1]
            previous_price = closes[-2] if len(closes) > 1 else last_price
            price_change = last_price - previous_price
            price_change_pct = (price_change / previous_price) if previous_price else 0.0
            volume = _safe_int(history["Volume"].dropna().iloc[-1] if "Volume" in history.columns and not history["Volume"].dropna().empty else 0)

            open_interest = 0
            try:
                info = getattr(ticker, "fast_info", None)
                if info is not None:
                    open_interest = _safe_int(getattr(info, "open_interest", None), 0)
                    if open_interest == 0 and isinstance(info, dict):
                        open_interest = _safe_int(info.get("open_interest") or info.get("openInterest"), 0)
            except Exception:
                open_interest = 0

            rows.append(
                _normalize_contract_row(
                    {
                        "symbol": symbol,
                        "expiry_date": CONTRACT_EXPIRIES.get(symbol),
                        "last_price": round(last_price, 2),
                        "price_change": round(price_change, 2),
                        "price_change_pct": round(price_change_pct, 6),
                        "volume": volume,
                        "open_interest": open_interest,
                        "captured_at": _now_iso(),
                        "source": "yfinance",
                    }
                )
            )
        except Exception:
            continue

    rows.sort(key=lambda row: CONTRACT_SYMBOLS.index(row["symbol"]) if row["symbol"] in CONTRACT_SYMBOLS else 999)
    return rows


def _compute_snapshot(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {}

    front = contracts[0]
    back = contracts[-1]
    front_price = _safe_float(front.get("last_price"))
    back_price = _safe_float(back.get("last_price"))

    return {
        "frontPrice": round(front_price, 2),
        "curveShape": "Contango" if back_price > front_price else "Backwardation",
        "frontSymbol": front.get("symbol"),
        "totalVolume": sum(_safe_int(row.get("volume")) for row in contracts),
        "totalOpenInterest": sum(_safe_int(row.get("open_interest")) for row in contracts),
        "asOf": front.get("captured_at"),
        "priceSource": front.get("source", "unknown"),
        "oiSource": "databento_statistics" if sum(_safe_int(row.get("open_interest")) for row in contracts) > 0 else None,
    }


def _load_contract_rows(refresh: bool = False) -> list[dict[str, Any]]:
    if refresh:
        rows = _fallback_contract_rows()
        if rows:
            _with_runtime_cache("contracts", rows)
            return rows

    if not refresh:
        cached = _with_runtime_cache("contracts")
        if isinstance(cached, list) and cached:
            return [dict(row) for row in cached]

    data = _read_json("contracts.json")
    if isinstance(data, list) and data:
        rows = [_normalize_contract_row(row) for row in data if isinstance(row, dict)]
        _with_runtime_cache("contracts", rows)
        return rows

    rows = _fallback_contract_rows()
    if rows:
        _with_runtime_cache("contracts", rows)
        return rows

    return []


def load_contracts_payload(refresh: bool = False) -> list[dict[str, Any]]:
    return _load_contract_rows(refresh=refresh)


def load_snapshot_payload(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _with_runtime_cache("snapshot")
        if isinstance(cached, dict) and cached:
            return dict(cached)

    data = _read_json("snapshot.json")
    if isinstance(data, dict) and data:
        _with_runtime_cache("snapshot", data)
        return data

    snapshot = _compute_snapshot(_load_contract_rows(refresh=refresh))
    if snapshot:
        _with_runtime_cache("snapshot", snapshot)
    return snapshot


def load_news_payload(limit: int = 3, refresh: bool = False) -> list[dict[str, Any]]:
    if not refresh:
        cached = _with_runtime_cache("news")
        if isinstance(cached, list) and cached:
            return [dict(row) for row in cached[:limit]]

    data = _read_json("news.json")
    if not isinstance(data, list):
        return []

    rows = [row for row in data if isinstance(row, dict)]
    _with_runtime_cache("news", rows)
    return rows[:limit]


def load_brief_payload(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _with_runtime_cache("brief")
        if isinstance(cached, dict) and cached:
            return dict(cached)

    data = _read_json("roaster_brief.json")
    if isinstance(data, dict) and data:
        _with_runtime_cache("brief", data)
        return data

    return {}


def load_market_report_payload(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _with_runtime_cache("market-report")
        if isinstance(cached, dict) and cached:
            return dict(cached)

    data = _read_json("latest_market_report.json")
    if isinstance(data, dict) and data:
        _with_runtime_cache("market-report", data)
        return data

    return {}


def _read_history_frame():
    pd = _import_pandas()
    text = _read_text("coffee_xgb_proj4_history.csv")
    if text is None:
        return None

    from io import StringIO

    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        return None
    return frame


def _import_pandas():
    import pandas as pd

    return pd


def _import_xgboost():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor
    except Exception:
        return None


def _build_weekly_frame(history_frame):
    pd = _import_pandas()
    frame = history_frame.copy()
    if "Date" not in frame.columns or "coffee_c" not in frame.columns:
        return pd.DataFrame()

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["coffee_c"] = pd.to_numeric(frame["coffee_c"], errors="coerce")
    frame = frame.dropna(subset=["Date", "coffee_c"]).sort_values("Date")
    if frame.empty:
        return pd.DataFrame()

    weekly = frame.set_index("Date")["coffee_c"].resample("W-FRI").last().dropna().to_frame(name="coffee_c")
    weekly = weekly.reset_index()
    return weekly


def _feature_frame_from_prices(price_frame):
    pd = _import_pandas()
    frame = price_frame.copy()
    prices = frame["coffee_c"].astype(float)

    feature_frame = pd.DataFrame(index=frame.index)
    for lag in (1, 2, 3, 4, 5, 8, 13):
        feature_frame[f"lag_{lag}"] = prices.shift(lag)

    for window in (4, 8, 13):
        feature_frame[f"rolling_mean_{window}"] = prices.shift(1).rolling(window).mean()
        feature_frame[f"rolling_std_{window}"] = prices.shift(1).rolling(window).std()

    feature_frame["weekly_return_1"] = prices.pct_change()
    feature_frame["weekly_return_4"] = prices.pct_change(4)
    feature_frame["month"] = pd.to_datetime(frame["Date"]).dt.month
    feature_frame["weekofyear"] = pd.to_datetime(frame["Date"]).dt.isocalendar().week.astype(int)
    return feature_frame


def _latest_feature_row(price_history: list[float], next_date: datetime) -> dict[str, float]:
    def lag(offset: int) -> float:
        if len(price_history) >= offset:
            return float(price_history[-offset])
        return float(price_history[0])

    def mean(window: int) -> float:
        values = price_history[-window:]
        return float(sum(values) / len(values)) if values else float(price_history[-1])

    def std(window: int) -> float:
        values = price_history[-window:]
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        return float(math.sqrt(max(variance, 0.0)))

    latest = float(price_history[-1])
    previous = float(price_history[-2]) if len(price_history) > 1 else latest
    return {
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_4": lag(4),
        "lag_5": lag(5),
        "lag_8": lag(8),
        "lag_13": lag(13),
        "rolling_mean_4": mean(4),
        "rolling_mean_8": mean(8),
        "rolling_mean_13": mean(13),
        "rolling_std_4": std(4),
        "rolling_std_8": std(8),
        "rolling_std_13": std(13),
        "weekly_return_1": (latest / previous - 1.0) if previous else 0.0,
        "weekly_return_4": (latest / lag(4) - 1.0) if lag(4) else 0.0,
        "month": next_date.month,
        "weekofyear": next_date.isocalendar().week,
    }


def _train_projection_model(weekly_frame):
    pd = _import_pandas()
    feature_frame = _feature_frame_from_prices(weekly_frame)
    modeling_frame = weekly_frame.join(feature_frame)
    modeling_frame["target"] = modeling_frame["coffee_c"].shift(-1)
    modeling_frame = modeling_frame.dropna().reset_index(drop=True)

    feature_columns = [column for column in modeling_frame.columns if column not in {"Date", "coffee_c", "target"}]
    if len(modeling_frame) < 24 or not feature_columns:
        return None, feature_columns, 0.0, modeling_frame

    model_cls = _import_xgboost()
    if model_cls is None:
        return None, feature_columns, 0.0, modeling_frame

    train_frame = modeling_frame.iloc[:-8].copy() if len(modeling_frame) > 30 else modeling_frame.iloc[:-4].copy()
    if train_frame.empty:
        train_frame = modeling_frame.copy()

    model = model_cls(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(train_frame[feature_columns], train_frame["target"].astype(float))

    predictions = model.predict(train_frame[feature_columns])
    residuals = train_frame["target"].astype(float).to_numpy() - predictions
    residual_std = float(pd.Series(residuals).std(ddof=1)) if len(residuals) > 1 else 0.0
    return model, feature_columns, residual_std, modeling_frame


def _recursive_projection(weekly_frame, horizon_weeks: int = 26):
    pd = _import_pandas()
    model, feature_columns, residual_std, modeling_frame = _train_projection_model(weekly_frame)

    history_prices = [float(value) for value in weekly_frame["coffee_c"].astype(float).tolist()]
    last_actual_date = pd.to_datetime(weekly_frame["Date"].iloc[-1]).to_pydatetime()
    price_history = history_prices[:]
    forecast_rows: list[dict[str, Any]] = []
    simple_mode = model is None or not feature_columns

    return_window = 4
    recent_returns: list[float] = []
    for index in range(1, len(price_history)):
        previous = price_history[index - 1]
        current = price_history[index]
        if previous > 0 and current > 0:
            recent_returns.append(math.log(current / previous))

    for step in range(1, horizon_weeks + 1):
        next_date = last_actual_date + timedelta(days=7 * step)
        feature_row = _latest_feature_row(price_history, next_date)

        if simple_mode:
            anchor_return = sum(recent_returns[-return_window:]) / len(recent_returns[-return_window:]) if recent_returns else 0.0
            predicted_price = price_history[-1] * math.exp(anchor_return)
        else:
            feature_frame = pd.DataFrame([feature_row], columns=feature_columns).fillna(0.0)
            predicted_price = float(model.predict(feature_frame)[0])
            if not math.isfinite(predicted_price) or predicted_price <= 0:
                predicted_price = price_history[-1]
            anchor_return = sum(recent_returns[-return_window:]) / len(recent_returns[-return_window:]) if recent_returns else 0.0

        previous_price = price_history[-1]
        predicted_log_return = math.log(predicted_price / previous_price) if predicted_price > 0 and previous_price > 0 else 0.0
        raw_1w_log_return = recent_returns[-1] if recent_returns else 0.0

        forecast_rows.append(
            {
                "as_of_date": last_actual_date.date().isoformat(),
                "step_week": step,
                "Date": next_date.date().isoformat(),
                "predicted_weekly_log_return": predicted_log_return,
                "projected_price": predicted_price,
                "anchor_weekly_log_return": anchor_return,
                "raw_1w_log_return": raw_1w_log_return,
            }
        )

        price_history.append(predicted_price)
        if previous_price > 0 and predicted_price > 0:
            recent_returns.append(math.log(predicted_price / previous_price))

    projection_rows = [
        {
            "date": row["Date"],
            "forecast": row["projected_price"],
            "lower_95": max(row["projected_price"] - 1.96 * residual_std, 0.0),
            "upper_95": row["projected_price"] + 1.96 * residual_std,
            "step": row["step_week"],
        }
        for row in forecast_rows
    ]

    return {
        "history_frame": weekly_frame,
        "forecast_rows": forecast_rows,
        "projection_rows": projection_rows,
        "as_of_date": last_actual_date.date().isoformat(),
    }


def _build_projection_payload(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _with_runtime_cache("projected-spot")
        if isinstance(cached, dict) and cached:
            return dict(cached)

    history_text = _read_text("coffee_xgb_proj4_history.csv")
    forecast_text = _read_text("coffee_xgb_proj4_rolling_path.csv")

    if history_text and forecast_text and not refresh:
        payload = {
            "format": "projected-spot-csv.v1",
            "files": {"history": "coffee_xgb_proj4_history.csv", "forecast": "coffee_xgb_proj4_rolling_path.csv"},
            "asOfDate": _extract_as_of_date(forecast_text),
            "historyCsv": history_text,
            "forecastCsv": forecast_text,
        }
        _with_runtime_cache("projected-spot", payload)
        return payload

    history_frame = _read_history_frame()
    if history_frame is None:
        return {
            "format": "projected-spot-csv.v1",
            "files": {"history": "coffee_xgb_proj4_history.csv", "forecast": "coffee_xgb_proj4_rolling_path.csv"},
            "asOfDate": None,
            "historyCsv": history_text or "",
            "forecastCsv": forecast_text or "",
        }

    pd = _import_pandas()
    weekly_frame = _build_weekly_frame(history_frame)
    if weekly_frame.empty:
        return {
            "format": "projected-spot-csv.v1",
            "files": {"history": "coffee_xgb_proj4_history.csv", "forecast": "coffee_xgb_proj4_rolling_path.csv"},
            "asOfDate": None,
            "historyCsv": history_frame.to_csv(index=False),
            "forecastCsv": "",
        }

    projection = _recursive_projection(weekly_frame, horizon_weeks=26)
    forecast_frame = pd.DataFrame(projection["forecast_rows"])
    projection_frame = pd.DataFrame(projection["projection_rows"])

    payload = {
        "format": "projected-spot-csv.v1",
        "files": {
            "history": "coffee_xgb_proj4_history.csv",
            "forecast": "coffee_xgb_proj4_rolling_path.csv",
            "projection": "coffee_spot_projection_6m.csv",
        },
        "asOfDate": projection["as_of_date"],
        "historyCsv": history_frame.to_csv(index=False),
        "forecastCsv": forecast_frame.to_csv(index=False),
        "projectionCsv": projection_frame.to_csv(index=False),
    }
    _with_runtime_cache("projected-spot", payload)
    return payload


def _extract_as_of_date(forecast_csv: str) -> str | None:
    lines = [line for line in forecast_csv.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header = [column.strip().lower() for column in lines[0].split(",")]
    row = [column.strip() for column in lines[1].split(",")]

    for candidate in ("as_of_date", "asofdate"):
        if candidate in header:
            index = header.index(candidate)
            if index < len(row):
                return row[index] or None
    return None


def load_projected_spot_payload(refresh: bool = False) -> dict[str, Any]:
    payload = _build_projection_payload(refresh=refresh)
    if payload.get("projectionCsv") is None and "forecastCsv" not in payload:
        return payload
    return payload


def build_root_payload() -> dict[str, Any]:
    return {
        "message": "Coffee market API is running.",
        "endpoints": {
            "health": "/health",
            "contracts": "/contracts",
            "snapshot": "/snapshot",
            "news": "/news",
            "brief": "/brief",
            "market_report": "/market-report",
            "projected_spot": "/projected-spot",
            "refresh": "/refresh",
        },
        "data_roots": {
            "website_public_data": str(WEBSITE_DATA_DIR),
            "legacy_data": str(LEGACY_DATA_DIR),
            "legacy_outputs": str(LEGACY_OUTPUT_DIR),
        },
    }


def build_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": _now_iso(),
        "available_files": {
            "contracts": _first_existing_path("contracts.json") is not None,
            "snapshot": _first_existing_path("snapshot.json") is not None,
            "news": _first_existing_path("news.json") is not None,
            "brief": _first_existing_path("roaster_brief.json") is not None,
            "market_report": _first_existing_path("latest_market_report.json") is not None,
            "projected_history": _first_existing_path("coffee_xgb_proj4_history.csv") is not None,
            "projected_forecast": _first_existing_path("coffee_xgb_proj4_rolling_path.csv") is not None,
        },
        "runtime_cache_keys": sorted(RUNTIME_CACHE.keys()),
        "has_internal_token": bool(os.environ.get("INTERNAL_API_TOKEN")),
    }


def build_market_refresh_payload() -> dict[str, Any]:
    contracts = _fallback_contract_rows()
    if not contracts:
        contracts = _load_contract_rows(refresh=True)
    else:
        _with_runtime_cache("contracts", contracts)

    snapshot = _compute_snapshot(contracts)
    if snapshot:
        _with_runtime_cache("snapshot", snapshot)

    projection = _build_projection_payload(refresh=True)
    _with_runtime_cache("projected-spot", projection)

    return {
        "refreshed_at": _now_iso(),
        "contracts": contracts,
        "snapshot": snapshot,
        "projected_spot": projection,
    }


def require_internal_token_if_configured(request) -> Any | None:
    expected_token = os.environ.get("INTERNAL_API_TOKEN")
    if not expected_token:
        return None

    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    if token == expected_token:
        return None

    from flask import jsonify

    response = jsonify({"error": "Unauthorized"})
    response.status_code = 401
    return response