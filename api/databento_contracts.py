from __future__ import annotations

import base64
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

DATABENTO_URL = "https://hist.databento.com/v0/timeseries.get_range"
MONTH_CODES = {"H": "Mar", "K": "May", "N": "Jul", "U": "Sep", "Z": "Dec"}
STANDARD_MONTH_CODES = set(MONTH_CODES)
STAT_TYPES = {
    "open": 1,
    "settlement": 3,
    "low": 4,
    "high": 5,
    "cleared_volume": 6,
    "open_interest": 9,
    "close": 11,
    "net_change": 12,
}
_RAW_SYMBOL_RE = re.compile(r"^KC\s+FM([HKN UZ])(\d{4})!$".replace(" ", ""))
_CACHE: tuple[float, dict[str, Any]] | None = None
_CACHE_SECONDS = 600
ROLL_DAYS_BEFORE_EXPIRY = 10
UNDEFINED_TIMESTAMPS = {"18446744073709551615", "9223372036854775807"}


class DatabentoProviderError(RuntimeError):
    def __init__(self, reason: str, *, status: int | None = None, missing_fields: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.missing_fields = missing_fields or []


def _decode_databento_timestamp(value: Any) -> datetime | None:
    """Decode Databento Unix-nanosecond timestamps, rejecting sentinels."""
    text = str(value) if value is not None else ""
    if not text or text in UNDEFINED_TIMESTAMPS:
        return None
    try:
        number = int(text)
        if number <= 0 or number >= 10**20:
            return None
        return datetime.fromtimestamp(number / 1_000_000_000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _licensed_end() -> datetime:
    # IFUS historical access is delayed; stay before the normal cutoff.
    current = datetime.now(timezone.utc)
    return (current - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)


def _headers() -> dict[str, str]:
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        raise DatabentoProviderError("configuration_error")
    token = base64.b64encode(f"{key}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _request_jsonl(data: dict[str, str]) -> list[dict[str, Any]]:
    try:
        response = requests.post(DATABENTO_URL, headers=_headers(), data=data, timeout=30)
    except requests.Timeout as exc:
        raise DatabentoProviderError("timeout") from exc
    except requests.RequestException as exc:
        raise DatabentoProviderError("network_error") from exc
    if not response.ok:
        if response.status_code == 402:
            reason = "insufficient_budget"
        elif response.status_code in (401, 403):
            reason = "authentication_failure" if response.status_code == 401 else "entitlement_failure"
        elif response.status_code == 404:
            reason = "endpoint_not_found"
        elif response.status_code == 422:
            reason = "invalid_request_or_unavailable_range"
        else:
            reason = "databento_http_error"
        raise DatabentoProviderError(reason, status=response.status_code)
    try:
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]
    except (TypeError, ValueError) as exc:
        raise DatabentoProviderError("malformed_provider_response") from exc


def _nested(record: dict[str, Any], key: str) -> Any:
    return record.get(key, record.get("hd", {}).get(key))


def _int(value: Any) -> int | None:
    try:
        value = int(value)
        return value if value < 10**30 else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) >= 10**8:
        number /= 1_000_000_000
    return number if math.isfinite(number) else None


def _positive_price(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _definition_rows() -> list[dict[str, Any]]:
    end = _licensed_end()
    start = end - timedelta(days=10)
    records = _request_jsonl({
        "dataset": "IFUS.IMPACT",
        "symbols": "KC.FUT",
        "stype_in": "parent",
        "schema": "definition",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "encoding": "json",
    })
    by_symbol: dict[str, dict[str, Any]] = {}
    for record in records:
        raw = str(_nested(record, "raw_symbol") or "").strip()
        instrument_class = _nested(record, "instrument_class")
        security_type = _nested(record, "security_type")
        asset = _nested(record, "asset")
        expiration_ns = _nested(record, "expiration")
        match = _RAW_SYMBOL_RE.fullmatch(raw)
        if instrument_class != "F" or security_type != "FUT" or asset != "KC" or not match:
            continue
        try:
            expiration = datetime.fromtimestamp(int(expiration_ns) / 1_000_000_000, tz=timezone.utc).date()
        except (TypeError, ValueError, OSError):
            continue
        if expiration <= date.today():
            continue
        by_symbol[raw] = {
            "symbol": raw,
            "instrumentId": _nested(record, "instrument_id"),
            "monthCode": match.group(1),
            "year": 2000 + int(match.group(2)),
            "expirationDate": expiration.isoformat(),
        }
    return sorted(by_symbol.values(), key=lambda row: row["expirationDate"])


def _display_symbol(raw: str, year: int, month_code: str) -> str:
    return f"KC{month_code}{year % 100:02d}"


def _latest_statistics(definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    symbols = [row["symbol"] for row in definitions]
    instrument_symbols = {str(row["instrumentId"]): row["symbol"] for row in definitions}
    end = _licensed_end()
    start = end - timedelta(days=7)
    records = _request_jsonl({
        "dataset": "IFUS.IMPACT",
        "symbols": ",".join(symbols),
        "stype_in": "raw_symbol",
        "schema": "statistics",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "encoding": "json",
    })
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        raw = instrument_symbols.get(str(_nested(record, "instrument_id")), "")
        stat_type = _nested(record, "stat_type")
        if raw not in symbols or stat_type is None:
            continue
        try:
            stat_type = int(stat_type)
        except (TypeError, ValueError):
            continue
        key = next((name for name, value in STAT_TYPES.items() if value == stat_type), None)
        if key is None:
            continue
        event = _nested(record, "ts_event") or _nested(record, "ts_recv")
        event_text = str(event)
        existing = latest.get(f"{raw}:{key}")
        if existing is None or event_text > existing["event"]:
            latest[f"{raw}:{key}"] = {
                "value": record.get("price") if key not in {"cleared_volume", "open_interest"} else record.get("quantity"),
                "event": event_text,
                "reference": _nested(record, "ts_ref"),
            }
    grouped: dict[str, dict[str, Any]] = {}
    for compound, item in latest.items():
        raw, key = compound.rsplit(":", 1)
        grouped.setdefault(raw, {})[key] = item["value"]
        try:
            published = _decode_databento_timestamp(item["event"])
            if published is not None:
                grouped[raw]["asOf"] = published.isoformat()
                reference = item.get("reference")
                reference_date = _decode_databento_timestamp(reference)
                grouped[raw]["statDate"] = (reference_date or published).date().isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    return grouped


def _business_days_until(start: date, end: date) -> int:
    cursor = start
    count = 0
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            count += 1
    return count


def _select_forward_contract(definitions: list[dict[str, Any]], current_symbol: str | None = None) -> dict[str, Any]:
    if current_symbol:
        current = next((row for row in definitions if row["symbol"] == current_symbol), None)
        if current and _business_days_until(date.today(), date.fromisoformat(current["expirationDate"])) > ROLL_DAYS_BEFORE_EXPIRY:
            return current
        if current:
            later = [row for row in definitions if row["expirationDate"] > current["expirationDate"]]
            if later:
                return later[0]
    suitable = [row for row in definitions if _business_days_until(date.today(), date.fromisoformat(row["expirationDate"])) > ROLL_DAYS_BEFORE_EXPIRY]
    if not suitable:
        raise DatabentoProviderError("no_contract_found")
    return suitable[0]


def fetch_latest_daily_observation(current_symbol: str | None = None) -> dict[str, Any]:
    definitions = _definition_rows()
    selected = _select_forward_contract(definitions, current_symbol)
    stats = _latest_statistics([selected])
    values = stats.get(selected["symbol"], {})
    price = _positive_price(values.get("settlement"))
    open_price = _positive_price(values.get("open"))
    high = _positive_price(values.get("high"))
    low = _positive_price(values.get("low"))
    required_values = {"price": price, "open": open_price, "high": high, "low": low}
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        raw_values = {name: values.get("settlement" if name == "price" else name) for name in missing}
        if all(value in (None, "", "0", 0, "0.0") for value in raw_values.values()):
            raise DatabentoProviderError("invalid_ohlc", missing_fields=missing)
        raise DatabentoProviderError("missing_ohlc", missing_fields=missing)
    stat_date = values.get("statDate")
    try:
        parsed_date = date.fromisoformat(str(stat_date))
    except (TypeError, ValueError):
        raise DatabentoProviderError("invalid_market_date")
    today = datetime.now(timezone.utc).date()
    if parsed_date > today or (today - parsed_date).days > 14:
        raise DatabentoProviderError("invalid_market_date")
    return {
        "date": parsed_date.isoformat(),
        "price": price,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": _int(values.get("cleared_volume")),
        "changePercent": None,
        "source": "databento",
        "sourceContract": selected["symbol"],
        "sourceInstrumentId": str(selected["instrumentId"]),
        "sourceRetrievedAt": datetime.now(timezone.utc).isoformat(),
        "asOf": values.get("asOf"),
    }


def fetch_contracts() -> dict[str, Any]:
    global _CACHE
    now = datetime.now(timezone.utc).timestamp()
    if _CACHE and now - _CACHE[0] < _CACHE_SECONDS:
        return _CACHE[1]

    definitions = _definition_rows()[:4]
    if not definitions:
        raise RuntimeError("Databento returned no active outright Coffee C futures")
    stats = _latest_statistics(definitions)
    contracts = []
    for definition in definitions:
        raw = definition["symbol"]
        values = stats.get(raw, {})
        settlement = _positive_price(values.get("settlement"))
        close = _positive_price(values.get("close"))
        price = settlement or close
        volume = _int(values.get("cleared_volume"))
        open_interest = _int(values.get("open_interest"))
        if price is None:
            continue
        contracts.append({
            "symbol": raw,
            "displaySymbol": _display_symbol(raw, definition["year"], definition["monthCode"]),
            "label": f"{MONTH_CODES[definition['monthCode']]} {definition['year']}",
            "expiryDate": definition["expirationDate"],
            "lastPrice": price,
            "settlementPrice": settlement,
            "priceChange": _number(values.get("net_change")),
            "priceChangePct": None,
            "volume": volume,
            "openInterest": open_interest,
            "source": "databento",
            "asOf": values.get("asOf"),
        })
    if not contracts:
        raise RuntimeError("Databento returned no valid Coffee C settlement prices")
    payload = {
        "source": "databento",
        "asOf": max((contract["asOf"] for contract in contracts if contract["asOf"]), default=None),
        "contracts": contracts,
    }
    _CACHE = (now, payload)
    return payload
