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
    "settlement": 3,
    "cleared_volume": 6,
    "open_interest": 9,
    "close": 11,
    "net_change": 12,
}
_RAW_SYMBOL_RE = re.compile(r"^KC\s+FM([HKN UZ])(\d{4})!$".replace(" ", ""))
_CACHE: tuple[float, dict[str, Any]] | None = None
_CACHE_SECONDS = 600


def _licensed_end() -> datetime:
    # IFUS historical access is delayed; stay before the normal cutoff.
    current = datetime.now(timezone.utc)
    return (current - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)


def _headers() -> dict[str, str]:
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is not configured")
    token = base64.b64encode(f"{key}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _request_jsonl(data: dict[str, str]) -> list[dict[str, Any]]:
    response = requests.post(DATABENTO_URL, headers=_headers(), data=data, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Databento HTTP {response.status_code}: {response.text[:240]}")
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


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
            }
    grouped: dict[str, dict[str, Any]] = {}
    for compound, item in latest.items():
        raw, key = compound.rsplit(":", 1)
        grouped.setdefault(raw, {})[key] = item["value"]
        try:
            timestamp = int(item["event"])
            if timestamp > 10**15:
                grouped[raw]["asOf"] = datetime.fromtimestamp(timestamp / 1_000_000_000, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return grouped


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
