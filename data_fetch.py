#!/usr/bin/env python3
"""
Robust contract data fetcher for ICE Coffee C futures.

What this version fixes:
- Uses the official Databento Python client.
- Automatically retries Databento historical requests at the latest licensed cutoff
  when the requested range is not available under the current subscription.
- Avoids future dates and keeps end bounds exclusive.
- Uses Yahoo Finance's public quote endpoint as the main fallback for individual
  ICE coffee contract quotes.
- Keeps API Ninjas as a last-resort front-price fallback only.
- Logs actionable errors and still writes output files.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


# ── ENV ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(ROOT / ".env")
_load_dotenv(ROOT / ".env.local")

DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", "")
API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY", "")

WEB_PUBLIC_DATA = ROOT / "web" / "public" / "data"
DATA_DIR = ROOT / "data"
WEB_PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }
)


# ── CONTRACT CONFIG ─────────────────────────────────────────────────

KC_EXPIRY_MAP = {
    "KCK26": "2026-05-19",
    "KCN26": "2026-07-20",
    "KCU26": "2026-09-18",
    "KCZ26": "2026-12-18",
}

YAHOO_SYMBOL_MAP = {
    "KCK26": "KCK26.NYB",
    "KCN26": "KCN26.NYB",
    "KCU26": "KCU26.NYB",
    "KCZ26": "KCZ26.NYB",
}


def _active_symbols() -> list[str]:
    return list(KC_EXPIRY_MAP.keys())


# ── HELPERS ─────────────────────────────────────────────────────────

def _write_json(data: object, *paths: Path) -> None:
    text = json.dumps(data, indent=2, default=str)
    for p in paths:
        p.write_text(text, encoding="utf-8")


def _log_exception(prefix: str, exc: Exception) -> None:
    print(f"{prefix}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def _extract_license_cutoff(exc: Exception) -> datetime | None:
    """
    Parse Databento messages like:
    Try again with an end time before 2026-04-12T22:05:00.000000000Z.
    """
    text = str(exc)
    match = re.search(
        r"before\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z",
        text,
    )
    if not match:
        return None

    try:
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _normalize_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


# ── CONTRACT SOURCES ───────────────────────────────────────────────

def _databento_get_range(client: Any, symbols: list[str], end_dt: datetime) -> list[dict[str, Any]]:
    """Fetch recent daily bars ending strictly before end_dt."""
    end_dt = end_dt.astimezone(timezone.utc)
    start_dt = end_dt - timedelta(days=7)

    store = client.timeseries.get_range(
        dataset="IFUS.IMPACT",
        symbols=symbols,
        schema="ohlcv-1d",
        stype_in="raw_symbol",
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
    )

    df = store.to_df()
    if df.empty:
        return []

    df = df.reset_index()
    if "symbol" not in df.columns or "ts_event" not in df.columns:
        raise RuntimeError(
            f"Unexpected Databento dataframe columns: {sorted(df.columns.tolist())}"
        )

    latest = (
        df.sort_values("ts_event")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
    )

    now = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        open_px = float(row["open"])
        last = float(row["close"])
        change = last - open_px

        results.append(
            {
                "symbol": symbol,
                "expiry_date": KC_EXPIRY_MAP.get(symbol),
                "last_price": round(last, 2),
                "price_change": round(change, 2),
                "price_change_pct": round(change / open_px, 6) if open_px else 0,
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": 0,
                "captured_at": now.isoformat(),
                "source": "databento",
            }
        )

    order = {symbol: i for i, symbol in enumerate(symbols)}
    results.sort(key=lambda item: order.get(item["symbol"], 999))
    return results


# ✅ PRIMARY: Databento (licensed historical data)
def _fetch_databento() -> list[dict[str, Any]]:
    if not DATABENTO_API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    try:
        import databento as db
    except Exception as exc:
        raise RuntimeError(
            "The databento package is not installed. Run: pip install databento"
        ) from exc

    client = db.Historical(DATABENTO_API_KEY)
    symbols = _active_symbols()
    now = datetime.now(timezone.utc)

    # Historical data older than 24 hours is the intended range for this API.
    # Stay comfortably behind the edge to reduce license/range errors.
    preferred_end = now - timedelta(days=2)

    try:
        data = _databento_get_range(client, symbols, preferred_end)
        if data:
            return data
    except Exception as exc:
        cutoff = _extract_license_cutoff(exc)
        if cutoff is None:
            raise
        retry_end = cutoff - timedelta(seconds=1)
        print(
            f"[contracts] Databento licensed cutoff detected; retrying with end={retry_end.isoformat()}"
        )
        data = _databento_get_range(client, symbols, retry_end)
        if data:
            return data

    return []


# ⚠️ FALLBACK: Yahoo Finance via yfinance (handles crumb/cookie auth automatically)
def _fetch_yahoo_quotes() -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "The yfinance package is not installed. Run: pip install yfinance"
        ) from exc

    captured_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for base_symbol in _active_symbols():
        yahoo_symbol = YAHOO_SYMBOL_MAP.get(base_symbol)
        if not yahoo_symbol:
            continue

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.fast_info

            last_price = _normalize_price(getattr(info, "last_price", None))
            if last_price is None:
                # Fall back to history if fast_info is empty
                hist = ticker.history(period="2d")
                if hist.empty:
                    errors.append(f"{yahoo_symbol}: no price data")
                    continue
                last_price = _normalize_price(hist["Close"].iloc[-1])
                prev_close = _normalize_price(hist["Close"].iloc[-2]) if len(hist) > 1 else None
            else:
                prev_close = _normalize_price(getattr(info, "previous_close", None))

            if last_price is None:
                errors.append(f"{yahoo_symbol}: price is None after fallback")
                continue

            price_change = round(last_price - prev_close, 2) if prev_close else 0.0
            price_change_pct = round(price_change / prev_close, 6) if prev_close else 0.0
            volume = int(getattr(info, "three_month_average_volume", None) or 0)

            results.append(
                {
                    "symbol": base_symbol,
                    "expiry_date": KC_EXPIRY_MAP.get(base_symbol),
                    "last_price": last_price,
                    "price_change": price_change,
                    "price_change_pct": price_change_pct,
                    "volume": volume,
                    "open_interest": 0,
                    "captured_at": captured_at,
                    "source": "yahoo_finance",
                }
            )
        except Exception as exc:
            errors.append(f"{yahoo_symbol}: {exc}")

    if not results:
        raise RuntimeError(
            f"yfinance returned no usable rows. Errors: {errors}"
        )

    return results


# 🪶 LAST RESORT: API Ninjas (front price only)
def _fetch_api_ninjas() -> list[dict[str, Any]]:
    if not API_NINJAS_KEY:
        raise ValueError("Missing API_NINJAS_KEY")

    candidates = [
        ("coffee", "https://api.api-ninjas.com/v1/commodityprice"),
        ("Coffee C", "https://api.api-ninjas.com/v1/commodityprice"),
        ("Arabica Coffee", "https://api.api-ninjas.com/v1/commodityprice"),
    ]
    last_exc: Exception | None = None

    for name, url in candidates:
        try:
            resp = SESSION.get(
                url,
                params={"name": name},
                headers={"X-Api-Key": API_NINJAS_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price")
            if price is None:
                continue

            now = datetime.now(timezone.utc)
            return [
                {
                    "symbol": "KC_FRONT",
                    "expiry_date": None,
                    "last_price": float(price),
                    "price_change": 0,
                    "price_change_pct": 0,
                    "volume": 0,
                    "open_interest": 0,
                    "captured_at": now.isoformat(),
                    "source": "api_ninjas",
                }
            ]
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("API Ninjas returned no usable price")


# 🔄 MASTER FETCH
def fetch_contracts() -> list[dict[str, Any]]:
    try:
        data = _fetch_databento()
        if data:
            print("[contracts] Using Databento")
            return data
        print("[contracts] Databento returned no rows")
    except Exception as exc:
        _log_exception("[contracts] Databento failed", exc)

    try:
        data = _fetch_yahoo_quotes()
        if data:
            print("[contracts] Using Yahoo Finance fallback")
            return data
        print("[contracts] Yahoo returned no rows")
    except Exception as exc:
        _log_exception("[contracts] Yahoo fallback failed", exc)

    try:
        data = _fetch_api_ninjas()
        if data:
            print("[contracts] Using API Ninjas fallback")
            return data
        print("[contracts] API Ninjas returned no rows")
    except Exception as exc:
        _log_exception("[contracts] API Ninjas failed", exc)

    print("[contracts] No real contract data available — returning empty result")
    return []


# ── SNAPSHOT ───────────────────────────────────────────────────────

def compute_snapshot(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {}

    front = contracts[0]
    back = contracts[-1]
    front_price = float(front["last_price"])
    back_price = float(back["last_price"])

    return {
        "frontPrice": front_price,
        "curveShape": "Contango" if back_price > front_price else "Backwardation",
        "frontSymbol": front["symbol"],
        "asOf": front["captured_at"],
        "source": front.get("source", "unknown"),
    }


# ── MAIN ───────────────────────────────────────────────────────────

def main() -> None:
    print("Fetching contracts...")

    contracts = fetch_contracts()
    snapshot = compute_snapshot(contracts)

    _write_json(contracts, WEB_PUBLIC_DATA / "contracts.json")
    _write_json(snapshot, WEB_PUBLIC_DATA / "snapshot.json")

    print(f"Wrote {len(contracts)} contract rows")
    print("Done.")


if __name__ == "__main__":
    main()
