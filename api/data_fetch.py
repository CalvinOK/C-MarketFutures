#!/usr/bin/env python3
"""
ICE Coffee C contract + open interest fetcher.

What this version does:
- Fetches latest contract bars from Databento when available.
- Fetches open interest from Databento's statistics schema.
- Uses ts_ref for the business date of the statistic when available.
- Chooses the most recently published statistic per symbol/stat_type.
- Retries historical requests at the licensed cutoff if Databento rejects the end time.
- Falls back to Yahoo Finance for prices if Databento bars are unavailable.
- Never guesses open interest from OHLCV or unrelated schemas.
- Always writes contracts.json and snapshot.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


# ── PATHS / ENV ─────────────────────────────────────────────────────

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"

# On Vercel, website/public/data is read-only (/var/task). Write JSON outputs
# to RUNTIME_DATA_DIR (/tmp/...) so the Flask API can find them there first.
_runtime = os.environ.get("RUNTIME_DATA_DIR", "").strip()
WEB_PUBLIC_DATA = Path(_runtime) if _runtime else PROJECT_ROOT / "website" / "public" / "data"

WEB_PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
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
_load_dotenv(PROJECT_ROOT / ".env")
_load_dotenv(PROJECT_ROOT / ".env.local")

DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", "")
API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY", "")

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


# ── CONTRACT CONFIG ────────────────────────────────────────────────

KC_EXPIRY_MAP: dict[str, str] = {
    "KCK26": "2026-05-19",
    "KCN26": "2026-07-20",
    "KCU26": "2026-09-18",
    "KCZ26": "2026-12-18",
}

YAHOO_SYMBOL_MAP: dict[str, str] = {
    "KCK26": "KCK26.NYB",
    "KCN26": "KCN26.NYB",
    "KCU26": "KCU26.NYB",
    "KCZ26": "KCZ26.NYB",
}

DEFAULT_DATASET = "IFUS.IMPACT"


def _active_symbols() -> list[str]:
    return list(KC_EXPIRY_MAP.keys())


# ── GENERIC HELPERS ────────────────────────────────────────────────

def _write_json(data: object, *paths: Path) -> None:
    text = json.dumps(data, indent=2, default=str)
    for path in paths:
        path.write_text(text, encoding="utf-8")


def _log_exception(prefix: str, exc: Exception) -> None:
    print(f"{prefix}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def _extract_license_cutoff(exc: Exception) -> datetime | None:
    """
    Parse Databento historical errors like:
    'Try again with an end time before 2026-04-12T22:05:00.000000000Z'
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_date_like(value: Any) -> str | None:
    """
    Prefer YYYY-MM-DD for timestamps/dates.
    """
    if value is None:
        return None

    if pd is not None:
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            return ts.strftime("%Y-%m-%d")
        except Exception:
            pass

    try:
        if isinstance(value, datetime):
            return value.date().isoformat()
        return str(value)[:10]
    except Exception:
        return None


# ── DATABENTO HELPERS ──────────────────────────────────────────────

def _get_db_client() -> Any:
    if not DATABENTO_API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    try:
        import databento as db
    except Exception as exc:
        raise RuntimeError(
            "The databento package is not installed. Run: pip install databento"
        ) from exc

    return db.Historical(DATABENTO_API_KEY)


def _db_get_range_with_cutoff_retry(
    client: Any,
    *,
    dataset: str,
    symbols: list[str] | str,
    schema: str,
    stype_in: str,
    start: str,
    end: str,
) -> Any:
    """
    Retry once at the latest licensed cutoff if needed.
    """
    try:
        return client.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            stype_in=stype_in,
            start=start,
            end=end,
        )
    except Exception as exc:
        cutoff = _extract_license_cutoff(exc)
        if cutoff is None:
            raise
        retry_end = (cutoff - timedelta(seconds=1)).isoformat()
        print(f"[databento] licensed cutoff detected; retrying end={retry_end}")
        return client.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            stype_in=stype_in,
            start=start,
            end=retry_end,
        )


# ── PRICE FETCH: DATABENTO ─────────────────────────────────────────

def _fetch_databento_bars(
    symbols: list[str],
    *,
    dataset: str = DEFAULT_DATASET,
) -> list[dict[str, Any]]:
    """
    Fetch latest daily bars for the requested symbols.
    """
    client = _get_db_client()

    now = _utc_now()
    # Keep comfortably behind realtime / licensing edge.
    preferred_end = now - timedelta(days=2)
    start_dt = preferred_end - timedelta(days=7)

    store = _db_get_range_with_cutoff_retry(
        client,
        dataset=dataset,
        symbols=symbols,
        schema="ohlcv-1d",
        stype_in="raw_symbol",
        start=start_dt.isoformat(),
        end=preferred_end.isoformat(),
    )

    df = store.to_df()
    if df.empty:
        return []

    df = df.reset_index()

    required = {"symbol", "ts_event", "open", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Databento bars missing columns: {sorted(missing)}")

    latest = (
        df.sort_values("ts_event")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
    )

    captured_at = now.isoformat()
    rows: list[dict[str, Any]] = []

    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        open_px = _safe_float(row.get("open"))
        close_px = _safe_float(row.get("close"))
        volume = _safe_int(row.get("volume"), 0)
        change = close_px - open_px
        change_pct = (change / open_px) if open_px else 0.0

        rows.append(
            {
                "symbol": symbol,
                "expiry_date": KC_EXPIRY_MAP.get(symbol),
                "last_price": round(close_px, 2),
                "price_change": round(change, 2),
                "price_change_pct": round(change_pct, 6),
                "volume": volume,
                "open_interest": 0,
                "captured_at": captured_at,
                "price_as_of": _format_date_like(row.get("ts_event")),
                "source": "databento",
                "dataset": dataset,
            }
        )

    order = {sym: i for i, sym in enumerate(symbols)}
    rows.sort(key=lambda item: order.get(item["symbol"], 999))
    return rows


# ── PRICE FALLBACK: YAHOO ──────────────────────────────────────────

def _fetch_yahoo_quotes(symbols: list[str]) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "The yfinance package is not installed. Run: pip install yfinance"
        ) from exc

    captured_at = _utc_now().isoformat()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for symbol in symbols:
        yahoo_symbol = YAHOO_SYMBOL_MAP.get(symbol)
        if not yahoo_symbol:
            errors.append(f"{symbol}: no Yahoo mapping")
            continue

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.fast_info

            last_price = _normalize_price(getattr(info, "last_price", None))
            prev_close = _normalize_price(getattr(info, "previous_close", None))
            volume = _safe_int(getattr(info, "last_volume", None), 0)

            # Always fall back to history if price or volume is missing.
            if last_price is None or volume == 0:
                hist = ticker.history(period="5d")
                if hist.empty:
                    if last_price is None:
                        errors.append(f"{yahoo_symbol}: no price history")
                        continue
                else:
                    if last_price is None:
                        last_price = _normalize_price(hist["Close"].iloc[-1])
                        prev_close = (
                            _normalize_price(hist["Close"].iloc[-2])
                            if len(hist) > 1
                            else None
                        )
                    if volume == 0 and "Volume" in hist.columns:
                        volume = _safe_int(hist["Volume"].iloc[-1], 0)

            if last_price is None:
                errors.append(f"{yahoo_symbol}: price unavailable")
                continue

            change = round(last_price - prev_close, 2) if prev_close else 0.0
            change_pct = round(change / prev_close, 6) if prev_close else 0.0

            rows.append(
                {
                    "symbol": symbol,
                    "expiry_date": KC_EXPIRY_MAP.get(symbol),
                    "last_price": last_price,
                    "price_change": change,
                    "price_change_pct": change_pct,
                    "volume": volume,
                    "open_interest": 0,
                    "captured_at": captured_at,
                    "price_as_of": captured_at[:10],
                    "source": "yahoo_finance",
                    "dataset": None,
                }
            )
        except Exception as exc:
            errors.append(f"{yahoo_symbol}: {exc}")

    if not rows:
        raise RuntimeError(f"Yahoo returned no usable rows. Errors: {errors}")

    order = {sym: i for i, sym in enumerate(symbols)}
    rows.sort(key=lambda item: order.get(item["symbol"], 999))
    return rows


# ── LAST RESORT FRONT PRICE ────────────────────────────────────────

def _fetch_api_ninjas_front_price() -> list[dict[str, Any]]:
    if not API_NINJAS_KEY:
        raise ValueError("Missing API_NINJAS_KEY")

    candidates = [
        "coffee",
        "Coffee C",
        "Arabica Coffee",
    ]

    last_exc: Exception | None = None

    for name in candidates:
        try:
            resp = SESSION.get(
                "https://api.api-ninjas.com/v1/commodityprice",
                params={"name": name},
                headers={"X-Api-Key": API_NINJAS_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price")
            if price is None:
                continue

            captured_at = _utc_now().isoformat()
            return [
                {
                    "symbol": "KC_FRONT",
                    "expiry_date": None,
                    "last_price": round(float(price), 2),
                    "price_change": 0.0,
                    "price_change_pct": 0.0,
                    "volume": 0,
                    "open_interest": 0,
                    "captured_at": captured_at,
                    "price_as_of": captured_at[:10],
                    "source": "api_ninjas",
                    "dataset": None,
                }
            ]
        except Exception as exc:
            last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError("API Ninjas returned no usable price")


def fetch_contract_prices(
    symbols: list[str] | None = None,
    *,
    dataset: str = DEFAULT_DATASET,
) -> list[dict[str, Any]]:
    if symbols is None:
        symbols = _active_symbols()

    try:
        rows = _fetch_databento_bars(symbols, dataset=dataset)
        if rows:
            print("[prices] using Databento")
            return rows
        print("[prices] Databento returned no rows")
    except Exception as exc:
        _log_exception("[prices] Databento failed", exc)

    try:
        rows = _fetch_yahoo_quotes(symbols)
        if rows:
            print("[prices] using Yahoo fallback")
            return rows
        print("[prices] Yahoo returned no rows")
    except Exception as exc:
        _log_exception("[prices] Yahoo failed", exc)

    try:
        rows = _fetch_api_ninjas_front_price()
        if rows:
            print("[prices] using API Ninjas fallback")
            return rows
        print("[prices] API Ninjas returned no rows")
    except Exception as exc:
        _log_exception("[prices] API Ninjas failed", exc)

    print("[prices] no real price data available")
    return []


# ── OPEN INTEREST: DATABENTO STATISTICS ────────────────────────────

def _resolve_open_interest_stat_types() -> list[Any]:
    """
    Return a list of candidate stat_type values to try.

    Databento documents that futures OI is in the statistics schema and is
    distinguished by stat_type. Exact enum names can vary by client version,
    so we support both enum values and string fallbacks.
    """
    candidates: list[Any] = []

    try:
        import databento as db

        enum_obj = getattr(db, "StatType", None)
        if enum_obj is not None:
            for name in (
                "OPEN_INTEREST",
                "OPEN_INTEREST_CLOSE",
                "OPEN_INTEREST_EOD",
            ):
                if hasattr(enum_obj, name):
                    candidates.append(getattr(enum_obj, name))
    except Exception:
        pass

    # String fallbacks for client/API compatibility.
    for s in (
        "open_interest",
        "open_interest_close",
        "open_interest_eod",
    ):
        candidates.append(s)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[Any] = []
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _extract_latest_oi_from_statistics_df(
    df: "pd.DataFrame",
    requested_symbols: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Return latest open-interest record per symbol.

    We use:
    - the latest publication timestamp per symbol for the selected stat_type
    - ts_ref as the business/session date when present
    """
    errors: list[str] = []

    if df.empty:
        return {}, ["statistics dataframe is empty"]

    work = df.reset_index().copy()

    if "symbol" not in work.columns:
        return {}, ["statistics data missing 'symbol' column"]

    if "stat_type" not in work.columns:
        return {}, ["statistics data missing 'stat_type' column"]

    # Find the numeric/value column.
    value_col = None
    for candidate in ("value", "price", "quantity", "stat", "val"):
        if candidate in work.columns:
            value_col = candidate
            break

    if value_col is None:
        numeric_candidates = [
            c for c in work.columns
            if c not in {"symbol", "stat_type", "ts_event", "ts_recv", "ts_ref"}
        ]
        for c in numeric_candidates:
            try:
                if pd is not None:
                    pd.to_numeric(work[c], errors="raise")
                    value_col = c
                    break
            except Exception:
                continue

    if value_col is None:
        return {}, [f"statistics data has no obvious value column: {work.columns.tolist()}"]

    # Prefer ts_recv if present because Databento says filtering is based on the
    # primary/index timestamp, and latest publication matters for statistics.
    sort_col = "ts_recv" if "ts_recv" in work.columns else "ts_event" if "ts_event" in work.columns else None
    if sort_col is None:
        return {}, ["statistics data missing both ts_recv and ts_event"]

    # Normalize symbols as strings.
    work["symbol"] = work["symbol"].astype(str)

    # Keep only rows for requested symbols, if possible.
    wanted = set(requested_symbols)
    work = work[work["symbol"].isin(wanted)]
    if work.empty:
        return {}, ["statistics returned no rows for requested symbols"]

    # Sort by latest publication and keep last row per symbol.
    work = work.sort_values(sort_col)
    latest = work.groupby("symbol", as_index=False).tail(1)

    out: dict[str, dict[str, Any]] = {}

    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        oi_value = _safe_int(row.get(value_col), 0)
        stat_date = _format_date_like(row.get("ts_ref")) or _format_date_like(row.get(sort_col))

        out[symbol] = {
            "symbol": symbol,
            "open_interest": oi_value,
            "oi_date": stat_date,
            "published_at": str(row.get(sort_col)) if row.get(sort_col) is not None else None,
            "ts_ref": str(row.get("ts_ref")) if row.get("ts_ref") is not None else None,
            "stat_type": str(row.get("stat_type")),
            "value_column": value_col,
        }

    missing = [sym for sym in requested_symbols if sym not in out]
    for sym in missing:
        errors.append(f"{sym}: no OI statistic found")

    return out, errors


def fetch_open_interest(
    symbols: list[str] | None = None,
    *,
    dataset: str = DEFAULT_DATASET,
    days_back: int = 14,
    export_csv: bool = False,
) -> dict[str, Any]:
    """
    Fetch latest open interest for each symbol from Databento statistics schema.

    Returns:
        {
          "rows": [...],
          "latest": {"KCK26": 12345, ...},
          "details": {"KCK26": {...}, ...},
          "errors": [...],
          "fetched_at": "...",
          "dataset": "...",
        }
    """
    if pd is None:
        raise RuntimeError("pandas is required for open interest processing")

    if symbols is None:
        symbols = _active_symbols()

    client = _get_db_client()
    now = _utc_now()

    # Statistics may publish after the session date. Give enough lookback.
    end_dt = now - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=max(days_back, 3))

    stat_type_candidates = _resolve_open_interest_stat_types()
    all_errors: list[str] = []
    details: dict[str, dict[str, Any]] = {}

    for stat_type in stat_type_candidates:
        for stype_in in ("raw_symbol", "smart"):
            try:
                store = _db_get_range_with_cutoff_retry(
                    client,
                    dataset=dataset,
                    symbols=symbols,
                    schema="statistics",
                    stype_in=stype_in,
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                )
                df = store.to_df()
                if df.empty:
                    all_errors.append(f"{stat_type}/{stype_in}: statistics returned no rows")
                    break  # empty means this stat_type has no data; skip to next

                work = df.reset_index().copy()
                if "stat_type" not in work.columns:
                    all_errors.append(f"{stat_type}/{stype_in}: statistics missing stat_type column")
                    break

                # Match enum or string by normalized string representation.
                normalized_target = str(stat_type).lower()
                mask = work["stat_type"].astype(str).str.lower() == normalized_target
                filtered = work[mask]

                # Some client versions stringify enums differently. Fall back to contains check.
                if filtered.empty:
                    filtered = work[
                        work["stat_type"].astype(str).str.lower().str.contains("open_interest", regex=False)
                    ]

                if filtered.empty:
                    all_errors.append(f"{stat_type}/{stype_in}: no matching OI rows")
                    break

                extracted, errors = _extract_latest_oi_from_statistics_df(filtered, symbols)
                all_errors.extend(errors)

                if extracted:
                    details = extracted
                    break  # success — stop trying stype variants

            except Exception as exc:
                err_str = str(exc)
                all_errors.append(f"{stat_type}/{stype_in}: {type(exc).__name__}: {exc}")
                # Only retry with "smart" if this looks like a symbology resolution failure.
                if stype_in == "raw_symbol" and (
                    "symbology" in err_str.lower() or "422" in err_str
                ):
                    continue  # try stype_in="smart"
                break  # non-symbology error or already tried smart; move to next stat_type

        if details:
            break

    # Yahoo Finance fallback when Databento returns nothing.
    if not details:
        try:
            yahoo_oi = _fetch_yahoo_open_interest(symbols)
            if yahoo_oi:
                print("[oi] using Yahoo Finance fallback for open interest")
                fetched_at_str = now.isoformat()
                for sym, oi_val in yahoo_oi.items():
                    details[sym] = {
                        "symbol": sym,
                        "open_interest": oi_val,
                        "oi_date": fetched_at_str[:10],
                        "published_at": fetched_at_str,
                        "ts_ref": None,
                        "stat_type": "yahoo_finance",
                        "value_column": "openInterest",
                    }
        except Exception as exc:
            all_errors.append(f"yahoo_oi fallback: {type(exc).__name__}: {exc}")

    latest = {sym: details[sym]["open_interest"] for sym in details}
    rows = [details[sym] for sym in symbols if sym in details]

    if export_csv and rows and pd is not None:
        try:
            csv_path = DATA_DIR / f"open_interest_{now.strftime('%Y%m%d_%H%M%S')}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception as exc:
            all_errors.append(f"csv export failed: {exc}")

    return {
        "rows": rows,
        "latest": latest,
        "details": details,
        "errors": all_errors,
        "fetched_at": now.isoformat(),
        "dataset": dataset,
        "days_back": days_back,
        "stat_type_candidates": [str(x) for x in stat_type_candidates],
    }


# ── OPEN INTEREST: YAHOO FALLBACK ─────────────────────────────────

def _fetch_yahoo_open_interest(symbols: list[str]) -> dict[str, int]:
    """
    Fetch open interest from Yahoo Finance as a fallback.
    Returns a dict of {symbol: open_interest}.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "The yfinance package is not installed. Run: pip install yfinance"
        ) from exc

    result: dict[str, int] = {}
    errors: list[str] = []

    for symbol in symbols:
        yahoo_symbol = YAHOO_SYMBOL_MAP.get(symbol)
        if not yahoo_symbol:
            errors.append(f"{symbol}: no Yahoo mapping")
            continue
        try:
            ticker = yf.Ticker(yahoo_symbol)
            oi = ticker.info.get("openInterest")
            if oi is not None:
                result[symbol] = _safe_int(oi, 0)
            else:
                errors.append(f"{yahoo_symbol}: openInterest not in info")
        except Exception as exc:
            errors.append(f"{yahoo_symbol}: {exc}")

    if errors:
        print(f"[oi/yahoo] notes: {'; '.join(errors)}")

    return result


# ── MERGE / SNAPSHOT ───────────────────────────────────────────────

def enrich_contracts_with_open_interest(
    contracts: list[dict[str, Any]],
    oi_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    details = oi_payload.get("details", {})

    for contract in contracts:
        symbol = contract.get("symbol")
        if symbol in details:
            contract["open_interest"] = _safe_int(details[symbol].get("open_interest"), 0)
            contract["oi_date"] = details[symbol].get("oi_date")
            contract["oi_published_at"] = details[symbol].get("published_at")
            contract["oi_source"] = "databento_statistics"

    return contracts


def compute_snapshot(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {}

    front = contracts[0]
    back = contracts[-1]

    front_price = _safe_float(front.get("last_price"), 0.0)
    back_price = _safe_float(back.get("last_price"), 0.0)
    total_volume = sum(_safe_int(c.get("volume"), 0) for c in contracts)
    total_open_interest = sum(_safe_int(c.get("open_interest"), 0) for c in contracts)

    return {
        "frontPrice": round(front_price, 2),
        "curveShape": "Contango" if back_price > front_price else "Backwardation",
        "frontSymbol": front.get("symbol"),
        "totalVolume": total_volume,
        "totalOpenInterest": total_open_interest,
        "asOf": front.get("captured_at"),
        "priceSource": front.get("source", "unknown"),
        "oiSource": "databento_statistics" if total_open_interest > 0 else None,
    }


# ── OPTIONAL DEBUG ─────────────────────────────────────────────────

def diagnose_statistics_schema(
    symbols: list[str] | None = None,
    *,
    dataset: str = DEFAULT_DATASET,
    days_back: int = 14,
) -> None:
    """
    Print the columns and sample rows from the statistics schema so you can see
    what your account/dataset actually returns.
    """
    if pd is None:
        raise RuntimeError("pandas is required for diagnostics")

    if symbols is None:
        symbols = _active_symbols()[:1]

    client = _get_db_client()
    now = _utc_now()
    end_dt = now - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=max(days_back, 3))

    store = _db_get_range_with_cutoff_retry(
        client,
        dataset=dataset,
        symbols=symbols,
        schema="statistics",
        stype_in="raw_symbol",
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
    )

    df = store.to_df()
    if df.empty:
        print("No statistics data returned.")
        return

    work = df.reset_index()
    print("\n[diagnose] columns:")
    print(work.columns.tolist())

    print("\n[diagnose] unique stat_type values:")
    try:
        print(sorted(work["stat_type"].astype(str).unique().tolist()))
    except Exception:
        print("Could not enumerate stat_type values")

    print("\n[diagnose] last 10 rows:")
    print(work.tail(10).to_string())


# ── MAIN ───────────────────────────────────────────────────────────

def main() -> None:
    symbols = _active_symbols()

    print("Fetching contract prices...")
    contracts = fetch_contract_prices(symbols=symbols, dataset=DEFAULT_DATASET)

    print("Fetching open interest...")
    try:
        oi_payload = fetch_open_interest(
            symbols=symbols,
            dataset=DEFAULT_DATASET,
            days_back=14,
            export_csv=False,
        )

        if oi_payload.get("errors"):
            print("[oi] warnings:")
            for msg in oi_payload["errors"]:
                print(f"  - {msg}")

        contracts = enrich_contracts_with_open_interest(contracts, oi_payload)

        found = len(oi_payload.get("latest", {}))
        print(f"[oi] enriched {found} contract(s) with OI")

    except Exception as exc:
        _log_exception("[oi] fetch failed; continuing without OI", exc)

    snapshot = compute_snapshot(contracts)

    _write_json(contracts, WEB_PUBLIC_DATA / "contracts.json")
    _write_json(snapshot, WEB_PUBLIC_DATA / "snapshot.json")

    print(f"Wrote {len(contracts)} contract row(s)")
    print("Done.")


if __name__ == "__main__":
    if "--diagnose-statistics" in sys.argv:
        diagnose_statistics_schema()
    else:
        main()