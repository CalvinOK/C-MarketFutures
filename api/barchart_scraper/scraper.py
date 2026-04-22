#!/usr/bin/env python3
"""Barchart coffee futures scraper.

Target page:
https://www.barchart.com/futures/quotes/KC*0/futures-prices

Flow:
1) Bootstrap a requests session from the page HTML so Barchart sets its cookies.
2) Fetch the underlying quote JSON endpoint directly.
3) Fall back to HTML / Playwright parsing only if the JSON path fails.
4) Validate/transform rows into contracts payload.
5) Derive snapshot payload.
6) Atomically write contracts.json and snapshot.json.

This script keeps last-known-good files by only replacing files after a successful scrape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - import only required when fallback is used
    async_playwright = None

TARGET_URL = "https://www.barchart.com/futures/quotes/KC*0/futures-prices"
QUOTE_JSON_URL = "https://www.barchart.com/proxies/core-api/v1/quotes/get"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.25
DEFAULT_OUTPUT_DIR = (
    Path(os.getenv("RUNTIME_DATA_DIR", "/tmp/coffee-market-data"))
    if os.getenv("VERCEL")
    else (Path(__file__).resolve().parents[2] / "website" / "public" / "data")
)

MONTH_CODE_TO_MONTH = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

MONTH_NUM_TO_NAME = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

SYMBOL_RE = re.compile(r"\bKC([FGHJKMNQUVXZ])(\d{2})\b")


@dataclass
class RawRow:
    symbol: str
    last_price: str
    price_change: str
    price_change_pct: str
    volume: str
    open_interest: str
    price_as_of: str | None = None
    oi_date: str | None = None
    oi_published_at: str | None = None


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, default=str))


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("barchart_scraper")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def parse_compact_number(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text or text.upper() in {"N/A", "-", "--"}:
        return None

    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]

    text = text.strip()
    try:
        return int(round(float(text) * multiplier))
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text or text.upper() in {"N/A", "-", "--"}:
        return None
    text = text.replace("US¢/lb", "").replace("¢/lb", "").strip()

    # Handle unicode minus and optional leading plus sign.
    text = text.replace("\u2212", "-")

    # Parentheses notation: (1.25) -> -1.25
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    text = text.replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_pct(value: str | None) -> float | None:
    return parse_float(value)


def normalize_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    match = SYMBOL_RE.search(symbol.upper())
    if not match:
        return None
    return f"KC{match.group(1)}{match.group(2)}"


def symbol_to_expiry_date(symbol: str) -> str | None:
    match = SYMBOL_RE.fullmatch(symbol)
    if not match:
        return None
    month_code, yy = match.group(1), match.group(2)
    month = MONTH_CODE_TO_MONTH.get(month_code)
    if not month:
        return None
    year = 2000 + int(yy)
    return f"{year:04d}-{month:02d}-01"


def month_year_from_symbol(symbol: str) -> tuple[str, int] | tuple[None, None]:
    match = SYMBOL_RE.fullmatch(symbol)
    if not match:
        return None, None
    month = MONTH_CODE_TO_MONTH.get(match.group(1))
    if not month:
        return None, None
    return MONTH_NUM_TO_NAME[month], 2000 + int(match.group(2))


def value_from_header(cells: list[str], headers: list[str], *needles: str) -> str:
    for needle in needles:
        for idx, header in enumerate(headers):
            if needle in header and idx < len(cells):
                return cells[idx]
    return ""


def build_session_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def build_quote_headers(xsrf_token: str | None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": TARGET_URL,
        "Origin": "https://www.barchart.com",
    }
    if xsrf_token:
        headers["X-XSRF-TOKEN"] = xsrf_token
    return headers


def fetch_quote_json(url: str, timeout_seconds: int, retries: int, backoff_seconds: float, logger: logging.Logger) -> dict[str, Any] | None:
    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            page_response = session.get(url, headers=build_session_headers(), timeout=timeout_seconds)
            page_response.raise_for_status()

            xsrf_token = session.cookies.get("XSRF-TOKEN")
            if xsrf_token:
                xsrf_token = unquote(xsrf_token)

            quote_response = session.get(
                QUOTE_JSON_URL,
                params={
                    "fields": "symbol,contractSymbol,lastPrice,priceChange,openPrice,highPrice,lowPrice,previousPrice,volume,openInterest,tradeTime,symbolCode,symbolType,hasOptions",
                    "lists": "futures.contractInRoot",
                    "root": "KC",
                    "meta": "field.shortName,field.type,field.description,lists.lastUpdate",
                    "page": 1,
                    "limit": 100,
                    "hasOptions": "true",
                    "raw": 1,
                },
                headers=build_quote_headers(xsrf_token),
                timeout=timeout_seconds,
            )
            quote_response.raise_for_status()

            payload = quote_response.json()
            if isinstance(payload, dict):
                return payload
            return None
        except Exception as exc:
            log_event(
                logger,
                "quote_json_failed",
                attempt=attempt,
                retries=retries,
                error=str(exc),
            )
            if attempt == retries:
                return None
            sleep_for = backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.35)
            time.sleep(sleep_for)

    return None


def extract_rows_from_table_html(html: str) -> list[RawRow]:
    soup = BeautifulSoup(html, "html.parser")
    raw_rows: list[RawRow] = []

    for table in soup.find_all("table"):
        header_cells = table.select("thead tr th")
        headers = [h.get_text(" ", strip=True).lower() for h in header_cells]
        if not headers:
            continue

        joined = " ".join(headers)
        if "open" not in joined or "volume" not in joined:
            continue

        for tr in table.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if not cells:
                continue

            symbol = None
            for cell in cells:
                symbol = normalize_symbol(cell)
                if symbol:
                    break
            if not symbol:
                continue

            # Field mapping comments (from table columns):
            # - symbol: symbol/month column
            # - last_price: "Last" or "Close"
            # - price_change: "Change"
            # - price_change_pct: "% Change" / "Percent Change"
            # - volume: "Volume"
            # - open_interest: "Open Interest"
            raw_rows.append(
                RawRow(
                    symbol=symbol,
                    last_price=value_from_header(cells, headers, "last", "close", "settle"),
                    price_change=value_from_header(cells, headers, "change"),
                    price_change_pct=value_from_header(cells, headers, "% change", "percent change"),
                    volume=value_from_header(cells, headers, "volume"),
                    open_interest=value_from_header(cells, headers, "open interest"),
                )
            )

    return dedupe_rows(raw_rows)


def extract_rows_from_quote_json(payload: dict[str, Any]) -> list[RawRow]:
    raw_rows: list[RawRow] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue

        raw = item.get("raw") if isinstance(item.get("raw"), dict) else item
        symbol = normalize_symbol(str(raw.get("symbol") or item.get("symbol") or ""))
        if not symbol:
            continue

        trade_time = raw.get("tradeTime") or item.get("tradeTime")
        price_as_of = str(trade_time) if isinstance(trade_time, str) else None

        raw_rows.append(
            RawRow(
                symbol=symbol,
                last_price=str(raw.get("lastPrice") or item.get("lastPrice") or ""),
                price_change=str(raw.get("priceChange") or item.get("priceChange") or ""),
                price_change_pct=str(raw.get("priceChangePct") or item.get("priceChangePct") or ""),
                volume=str(raw.get("volume") or item.get("volume") or ""),
                open_interest=str(raw.get("openInterest") or item.get("openInterest") or ""),
                price_as_of=price_as_of,
            )
        )

    return dedupe_rows(raw_rows)


def dedupe_rows(rows: list[RawRow]) -> list[RawRow]:
    out: list[RawRow] = []
    seen: set[str] = set()
    for row in rows:
        if row.symbol in seen:
            continue
        seen.add(row.symbol)
        out.append(row)
    return out


async def extract_rows_with_playwright(url: str, timeout_ms: int) -> list[RawRow]:
    if async_playwright is None:
        raise RuntimeError("Playwright is not installed. Install with: pip install playwright")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(1500)

        data = await page.evaluate(
            r"""
            () => {
              const symbolRegex = /\bKC[FGHJKMNQUVXZ]\d{2}\b/;
              const tables = Array.from(document.querySelectorAll('table'));

              const target = tables.find((t) => {
                const text = (t.innerText || '').toLowerCase();
                return text.includes('open interest') && text.includes('volume') && text.includes('kc');
              });

              if (!target) return [];

              const headerCells = Array.from(target.querySelectorAll('thead th'));
              const headers = headerCells.map((h) => (h.textContent || '').trim().toLowerCase());

              const findIdx = (needles) => {
                for (const needle of needles) {
                  const idx = headers.findIndex((h) => h.includes(needle));
                  if (idx >= 0) return idx;
                }
                return -1;
              };

              const idxLast = findIdx(['last', 'close', 'settle']);
              const idxChange = findIdx(['change']);
              const idxPct = findIdx(['% change', 'percent change']);
              const idxVolume = findIdx(['volume']);
              const idxOI = findIdx(['open interest']);

              const rows = [];
              for (const tr of target.querySelectorAll('tbody tr')) {
                const cells = Array.from(tr.querySelectorAll('td')).map((td) => (td.textContent || '').trim());
                if (!cells.length) continue;

                let symbol = null;
                for (const cell of cells) {
                  const match = cell.match(symbolRegex);
                  if (match) {
                    symbol = match[0];
                    break;
                  }
                }
                if (!symbol) continue;

                const valueAt = (i) => (i >= 0 && i < cells.length ? cells[i] : '');
                rows.push({
                  symbol,
                  last_price: valueAt(idxLast),
                  price_change: valueAt(idxChange),
                  price_change_pct: valueAt(idxPct),
                  volume: valueAt(idxVolume),
                  open_interest: valueAt(idxOI),
                });
              }

              return rows;
            }
            """
        )

        await context.close()
        await browser.close()

    return dedupe_rows([RawRow(**row) for row in data])


def fetch_html_requests(
    url: str,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: float,
    logger: logging.Logger,
) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            log_event(
                logger,
                "request_failed",
                attempt=attempt,
                retries=retries,
                error=str(exc),
            )
            if attempt == retries:
                raise
            sleep_for = backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.35)
            time.sleep(sleep_for)

    raise RuntimeError("Unexpected retry loop exit")


def transform_rows_to_contracts(rows: list[RawRow], captured_at: str, logger: logging.Logger) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []

    for row in rows:
        symbol = normalize_symbol(row.symbol)
        if not symbol:
            continue

        expiry = symbol_to_expiry_date(symbol)
        month, year = month_year_from_symbol(symbol)

        last_price = parse_float(row.last_price)
        change = parse_float(row.price_change)
        change_pct = parse_pct(row.price_change_pct)
        volume = parse_compact_number(row.volume)
        open_interest = parse_compact_number(row.open_interest)

        # Ignore empty/N/A rows.
        if last_price is None:
            continue

        if change is None:
            change = 0.0
        if change_pct is None and last_price:
            change_pct = round((change / last_price) * 100.0, 4)
        elif change_pct is None:
            change_pct = 0.0

        if volume is None:
            volume = 0
        if open_interest is None:
            open_interest = 0

        if not expiry or month is None or year is None:
            log_event(logger, "skip_invalid_symbol", symbol=row.symbol)
            continue

        contract = {
            "symbol": symbol,
            "month": month,
            "year": year,
            "expiry_date": expiry,
            "last_price": round(last_price, 4),
            "price_change": round(change, 4),
            "price_change_pct": round(change_pct, 4),
            "volume": volume,
            "open_interest": open_interest,
            "captured_at": captured_at,
            "source": "barchart",
            "price_as_of": row.price_as_of or captured_at,
            "dataset": "coffee_futures",
            "oi_date": row.oi_date,
            "oi_published_at": row.oi_published_at,
            "oi_source": "barchart",
        }

        contracts.append(contract)

    contracts.sort(key=lambda c: c["expiry_date"])

    today = datetime.now(UTC).date().isoformat()
    active = [c for c in contracts if c["expiry_date"] >= today]
    return active if active else contracts


def validate_contracts(contracts: list[dict[str, Any]]) -> None:
    required = [
        "symbol",
        "expiry_date",
        "last_price",
        "price_change",
        "price_change_pct",
        "volume",
        "open_interest",
        "captured_at",
        "source",
    ]

    if not contracts:
        raise ValueError("No valid contracts extracted")

    for idx, contract in enumerate(contracts):
        for field in required:
            if field not in contract or contract[field] is None:
                raise ValueError(f"Contract at index {idx} missing required field: {field}")


def derive_curve_shape(contracts: list[dict[str, Any]]) -> str:
    if len(contracts) < 2:
        return "Flat"
    front = contracts[0]["last_price"]
    deferred = [c["last_price"] for c in contracts[1:] if c["last_price"] is not None]
    if not deferred:
        return "Flat"

    deferred_avg = sum(deferred) / len(deferred)
    if deferred_avg > front:
        return "Contango"
    if deferred_avg < front:
        return "Backwardation"
    return "Flat"


def derive_snapshot(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    front = contracts[0]
    return {
        "frontPrice": front["last_price"],
        "curveShape": derive_curve_shape(contracts),
        "frontSymbol": front["symbol"],
        "totalVolume": sum(int(c.get("volume", 0) or 0) for c in contracts),
        "totalOpenInterest": sum(int(c.get("open_interest", 0) or 0) for c in contracts),
        "asOf": front["captured_at"],
        "priceSource": "barchart",
        "oiSource": "barchart",
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp_file:
        tmp_file.write(json.dumps(payload, indent=2, ensure_ascii=True))
        tmp_file.write("\n")
        temp_name = tmp_file.name
    os.replace(temp_name, path)


def scrape_contracts(
    url: str,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: float,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    quote_payload = fetch_quote_json(
        url=url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
        logger=logger,
    )

    rows: list[RawRow] = []
    if quote_payload:
        rows = extract_rows_from_quote_json(quote_payload)
        log_event(logger, "quote_json_parser_result", rows=len(rows))

    if not rows:
        html = fetch_html_requests(
            url=url,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
            logger=logger,
        )

        rows = extract_rows_from_table_html(html)
        if rows:
            log_event(logger, "requests_parser_success", rows=len(rows))
        else:
            log_event(logger, "requests_parser_empty", message="No table rows found; trying Playwright")

        if not rows:
            rows = asyncio.run(extract_rows_with_playwright(url, timeout_ms=timeout_seconds * 1000))
            log_event(logger, "playwright_parser_result", rows=len(rows))

    captured_at = datetime.now(UTC).isoformat()
    contracts = transform_rows_to_contracts(rows, captured_at=captured_at, logger=logger)
    validate_contracts(contracts)

    return contracts, captured_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Barchart coffee futures and write JSON payloads")
    parser.add_argument("--url", default=TARGET_URL, help="Barchart futures page URL")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where contracts.json and snapshot.json are written",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.verbose)

    output_dir = Path(args.output_dir)
    contracts_path = output_dir / "contracts.json"
    snapshot_path = output_dir / "snapshot.json"

    try:
        log_event(logger, "scrape_start", url=args.url, output_dir=str(output_dir))

        contracts, _ = scrape_contracts(
            url=args.url,
            timeout_seconds=args.timeout,
            retries=args.retries,
            backoff_seconds=args.backoff,
            logger=logger,
        )
        snapshot = derive_snapshot(contracts)

        atomic_write_json(contracts_path, contracts)
        atomic_write_json(snapshot_path, snapshot)

        log_event(
            logger,
            "scrape_success",
            contracts_count=len(contracts),
            contracts_path=str(contracts_path),
            snapshot_path=str(snapshot_path),
        )
        return 0
    except Exception as exc:
        # Keep last-known-good output files by failing before any replace call when scrape fails.
        log_event(
            logger,
            "scrape_failed",
            error=str(exc),
            contracts_exists=contracts_path.exists(),
            snapshot_exists=snapshot_path.exists(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
