"""
fetch_logdata.py — Download historical daily OHLCV data from Databento and
write it to api/logdata/ in the same CSV format used by the existing pipeline.

IMPORTANT — spot data availability:
    Databento is an exchange market-data vendor. It does NOT provide OTC
    spot commodity prices (cash coffee, cash soybeans). The only way to
    obtain coffee or soybean price history via Databento is through their
    respective exchange-traded futures markets:

        Coffee  → Coffee "C" (KC) futures, ICE Futures U.S.
                  Databento dataset: IFUS.IMPACT
                  Continuous symbol: KC.c.0  (front-month, unadjusted)

        Soybeans → Soybean (ZS) futures, CME/CBOT via Globex
                   Databento dataset: GLBX.MDP3
                   Continuous symbol: ZS.c.0  (front-month, unadjusted)

    The script documents this clearly in the summary table and raises no
    errors — it simply switches to the futures proxy automatically.

Usage:
    # Full history (2007-08-24 to yesterday)
    python scripts/fetch_logdata.py

    # Custom range
    python scripts/fetch_logdata.py --start 2020-01-01 --end 2024-12-31

    # Dry-run (print plan, do not fetch)
    python scripts/fetch_logdata.py --dry-run

Dependencies:
    pip install databento pandas
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import databento as db


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
API_ROOT = SCRIPT_DIR.parent
LOGDATA_DIR = API_ROOT / "logdata"
LOGDATA_DIR.mkdir(parents=True, exist_ok=True)

# Output filenames must match what coffee_data_merged.py expects.
OUTPUT_FILES = {
    "coffee": LOGDATA_DIR / "CoffeeCData_log_returns.csv",
    "soybeans": LOGDATA_DIR / "US Soybeans Futures Historical Data_log_returns.csv",
}

# Earliest date that matches the existing logdata history.
DEFAULT_START = date(2007, 8, 24)


# ---------------------------------------------------------------------------
# Instrument catalogue
# ---------------------------------------------------------------------------
# Databento does not carry OTC spot prices. The entries below document the
# gap and specify the exchange-traded proxy used instead.

@dataclass
class Instrument:
    name: str
    spot_available: bool       # always False for these commodities on Databento
    spot_note: str
    dataset: str
    symbol: str
    stype_in: str
    schema: str
    output_key: str            # key into OUTPUT_FILES
    price_col_label: str       # label used in the Investing.com-style CSV header


INSTRUMENTS: list[Instrument] = [
    Instrument(
        name="Coffee",
        spot_available=False,
        spot_note="No OTC spot. Using ICE Coffee C front-month futures (KC.c.0).",
        dataset="IFUS.IMPACT",
        symbol="KC.c.0",
        stype_in="continuous",
        schema="ohlcv-1d",
        output_key="coffee",
        price_col_label="Price",
    ),
    Instrument(
        name="Soybeans",
        spot_available=False,
        spot_note="No OTC spot. Using CME/CBOT Soybean front-month futures (ZS.c.0).",
        dataset="GLBX.MDP3",
        symbol="ZS.c.0",
        stype_in="continuous",
        schema="ohlcv-1d",
        output_key="soybeans",
        price_col_label="Price",
    ),
]


# ---------------------------------------------------------------------------
# Databento helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "DATABENTO_API_KEY is not set. "
            "Export it before running:\n  export DATABENTO_API_KEY=your_key_here"
        )
    return key


def _licensed_end(exc: Exception) -> datetime | None:
    """
    Parse the cutoff timestamp from Databento errors like:
        'Try again with an end time before 2026-04-12T22:05:00.000000000Z'
    """
    match = re.search(
        r"before\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z",
        str(exc),
    )
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_databento_history(
    name: str,
    dataset: str,
    symbol: str,
    schema: str,
    start: date,
    end: date,
    stype_in: str = "raw_symbol",
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Databento and return a DataFrame.

    Columns returned: ts_event, open, high, low, close, volume
    The ts_event index is converted to a plain date column named 'Date'.

    Raises:
        EnvironmentError   — API key not set
        ValueError         — empty response or schema not available
        RuntimeError       — entitlement / symbol / network errors
    """
    if client is None:
        client = db.Historical(_api_key())

    start_str = start.isoformat()
    # Databento end is exclusive; add one day so the requested end date is included.
    end_exclusive = (end + timedelta(days=1)).isoformat()

    def _get(end_str: str) -> Any:
        return client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=schema,
            stype_in=stype_in,
            start=start_str,
            end=end_str,
        )

    print(f"  [{name}] fetching {schema} | {dataset} | {symbol} | {start_str} → {end.isoformat()}")

    store: Any
    try:
        store = _get(end_exclusive)
    except Exception as exc:
        cutoff = _licensed_end(exc)
        if cutoff is not None:
            retry_end = (cutoff - timedelta(seconds=1)).isoformat()
            print(f"  [{name}] license cutoff detected — retrying with end={retry_end[:10]}")
            try:
                store = _get(retry_end)
            except Exception as exc2:
                raise RuntimeError(
                    f"[{name}] request failed after license-cutoff retry: {exc2}"
                ) from exc2
        elif "entitlement" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            raise RuntimeError(
                f"[{name}] entitlement error — your key may not cover {dataset}: {exc}"
            ) from exc
        elif "symbol" in str(exc).lower() or "422" in str(exc):
            raise ValueError(
                f"[{name}] symbol resolution failed for '{symbol}' in {dataset}: {exc}"
            ) from exc
        else:
            raise RuntimeError(f"[{name}] unexpected error: {exc}") from exc

    df: pd.DataFrame = store.to_df()

    if df.empty:
        raise ValueError(
            f"[{name}] Databento returned an empty response for {symbol} "
            f"in {dataset} ({start_str} → {end.isoformat()})."
        )

    df = df.reset_index()

    # Normalise the event timestamp to a plain date.
    ts_col = next((c for c in ("ts_event", "ts_recv", "date") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"[{name}] no timestamp column found in response: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df[ts_col], utc=True).dt.normalize().dt.date

    # Keep one row per date (take the last if there are intraday duplicates).
    df = (
        df.sort_values(ts_col)
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Databento OHLCV prices are in fixed-point integer (1e-9 scaling for some
    # schemas) or already floats. Normalise to a float in the native unit.
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Detect fixed-point encoding: realistic futures prices are < 10_000.
            # Values above 1_000_000 are almost certainly raw integer ticks.
            if df[col].median() > 1_000_000:
                df[col] /= 1_000_000_000.0

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    return df[["Date", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_volume(v: int) -> str:
    """Format volume as Investing.com does: 1234567 → '1.23M', 12345 → '12.35K'."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K"
    return str(v)


def build_logdata_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw OHLCV DataFrame into the Investing.com-style CSV format
    expected by coffee_data_merged.py:

        Date, Price, Open, High, Low, Vol., Change %, log_price, log_return, pct_return
    """
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"])
    out = out.sort_values("Date").reset_index(drop=True)

    out["Price"] = df["close"].values
    out["Open"] = df["open"].values
    out["High"] = df["high"].values
    out["Low"] = df["low"].values
    out["Vol."] = df["volume"].apply(_format_volume).values

    pct_chg = out["Price"].pct_change() * 100
    out["Change %"] = pct_chg.map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")

    log_price = np.log(out["Price"].where(out["Price"] > 0))
    log_return = log_price.diff()
    pct_return = out["Price"].pct_change()

    out["log_price"] = log_price
    out["log_return"] = log_return
    out["pct_return"] = pct_return

    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    instrument: str
    spot_available: str
    dataset: str
    symbol: str
    proxy_used: bool
    rows_fetched: int
    output_path: str
    error: str | None = None


def _print_summary_table(results: list[FetchResult]) -> None:
    col_widths = [14, 15, 15, 12, 11, 12, 44]
    headers = [
        "Instrument", "Spot Available", "Dataset", "Symbol",
        "Proxy Used", "Rows", "Output File",
    ]

    def row_str(cells: list[str]) -> str:
        return "  " + "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    divider = "  " + "-" * (sum(col_widths) + 2 * len(col_widths))

    print("\n" + "=" * 72)
    print("  FETCH SUMMARY")
    print("=" * 72)
    print(row_str(headers))
    print(divider)

    for r in results:
        status = "ERROR" if r.error else str(r.rows_fetched)
        cells = [
            r.instrument,
            r.spot_available,
            r.dataset,
            r.symbol,
            "Yes" if r.proxy_used else "No",
            status,
            Path(r.output_path).name if not r.error else r.error[:42],
        ]
        print(row_str(cells))

    print("=" * 72)


def _print_final_summary(results: list[FetchResult]) -> None:
    print("\nFINAL SUMMARY")
    print("-" * 48)
    print("  Spot data availability:")
    for r in results:
        print(f"    {r.instrument:<12} spot = {r.spot_available}")
    print()
    proxied = [r for r in results if r.proxy_used]
    if proxied:
        print("  Futures proxies used:")
        for r in proxied:
            print(f"    {r.instrument:<12} → {r.symbol}  ({r.dataset})")
    failed = [r for r in results if r.error]
    if failed:
        print()
        print("  Failed instruments:")
        for r in failed:
            print(f"    {r.instrument:<12} {r.error}")
    print()


# ---------------------------------------------------------------------------
# Merge with existing logdata (append new rows, do not duplicate)
# ---------------------------------------------------------------------------

def _merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    If an existing CSV is present, keep all its rows and append any new dates
    from new_df that are not already covered. Returns the merged frame sorted
    by Date with duplicates removed (new_df takes precedence for overlapping dates).
    """
    if not path.exists():
        return new_df

    try:
        existing = pd.read_csv(path, dtype=str)
        existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        existing = existing.dropna(subset=["Date"])
    except Exception as exc:
        print(f"  Warning: could not read existing file ({exc}); will overwrite.")
        return new_df

    combined = pd.concat([existing, new_df], ignore_index=True)
    # new_df rows are at the end → keep="last" means new data wins on overlap.
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    start: date,
    end: date,
    dry_run: bool = False,
    instruments: list[Instrument] | None = None,
) -> list[FetchResult]:
    if instruments is None:
        instruments = INSTRUMENTS

    client = db.Historical(_api_key()) if not dry_run else None

    results: list[FetchResult] = []

    for inst in instruments:
        print(f"\n{'─' * 60}")
        print(f"  {inst.name.upper()}")
        print(f"  Spot available : No  — {inst.spot_note}")
        print(f"  Dataset        : {inst.dataset}")
        print(f"  Symbol         : {inst.symbol}  (stype={inst.stype_in})")
        print(f"  Schema         : {inst.schema}")
        print(f"{'─' * 60}")

        if dry_run:
            print("  [dry-run] skipping fetch.")
            results.append(FetchResult(
                instrument=inst.name,
                spot_available="No",
                dataset=inst.dataset,
                symbol=inst.symbol,
                proxy_used=True,
                rows_fetched=0,
                output_path=str(OUTPUT_FILES[inst.output_key]),
            ))
            continue

        try:
            raw_df = fetch_databento_history(
                name=inst.name,
                dataset=inst.dataset,
                symbol=inst.symbol,
                schema=inst.schema,
                start=start,
                end=end,
                stype_in=inst.stype_in,
                client=client,
            )
        except Exception as exc:
            traceback.print_exc()
            results.append(FetchResult(
                instrument=inst.name,
                spot_available="No",
                dataset=inst.dataset,
                symbol=inst.symbol,
                proxy_used=True,
                rows_fetched=0,
                output_path=str(OUTPUT_FILES[inst.output_key]),
                error=str(exc)[:80],
            ))
            continue

        csv_df = build_logdata_csv(raw_df)
        out_path = OUTPUT_FILES[inst.output_key]
        merged_df = _merge_with_existing(csv_df, out_path)
        merged_df.to_csv(out_path, index=False)

        new_rows = len(csv_df)
        total_rows = len(merged_df)
        print(f"  Fetched {new_rows} rows. Total after merge: {total_rows}.")
        print(f"  Saved → {out_path}")

        results.append(FetchResult(
            instrument=inst.name,
            spot_available="No",
            dataset=inst.dataset,
            symbol=inst.symbol,
            proxy_used=True,
            rows_fetched=new_rows,
            output_path=str(out_path),
        ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}'. Expected YYYY-MM-DD.")


def _yesterday() -> date:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical Coffee and Soybean OHLCV data from Databento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=DEFAULT_START,
        metavar="YYYY-MM-DD",
        help=f"Start date (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date inclusive (default: yesterday)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fetch plan without making any API calls.",
    )
    args = parser.parse_args()

    end = args.end or _yesterday()

    if args.start > end:
        parser.error(f"--start ({args.start}) must be before --end ({end}).")

    print(f"\nDatabento historical fetch")
    print(f"  Date range : {args.start} → {end}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Output dir : {LOGDATA_DIR}")

    results = run(start=args.start, end=end, dry_run=args.dry_run)

    _print_summary_table(results)
    _print_final_summary(results)

    failed = [r for r in results if r.error]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
