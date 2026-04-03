from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests


ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_CLIMATE = "https://climate-api.open-meteo.com/v1/climate"
_ALPHA_VANTAGE_MIN_INTERVAL_SECONDS = 10.0
_ALPHA_VANTAGE_LAST_CALL_AT = 0.0
# Polite delay between every Open-Meteo chunk request (seconds).
_OPEN_METEO_INTER_CHUNK_DELAY = 1.0
# Retry settings for transient HTTP errors (429 Too Many Requests, 5xx).
_REQUEST_MAX_RETRIES = 6
_REQUEST_BACKOFF_BASE = 2.0  # seconds; wait doubles each retry


@dataclass(frozen=True)
class CoffeeRegion:
    name: str
    latitude: float
    longitude: float
    weight: float


DEFAULT_REGIONS: List[CoffeeRegion] = [
    CoffeeRegion("Brazil_Sul_de_Minas", -21.2427, -45.0000, 0.50),
    CoffeeRegion("Vietnam_Dak_Lak", 12.7100, 108.2378, 0.30),
    CoffeeRegion("Colombia_Eje_Cafetero", 4.8143, -75.6946, 0.20),
]


class DataSourceError(RuntimeError):
    pass


def _load_local_env_file(path: Path | None = None) -> None:
    env_path = path or Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# -----------------------------
# Generic helpers
# -----------------------------
def _request_json(url: str, params: Dict, timeout: int = 60) -> Dict:
    """Fetch *url* with *params*, retrying on 429 / 5xx with exponential backoff.

    Alpha Vantage calls are additionally rate-limited to at most one request
    every ``_ALPHA_VANTAGE_MIN_INTERVAL_SECONDS`` seconds to stay within the
    free-tier quota.
    """
    global _ALPHA_VANTAGE_LAST_CALL_AT

    if url == ALPHA_VANTAGE_BASE:
        elapsed = time.monotonic() - _ALPHA_VANTAGE_LAST_CALL_AT
        if elapsed < _ALPHA_VANTAGE_MIN_INTERVAL_SECONDS:
            time.sleep(_ALPHA_VANTAGE_MIN_INTERVAL_SECONDS - elapsed)

    last_exc: Exception | None = None
    for attempt in range(_REQUEST_MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=timeout)

        if url == ALPHA_VANTAGE_BASE:
            _ALPHA_VANTAGE_LAST_CALL_AT = time.monotonic()

        # Retry on 429 (rate-limited) or any 5xx server error.
        if resp.status_code == 429 or resp.status_code >= 500:
            # Honour Retry-After header when the server provides it.
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else _REQUEST_BACKOFF_BASE * (2 ** attempt)
            print(
                f"[data_sources] HTTP {resp.status_code} from {url!r} "
                f"(attempt {attempt + 1}/{_REQUEST_MAX_RETRIES}). "
                f"Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)
            last_exc = requests.exceptions.HTTPError(response=resp)
            continue

        # All other 4xx errors are not retriable.
        resp.raise_for_status()

        try:
            payload = resp.json()
        except Exception as exc:  # pragma: no cover
            raise DataSourceError(f"Could not decode JSON from {url}: {exc}") from exc

        if isinstance(payload, dict) and payload.get("Error Message"):
            raise DataSourceError(payload["Error Message"])
        if isinstance(payload, dict) and payload.get("Note"):
            raise DataSourceError(payload["Note"])
        return payload

    raise DataSourceError(
        f"Request to {url!r} failed after {_REQUEST_MAX_RETRIES} attempts."
    ) from last_exc


def _normalize_month_end_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col).sort_index()
    out.index = out.index.to_period("M").to_timestamp(how="end").normalize()
    out = out.groupby(level=0).last()
    out.index.name = "Date"
    return out


def _iter_date_chunks(start_date: str, end_date: str, years_per_chunk: int = 5):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.DateOffset(years=years_per_chunk) - pd.Timedelta(days=1), end)
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        chunk_start = chunk_end + pd.Timedelta(days=1)


# -----------------------------
# Alpha Vantage
# -----------------------------
_load_local_env_file()


def get_alpha_vantage_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("AV_API_KEY")
    if not key:
        raise DataSourceError(
            "Set ALPHAVANTAGE_API_KEY (or AV_API_KEY) in your environment before running."
        )
    return key


def fetch_alpha_vantage_economic_series(function_name: str, interval: str = "monthly") -> pd.DataFrame:
    params = {
        "function": function_name,
        "interval": interval,
        "apikey": get_alpha_vantage_key(),
    }
    payload = _request_json(ALPHA_VANTAGE_BASE, params)
    records = payload.get("data")
    if not records:
        raise DataSourceError(f"No data returned for Alpha Vantage function={function_name}")
    df = pd.DataFrame(records)
    if "date" not in df.columns or "value" not in df.columns:
        raise DataSourceError(f"Unexpected Alpha Vantage schema for {function_name}: {df.columns.tolist()}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = _normalize_month_end_index(df[["date", "value"]])
    return df.rename(columns={"value": function_name.lower()})


def fetch_alpha_vantage_fx_monthly(from_symbol: str, to_symbol: str) -> pd.DataFrame:
    params = {
        "function": "FX_MONTHLY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": get_alpha_vantage_key(),
    }
    payload = _request_json(ALPHA_VANTAGE_BASE, params)
    series_key = next((k for k in payload.keys() if "Time Series FX" in k), None)
    if not series_key:
        raise DataSourceError(f"Unexpected FX payload for {from_symbol}/{to_symbol}: {list(payload.keys())}")
    ts = pd.DataFrame(payload[series_key]).T.reset_index().rename(columns={"index": "date"})
    ts.columns = [str(c).strip() for c in ts.columns]

    close_col = next((c for c in ts.columns if c.endswith("close")), None)
    if not close_col:
        raise DataSourceError(f"Could not find close column in FX payload: {ts.columns.tolist()}")

    ts[close_col] = pd.to_numeric(ts[close_col], errors="coerce")
    out = _normalize_month_end_index(ts[["date", close_col]])
    name = f"fx_{from_symbol.lower()}{to_symbol.lower()}"
    return out.rename(columns={close_col: name})


# -----------------------------
# Open-Meteo
# -----------------------------
def fetch_open_meteo_historical_monthly(region: CoffeeRegion, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date, years_per_chunk=1):
        params = {
            "latitude": region.latitude,
            "longitude": region.longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "timezone": "UTC",
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "shortwave_radiation_sum",
                    "et0_fao_evapotranspiration",
                    "soil_moisture_0_to_7cm_mean",
                ]
            ),
        }
        payload = _request_json(OPEN_METEO_ARCHIVE, params)
        time.sleep(_OPEN_METEO_INTER_CHUNK_DELAY)
        daily = payload.get("daily")
        if not daily or "time" not in daily:
            raise DataSourceError(f"Unexpected historical weather payload for {region.name}")
        frames.append(pd.DataFrame(daily))

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    monthly = pd.DataFrame(index=df.resample("ME").mean().index)
    monthly[f"{region.name}_temp_mean"] = df["temperature_2m_mean"].resample("ME").mean()
    monthly[f"{region.name}_precip_sum"] = df["precipitation_sum"].resample("ME").sum()
    monthly[f"{region.name}_radiation_sum"] = df["shortwave_radiation_sum"].resample("ME").sum()
    monthly[f"{region.name}_et0_sum"] = df["et0_fao_evapotranspiration"].resample("ME").sum()
    monthly[f"{region.name}_soil_moisture_mean"] = df["soil_moisture_0_to_7cm_mean"].resample("ME").mean()
    monthly.index.name = "Date"
    return monthly


def fetch_open_meteo_climate_monthly(
    region: CoffeeRegion,
    start_date: str,
    end_date: str,
    model: str = "MRI_AGCM3_2_S",
) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date, years_per_chunk=1):
        params = {
            "latitude": region.latitude,
            "longitude": region.longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "models": model,
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "soil_moisture_0_to_10cm_mean",
                ]
            ),
        }
        payload = _request_json(OPEN_METEO_CLIMATE, params)
        time.sleep(_OPEN_METEO_INTER_CHUNK_DELAY)
        daily = payload.get("daily")
        if not daily or "time" not in daily:
            raise DataSourceError(f"Unexpected climate payload for {region.name}")
        frames.append(pd.DataFrame(daily))

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    monthly = pd.DataFrame(index=df.resample("ME").mean().index)
    monthly[f"{region.name}_temp_mean"] = df["temperature_2m_mean"].resample("ME").mean()
    monthly[f"{region.name}_precip_sum"] = df["precipitation_sum"].resample("ME").sum()
    monthly[f"{region.name}_soil_moisture_mean"] = df["soil_moisture_0_to_10cm_mean"].resample("ME").mean()
    monthly.index.name = "Date"
    return monthly


def combine_weighted_regions(frames: Iterable[pd.DataFrame], regions: Iterable[CoffeeRegion]) -> pd.DataFrame:
    frames = list(frames)
    regions = list(regions)
    combined = pd.concat(frames, axis=1).sort_index()

    weighted = pd.DataFrame(index=combined.index)
    prefixes = [
        "temp_mean",
        "precip_sum",
        "radiation_sum",
        "et0_sum",
        "soil_moisture_mean",
    ]
    for suffix in prefixes:
        cols = [f"{r.name}_{suffix}" for r in regions if f"{r.name}_{suffix}" in combined.columns]
        if not cols:
            continue
        values = 0.0
        total_weight = 0.0
        for r in regions:
            c = f"{r.name}_{suffix}"
            if c in combined.columns:
                values = values + combined[c] * r.weight
                total_weight += r.weight
        weighted[f"weather_{suffix}"] = values / total_weight

    return weighted.sort_index()
