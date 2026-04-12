
import os
import io
import re
import json
import time
import zipfile
import urllib.parse
import urllib.request
import socket
from urllib.error import URLError, HTTPError
import numpy as np
import pandas as pd
import databento as db
from pathlib import Path


def _load_local_env_file() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("export "):
                value = value[len("export "):].strip()
            if value and value[0] in {'\"', "'"} and value[-1:] == value[0]:
                value = value[1:-1]

            os.environ.setdefault(key, value)


_load_local_env_file()

# =========================
# Settings
# =========================
API_KEY = os.getenv("DATABENTO_API_KEY")
DATASET = "IFUS.IMPACT"
PARENT = "KC.FUT"

# Databento KC coverage
KC_START = "2018-12-23"
KC_END = None  # None = present
DATABENTO_KC_START = "2023-03-20"  # cheaper reliable IFUS.IMPACT coverage
CONTINUOUS_DEPTH = 6

# Cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
USE_CACHE = True
FORCE_REFRESH = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def out_path(filename: str) -> str:
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)

HTTP_TIMEOUT_SECONDS = 30
HTTP_MAX_RETRIES = 3
HTTP_RETRY_SLEEP_SECONDS = 2

print(f"USE_CACHE={USE_CACHE}, FORCE_REFRESH={FORCE_REFRESH}", flush=True)

# FX
FX_START = "2005-01-01"
FX_SERIES = "DEXBZUS"  # BRL per USD, FRED

# COT
COT_START_YEAR = 2005
COT_END_YEAR = pd.Timestamp.today().year

# Weather (historicals)
# Assumption: one coffee-relevant point in Minas Gerais, Brazil.
# Override these if you want a different location or multiple regions.
WEATHER_LAT = -18.95
WEATHER_LON = -46.99
WEATHER_NAME = "minas_gerais_coffee_belt"

WEATHER_HIST_START = "2005-01-01"
WEATHER_HIST_END = None  # present

# Weather forecast archive
# Upgraded to the Open-Meteo Historical Forecast API for deeper archived coverage.
# Historical forecasts are available from 2022 onward.
WEATHER_FORECAST_START = "2022-01-01"
WEATHER_FORECAST_END = None  # present
WEATHER_FORECAST_MAX_HORIZON_DAYS = 14
WEATHER_FORECAST_MODEL = "best_match"

# Outputs
OUT_STRIP = out_path("kc_futures_daily_strip.csv")
OUT_CONT = out_path("kc_continuous_daily.csv")
OUT_WEEKLY = out_path("kc_continuous_weekly_friday.csv")
OUT_CURVE_DAILY = out_path("kc_curve_daily.csv")
OUT_CURVE_WEEKLY = out_path("kc_curve_weekly_friday.csv")

OUT_FX_DAILY = out_path("brl_usd_daily.csv")
OUT_FX_WEEKLY = out_path("brl_usd_weekly_friday.csv")

OUT_COT_WEEKLY = out_path("coffee_cot_weekly.csv")

OUT_WEATHER_DAILY = out_path(f"{WEATHER_NAME}_weather_daily.csv")
OUT_WEATHER_WEEKLY = out_path(f"{WEATHER_NAME}_weather_weekly_friday.csv")
OUT_WEATHER_DAILY_ENRICHED = out_path(f"{WEATHER_NAME}_weather_daily_enriched.csv")
OUT_WEATHER_WEEKLY_ENRICHED = out_path(f"{WEATHER_NAME}_weather_weekly_enriched_friday.csv")
OUT_WEATHER_FORECAST_FRIDAY = out_path(f"{WEATHER_NAME}_weather_forecast_friday_long.csv")

OUT_WEEKLY_JOINED = out_path("kc_weekly_with_fx.csv")
OUT_WEEKLY_JOINED_OVERLAP = out_path("kc_weekly_with_fx_overlap_only.csv")

OUT_WEEKLY_WITH_COT = out_path("kc_weekly_with_fx_cot.csv")
OUT_WEEKLY_WITH_COT_OVERLAP = out_path("kc_weekly_with_fx_cot_overlap_only.csv")

OUT_WEEKLY_WITH_WEATHER = out_path("kc_weekly_with_fx_cot_weather.csv")
OUT_WEEKLY_WITH_WEATHER_OVERLAP = out_path("kc_weekly_with_fx_cot_weather_overlap_only.csv")


# Optional overrides for holiday-delayed COT releases.
# Key = Tuesday position date, Value = actual release date.
COT_RELEASE_OVERRIDES = {
    # "2024-12-24": "2024-12-30",
}

# =========================
# Helpers
# =========================
def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Missing required column. Tried {candidates}. "
            f"Available={list(df.columns)}"
        )
    return None


def normalize_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def to_float_price(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    non_na = out.dropna()
    if not non_na.empty and non_na.abs().median() > 1e6:
        out = out / 1e9
    return out.astype(float)


def fetch_url_bytes(url: str, timeout: int = HTTP_TIMEOUT_SECONDS) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    last_err = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (URLError, HTTPError, TimeoutError, socket.timeout) as err:
            last_err = err
            print(f"[HTTP retry {attempt}/{HTTP_MAX_RETRIES}] {url} failed: {err}", flush=True)
            if attempt < HTTP_MAX_RETRIES:
                time.sleep(HTTP_RETRY_SLEEP_SECONDS * attempt)

    raise RuntimeError(f"Failed to fetch URL after retries: {url}") from last_err


def fetch_json(url: str) -> dict:
    raw = fetch_url_bytes(url)
    return json.loads(raw.decode("utf-8"))


def fetch_fred_series_csv(series_id: str) -> pd.DataFrame:
    """
    Robust FX loader for BRL/USD-like history.

    Source order:
    1) cache
    2) Banco Central do Brasil PTAX API
    3) Frankfurter historical FX API
    4) local brl_usd_daily.csv fallback
    """
    cache_name = f"fx_{series_id}.pkl"
    cached = load_cached_df(cache_name)
    if cached is not None and not cached.empty:
        print(f"Loaded FX {series_id} from cache.", flush=True)
        return cached.copy()

    errors = []

    # Primary source: official Banco Central do Brasil PTAX API
    if series_id == "DEXBZUS":
        try:
            start_ts = pd.Timestamp(FX_START)
            end_ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
            start_bcb = start_ts.strftime("%m-%d-%Y")
            end_bcb = end_ts.strftime("%m-%d-%Y")

            bcb_url = (
                "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
                "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
                f"?@dataInicial='{start_bcb}'"
                f"&@dataFinalCotacao='{end_bcb}'"
                "&$top=100000"
                "&$format=json"
                "&$select=dataHoraCotacao,cotacaoCompra"
            )

            raw = fetch_url_bytes(bcb_url, timeout=max(HTTP_TIMEOUT_SECONDS, 60))
            payload = json.loads(raw.decode("utf-8"))
            values = payload.get("value", [])
            if not values:
                raise RuntimeError(f"BCB returned no values. Keys={list(payload.keys())}")

            df = pd.DataFrame(values)
            if "dataHoraCotacao" not in df.columns or "cotacaoCompra" not in df.columns:
                raise RuntimeError(f"Unexpected BCB columns: {df.columns.tolist()}")

            df["DATE"] = pd.to_datetime(df["dataHoraCotacao"], errors="coerce").dt.normalize()
            df[series_id] = pd.to_numeric(df["cotacaoCompra"], errors="coerce")

            df = (
                df[["DATE", series_id]]
                .dropna(subset=["DATE", series_id])
                .sort_values("DATE")
                .groupby("DATE", as_index=False)
                .last()
                .reset_index(drop=True)
            )

            if df.empty:
                raise RuntimeError("BCB returned no usable rows after cleaning.")

            save_cached_df(df, cache_name)
            print(f"Fetched FX {series_id} from BCB PTAX and cached it.", flush=True)
            return df
        except Exception as exc:
            errors.append(f"BCB PTAX failed: {exc}")
            print(f"BCB PTAX source failed for {series_id}: {exc}", flush=True)

    # Secondary source: Frankfurter historical API
    if series_id == "DEXBZUS":
        try:
            start_date = FX_START
            end_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            url = (
                "https://api.frankfurter.dev/v1/"
                f"{start_date}..{end_date}"
                "?base=USD&symbols=BRL"
            )
            raw = fetch_url_bytes(url, timeout=max(HTTP_TIMEOUT_SECONDS, 60))
            payload = json.loads(raw.decode("utf-8"))
            rates = payload.get("rates", {})
            if not isinstance(rates, dict) or not rates:
                raise RuntimeError(f"Frankfurter returned no rates: keys={list(payload.keys())}")

            rows = []
            for d, vals in rates.items():
                brl = vals.get("BRL") if isinstance(vals, dict) else None
                rows.append({"DATE": d, series_id: brl})

            df = pd.DataFrame(rows)
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            df = df.dropna(subset=["DATE", series_id]).sort_values("DATE").reset_index(drop=True)

            if df.empty:
                raise RuntimeError("Frankfurter returned no usable rows after cleaning.")

            save_cached_df(df, cache_name)
            print(f"Fetched FX {series_id} from Frankfurter and cached it.", flush=True)
            return df
        except Exception as exc:
            errors.append(f"Frankfurter failed: {exc}")
            print(f"Frankfurter source failed for {series_id}: {exc}", flush=True)

    # Final fallback: prior local output file
    for local_name in ("brl_usd_daily.csv", os.path.join("data", "brl_usd_daily.csv")):
        local_path = Path(local_name)
        if local_path.exists():
            try:
                local_df = pd.read_csv(local_path)
                if "date" in local_df.columns and "brl_per_usd" in local_df.columns:
                    local_df = local_df.rename(columns={"date": "DATE", "brl_per_usd": series_id})
                if "DATE" in local_df.columns and series_id in local_df.columns:
                    local_df["DATE"] = pd.to_datetime(local_df["DATE"], errors="coerce")
                    local_df[series_id] = pd.to_numeric(local_df[series_id], errors="coerce")
                    local_df = local_df.dropna(subset=["DATE", series_id]).sort_values("DATE").reset_index(drop=True)
                    if not local_df.empty:
                        save_cached_df(local_df, cache_name)
                        print(f"Loaded FX {series_id} from local file {local_name}.", flush=True)
                        return local_df
            except Exception as exc:
                errors.append(f"Local FX fallback failed: {exc}")

    raise RuntimeError(f"FX fetch failed for {series_id}. " + " | ".join(errors))

def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name: str) -> str:
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, name)


def load_cached_df(name: str) -> pd.DataFrame | None:
    path = cache_path(name)
    if USE_CACHE and (not FORCE_REFRESH) and os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception as exc:
            print(f"Ignoring unreadable cache {path}: {exc}", flush=True)
            return None
    return None


def save_cached_df(df: pd.DataFrame, name: str) -> None:
    df.to_pickle(cache_path(name))


def _safe_series_for_log(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    out = out.where(out > 0)
    return out


def safe_log_return_series(series: pd.Series, lag: int = 1) -> pd.Series:
    s = _safe_series_for_log(series)
    return np.log(s / s.shift(lag))


def compute_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    out = df.copy()
    px = _safe_series_for_log(out[price_col])
    out[price_col] = px
    out["log_ret_1d"] = safe_log_return_series(px, 1)
    out["log_ret_5d"] = safe_log_return_series(px, 5)
    out["log_ret_20d"] = safe_log_return_series(px, 20)
    out["vol_20d"] = out["log_ret_1d"].rolling(20).std()
    out["vol_60d"] = out["log_ret_1d"].rolling(60).std()
    return out


def make_friday_sample(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out[out[date_col].dt.weekday <= 4].copy()
    out["friday_week"] = out[date_col] + pd.to_timedelta(4 - out[date_col].dt.weekday, unit="D")

    out = (
        out.sort_values(["friday_week", date_col])
           .groupby("friday_week", as_index=False)
           .tail(1)
           .sort_values("friday_week")
           .reset_index(drop=True)
    )
    return out


def chunk_date_ranges(start_date: str, end_date: str, chunk_days: int = 365):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = start

    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def build_local_kc_strip() -> pd.DataFrame:
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "try1", "data", "CoffeeCData.csv"),
        os.path.join(base_dir, "CoffeeCData.csv"),
    ]
    source_path = next((path for path in candidates if os.path.exists(path)), None)
    if source_path is None:
        raise FileNotFoundError("No local CoffeeCData.csv fallback found.")

    df = pd.read_csv(source_path)
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Fallback coffee file has unexpected columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["settlement"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["date", "settlement"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return pd.DataFrame({
        "date": df["date"],
        "instrument_id": 0,
        "raw_symbol": "KC_LOCAL",
        "expiration": df["date"] + pd.Timedelta(days=365),
        "days_to_expiry": 365,
        "contract_rank": 1,
        "settlement": df["settlement"],
        "open_interest": np.nan,
    })


# =========================
# COT helpers
# =========================
def cot_legacy_text_zip_url(year: int) -> str:
    return f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"


def load_cot_legacy_year(year: int) -> pd.DataFrame:
    url = cot_legacy_text_zip_url(year)
    raw = fetch_url_bytes(url)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"No files found inside COT zip for year {year}")

        target_name = None
        for name in names:
            lower = name.lower()
            if lower.endswith(".txt") or lower.endswith(".csv"):
                target_name = name
                break
        if target_name is None:
            target_name = names[0]

        with zf.open(target_name) as f:
            df = pd.read_csv(f, low_memory=False)

    df = df.assign(source_year=year).copy()
    return df


def load_cot_legacy_futures_only(start_year: int, end_year: int) -> pd.DataFrame:
    cache_name = f"cot_legacy_{start_year}_{end_year}.pkl"
    cached = load_cached_df(cache_name)
    if cached is not None:
        print("Loaded COT legacy data from cache.")
        return cached.copy()

    frames = []
    for year in range(start_year, end_year + 1):
        try:
            df = load_cot_legacy_year(year)
            frames.append(df)
            print(f"Loaded COT legacy year {year}: {len(df):,} rows", flush=True)
        except Exception as e:
            print(f"Skipping COT year {year}: {e}", flush=True)

    if not frames:
        raise RuntimeError("No COT legacy data loaded.")

    out = pd.concat(frames, ignore_index=True)
    save_cached_df(out, cache_name)
    return out


def find_col_contains(df: pd.DataFrame, patterns: list[str], required: bool = True):
    pats = [p.upper() for p in patterns]
    for c in df.columns:
        cu = c.upper()
        if all(p in cu for p in pats):
            return c
    if required:
        raise KeyError(f"Could not find column containing patterns {patterns}. Available={list(df.columns)}")
    return None


def transform_coffee_cot_legacy(cot_raw: pd.DataFrame) -> pd.DataFrame:
    df = cot_raw.copy()

    market_col = find_col_contains(df, ["Market", "Exchange", "Names"])
    date_col = find_col_contains(df, ["As of Date", "YYYY-MM-DD"])

    df = df[df[market_col].astype(str).str.contains("COFFEE", case=False, na=False)].copy()
    if df.empty:
        raise RuntimeError("No Coffee rows found in COT legacy dataset.")

    df["position_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    oi_col = find_col_contains(df, ["Open Interest", "(All)"])
    ncl_col = find_col_contains(df, ["Noncommercial Positions-Long", "(All)"])
    ncs_col = find_col_contains(df, ["Noncommercial Positions-Short", "(All)"])
    ncsp_col = find_col_contains(df, ["Noncommercial Positions-Spreading", "(All)"])
    cl_col = find_col_contains(df, ["Commercial Positions-Long", "(All)"])
    cs_col = find_col_contains(df, ["Commercial Positions-Short", "(All)"])
    nrl_col = find_col_contains(df, ["Nonreportable Positions-Long", "(All)"])
    nrs_col = find_col_contains(df, ["Nonreportable Positions-Short", "(All)"])

    out = pd.DataFrame({
        "market_name": df[market_col].astype(str),
        "position_date": df["position_date"],
        "open_interest": pd.to_numeric(df[oi_col], errors="coerce"),
        "noncommercial_long": pd.to_numeric(df[ncl_col], errors="coerce"),
        "noncommercial_short": pd.to_numeric(df[ncs_col], errors="coerce"),
        "noncommercial_spreading": pd.to_numeric(df[ncsp_col], errors="coerce"),
        "commercial_long": pd.to_numeric(df[cl_col], errors="coerce"),
        "commercial_short": pd.to_numeric(df[cs_col], errors="coerce"),
        "nonreportable_long": pd.to_numeric(df[nrl_col], errors="coerce"),
        "nonreportable_short": pd.to_numeric(df[nrs_col], errors="coerce"),
    })

    out = out.dropna(subset=["position_date"]).copy()
    out = out.sort_values("position_date").drop_duplicates("position_date", keep="last").reset_index(drop=True)

    out["release_date"] = out["position_date"] + pd.Timedelta(days=3)

    if COT_RELEASE_OVERRIDES:
        override_map = {pd.Timestamp(k): pd.Timestamp(v) for k, v in COT_RELEASE_OVERRIDES.items()}
        out["release_date"] = out.apply(
            lambda row: override_map.get(row["position_date"], row["release_date"]),
            axis=1
        )

    out["spec_net"] = out["noncommercial_long"] - out["noncommercial_short"]
    out["comm_net"] = out["commercial_long"] - out["commercial_short"]
    out["spec_net_pct_oi"] = out["spec_net"] / out["open_interest"]
    out["comm_net_pct_oi"] = out["comm_net"] / out["open_interest"]

    out["spec_z_26w"] = (
        (out["spec_net_pct_oi"] - out["spec_net_pct_oi"].rolling(26).mean())
        / out["spec_net_pct_oi"].rolling(26).std()
    )
    out["spec_z_52w"] = (
        (out["spec_net_pct_oi"] - out["spec_net_pct_oi"].rolling(52).mean())
        / out["spec_net_pct_oi"].rolling(52).std()
    )
    out["comm_z_26w"] = (
        (out["comm_net_pct_oi"] - out["comm_net_pct_oi"].rolling(26).mean())
        / out["comm_net_pct_oi"].rolling(26).std()
    )
    out["comm_z_52w"] = (
        (out["comm_net_pct_oi"] - out["comm_net_pct_oi"].rolling(52).mean())
        / out["comm_net_pct_oi"].rolling(52).std()
    )

    return out


# =========================
# Weather helpers
# =========================
def fetch_open_meteo_historical_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Historical daily weather using Open-Meteo reanalysis archive.
    Uses ERA5 for long-run consistency over decades.
    """
    cache_name = f"weather_hist_{latitude}_{longitude}.pkl".replace(".", "_")
    cached = load_cached_df(cache_name)
    frames = []
    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)

    if cached is not None and not cached.empty:
        cached = cached.copy()
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.sort_values("date").drop_duplicates("date")
        cached_max = cached["date"].max()
        if cached_max >= req_end:
            print(f"Loaded weather history from cache through {cached_max.date()}.")
            return cached[cached["date"].between(req_start, req_end)].reset_index(drop=True)
        frames.append(cached[cached["date"] < req_start].copy())
        frames.append(cached[cached["date"].between(req_start, cached_max)].copy())
        start_date = (cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Loaded weather history cache through {cached_max.date()}; fetching incremental update.")

    if pd.Timestamp(start_date) > req_end:
        out = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date")
        out["rain_30d"] = out["weather_rain_mm"].rolling(30, min_periods=30).sum()
        save_cached_df(out, cache_name)
        return out[out["date"].between(req_start, req_end)].reset_index(drop=True)

    for chunk_start, chunk_end in chunk_date_ranges(start_date, end_date, chunk_days=365 * 5):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "daily": ",".join([
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
            ]),
            "timezone": "UTC",
            "models": "era5",
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
        payload = fetch_json(url)

        daily = payload.get("daily")
        if not daily or "time" not in daily:
            raise RuntimeError(f"Missing historical weather daily payload for {chunk_start} -> {chunk_end}")

        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "weather_tavg_c": pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce"),
            "weather_tmax_c": pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce"),
            "weather_tmin_c": pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce"),
            "weather_precip_mm": pd.to_numeric(daily.get("precipitation_sum"), errors="coerce"),
            "weather_rain_mm": pd.to_numeric(daily.get("rain_sum"), errors="coerce"),
        })
        frames.append(df)

        print(f"Loaded weather history {chunk_start} -> {chunk_end}: {len(df):,} rows", flush=True)
        time.sleep(0.2)

    out = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date")
    out["rain_30d"] = out["weather_rain_mm"].rolling(30, min_periods=30).sum()
    save_cached_df(out, cache_name)
    return out[out["date"].between(req_start, req_end)].reset_index(drop=True)


def build_weather_weekly_friday(weather_daily: pd.DataFrame) -> pd.DataFrame:
    weekly = make_friday_sample(weather_daily, "date").rename(columns={"date": "weather_source_date"})
    return weekly[[
        "friday_week",
        "weather_source_date",
        "weather_tavg_c",
        "weather_tmax_c",
        "weather_tmin_c",
        "weather_precip_mm",
        "weather_rain_mm",
        "rain_30d",
    ]].copy()


def fetch_open_meteo_historical_forecast_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    max_horizon_days: int = 14,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Pull archived daily forecasts from Open-Meteo Historical Forecast API.

    Compared with the previous-runs endpoint, this gives materially deeper
    archived coverage (2022+) and supports broader variable access.

    We anchor each weekly row on Friday F and query the forecast that would have
    been available just before that anchor using the Thursday (F-1) issue date.
    """
    cache_name = f"weather_fcst_{latitude}_{longitude}_{max_horizon_days}_{model or 'default'}.pkl".replace(".", "_")
    cached = load_cached_df(cache_name)
    all_rows = []
    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)
    start = req_start
    end = req_end

    if cached is not None and not cached.empty:
        cached = cached.copy()
        cached["friday_week"] = pd.to_datetime(cached["friday_week"])
        cached_max = cached["friday_week"].max()
        if cached_max >= req_end:
            print(f"Loaded weather forecast history from cache through {cached_max.date()}.", flush=True)
            return cached[cached["friday_week"].between(req_start, req_end)].reset_index(drop=True)
        all_rows.append(cached[cached["friday_week"] < req_start].copy())
        all_rows.append(cached[cached["friday_week"].between(req_start, cached_max)].copy())
        start = cached_max + pd.Timedelta(days=7)
        print(f"Loaded weather forecast cache through {cached_max.date()}; fetching incremental update.", flush=True)

    friday_range = pd.date_range(start=start, end=end, freq="W-FRI")
    print(f"Fetching archived forecast history from {start.date()} to {end.date()} for {len(friday_range)} Friday anchors...", flush=True)

    for i, friday in enumerate(friday_range, start=1):
        issue_date = friday - pd.Timedelta(days=1)
        forecast_end = friday + pd.Timedelta(days=max_horizon_days)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": issue_date.strftime("%Y-%m-%d"),
            "end_date": forecast_end.strftime("%Y-%m-%d"),
            "daily": ",".join([
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
            ]),
            "timezone": "UTC",
        }
        if model:
            params["models"] = model

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        try:
            payload = fetch_json(url)
        except Exception as exc:
            print(f"Skipping historical forecast for {friday.date()}: {exc}", flush=True)
            time.sleep(0.05)
            continue

        daily = payload.get("daily")
        if not daily or "time" not in daily:
            print(f"Skipping historical forecast for {friday.date()}: missing payload", flush=True)
            time.sleep(0.05)
            continue

        df = pd.DataFrame({
            "target_date": pd.to_datetime(daily["time"]),
            "forecast_tavg_c": pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce"),
            "forecast_tmax_c": pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce"),
            "forecast_tmin_c": pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce"),
            "forecast_precip_mm": pd.to_numeric(daily.get("precipitation_sum"), errors="coerce"),
            "forecast_rain_mm": pd.to_numeric(daily.get("rain_sum"), errors="coerce"),
        })

        df = df[(df["target_date"] > issue_date) & (df["target_date"] <= forecast_end)].copy()
        if df.empty:
            time.sleep(0.02)
            continue

        df["friday_week"] = friday.normalize()
        df["issue_date"] = issue_date.normalize()
        df["forecast_source_date"] = issue_date.normalize()
        df["forecast_horizon_days"] = (df["target_date"] - df["issue_date"]).dt.days
        all_rows.append(df[[
            "friday_week", "issue_date", "forecast_source_date",
            "forecast_horizon_days", "target_date",
            "forecast_rain_mm", "forecast_tavg_c", "forecast_tmin_c", "forecast_tmax_c"
        ]])

        if i % 5 == 0:
            print(f"Loaded archived forecast anchors through {friday.date()} ({i:,} Fridays)", flush=True)
        time.sleep(0.02)

    if not all_rows:
        return pd.DataFrame(columns=[
            "friday_week", "issue_date", "forecast_source_date", "forecast_horizon_days",
            "target_date", "forecast_rain_mm", "forecast_tavg_c", "forecast_tmin_c", "forecast_tmax_c"
        ])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["friday_week", "forecast_horizon_days", "target_date"]).reset_index(drop=True)
    save_cached_df(out, cache_name)
    return out[out["friday_week"].between(req_start, req_end)].reset_index(drop=True)


def summarize_weather_forecast_friday(forecast_long_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_long_df.empty:
        return pd.DataFrame(columns=["friday_week"])

    fc = forecast_long_df.copy()
    fc["forecast_horizon_days"] = pd.to_numeric(fc["forecast_horizon_days"], errors="coerce")

    def _summarize(g: pd.DataFrame) -> pd.Series:
        window_1_7 = g["forecast_horizon_days"].between(1, 7)
        window_8_14 = g["forecast_horizon_days"].between(8, 14)
        return pd.Series({
            "fcst_rain_1_7d": g.loc[window_1_7, "forecast_rain_mm"].sum(min_count=1),
            "fcst_rain_8_14d": g.loc[window_8_14, "forecast_rain_mm"].sum(min_count=1),
            "fcst_rain_1_14d": g.loc[g["forecast_horizon_days"].between(1, 14), "forecast_rain_mm"].sum(min_count=1),
            "fcst_tavg_1_7d": g.loc[window_1_7, "forecast_tavg_c"].mean(),
            "fcst_tavg_8_14d": g.loc[window_8_14, "forecast_tavg_c"].mean(),
            "fcst_tmin_1_7d": g.loc[window_1_7, "forecast_tmin_c"].min(),
            "fcst_tmax_1_7d": g.loc[window_1_7, "forecast_tmax_c"].max(),
            "fcst_cold_flag_1_7d": float((g.loc[window_1_7, "forecast_tmin_c"] <= 2.0).any()) if window_1_7.any() else np.nan,
            "fcst_heat_flag_1_7d": float((g.loc[window_1_7, "forecast_tmax_c"] >= 34.0).any()) if window_1_7.any() else np.nan,
        })

    out = fc.groupby("friday_week").apply(_summarize).reset_index()
    return out

# =========================
# Databento: KC strip (cost-optimized)
# =========================
strip_parts = []

# Keep a cheap local history fallback for older dates if available.
try:
    local_strip = build_local_kc_strip()
    local_cutoff = pd.Timestamp(DATABENTO_KC_START)
    strip_parts.append(local_strip[local_strip["date"] < local_cutoff].copy())
    print(f"Loaded local CoffeeCData.csv history through {(local_cutoff - pd.Timedelta(days=1)).date()}.", flush=True)
except Exception:
    pass

try:
    cache_name = f"databento_kc_continuous_depth{CONTINUOUS_DEPTH}.pkl"
    cached_db_strip = load_cached_df(cache_name)
    frames_db = []
    client = None

    bar_start_ts = max(pd.Timestamp(KC_START), pd.Timestamp(DATABENTO_KC_START))
    latest_safe_databento_day = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    requested_end_ts = pd.Timestamp(KC_END) if KC_END is not None else latest_safe_databento_day
    bar_end_ts = min(requested_end_ts, latest_safe_databento_day)

    if cached_db_strip is not None and not cached_db_strip.empty:
        cached_db_strip = cached_db_strip.copy()
        cached_db_strip["date"] = pd.to_datetime(cached_db_strip["date"])
        cached_max = cached_db_strip["date"].max()
        frames_db.append(cached_db_strip[cached_db_strip["date"] < bar_start_ts].copy())
        frames_db.append(cached_db_strip[cached_db_strip["date"].between(bar_start_ts, min(cached_max, bar_end_ts))].copy())
        print(f"Loaded Databento cache through {cached_max.date()}.", flush=True)
        fetch_start_ts = max(bar_start_ts, cached_max + pd.Timedelta(days=1))
    else:
        fetch_start_ts = bar_start_ts

    if fetch_start_ts <= bar_end_ts:
        client = db.Historical(API_KEY)
        continuous_symbols = [f"KC.v.{i}" for i in range(CONTINUOUS_DEPTH)]
        fetch_end_exclusive_ts = bar_end_ts + pd.Timedelta(days=1)
        print(
            f"Fetching Databento KC bars from {fetch_start_ts.date()} to {(fetch_end_exclusive_ts - pd.Timedelta(days=1)).date()}...",
            flush=True,
        )
        bars_raw = client.timeseries.get_range(
            dataset=DATASET,
            symbols=continuous_symbols,
            stype_in="continuous",
            schema="ohlcv-1d",
            start=fetch_start_ts.strftime("%Y-%m-%d"),
            end=fetch_end_exclusive_ts.strftime("%Y-%m-%d"),
        ).to_df().reset_index()

        if not bars_raw.empty:
            bars = bars_raw.copy()
            symbol_col = pick_col(bars, ["symbol", "raw_symbol"])
            ts_event_col = pick_col(bars, ["ts_event", "ts_recv"])
            close_col = pick_col(bars, ["close"])

            bars["date"] = normalize_ts(bars[ts_event_col]).dt.normalize()
            bars["raw_symbol"] = bars[symbol_col].astype(str)
            bars["settlement"] = to_float_price(bars[close_col])
            bars = bars.dropna(subset=["date", "raw_symbol", "settlement"]).copy()

            bars["contract_rank"] = (
                bars["raw_symbol"].str.extract(r"\.v\.(\d+)$", expand=False).astype(float) + 1
            )
            bars = bars.dropna(subset=["contract_rank"]).copy()
            bars["contract_rank"] = bars["contract_rank"].astype(int)

            bars["days_to_expiry"] = bars["contract_rank"] * 30
            bars["expiration"] = bars["date"] + pd.to_timedelta(bars["days_to_expiry"], unit="D")
            bars["instrument_id"] = bars["contract_rank"].astype(int)
            bars["open_interest"] = np.nan

            fetched_db_strip = bars[[
                "date", "instrument_id", "raw_symbol", "expiration",
                "days_to_expiry", "contract_rank", "settlement", "open_interest"
            ]].copy()
            frames_db.append(fetched_db_strip)
            merged_cache = pd.concat(frames_db, ignore_index=True).sort_values(["date", "contract_rank"]).drop_duplicates(["date", "raw_symbol"], keep="last")
            save_cached_df(merged_cache, cache_name)
            print(f"Fetched and cached Databento KC bars through {(fetch_end_exclusive_ts - pd.Timedelta(days=1)).date()}.", flush=True)
        else:
            print("No new Databento rows returned; using cached data only.", flush=True)

    else:
        print("Databento cache is already up to date enough; skipping live fetch.", flush=True)

    if frames_db:
        db_strip = pd.concat(frames_db, ignore_index=True).sort_values(["date", "contract_rank"]).drop_duplicates(["date", "raw_symbol"], keep="last")
        strip_parts.append(db_strip)
    elif cached_db_strip is not None:
        strip_parts.append(cached_db_strip)

except Exception as exc:
    if not strip_parts:
        print(f"Databento KC request failed ({exc}); falling back to local CoffeeCData.csv.", flush=True)
        strip_parts = [build_local_kc_strip()]
    else:
        print(f"Databento KC request failed ({exc}); using cached/local coffee data only.", flush=True)

strip = pd.concat(strip_parts, ignore_index=True)

strip["date"] = pd.to_datetime(strip["date"])
strip["days_to_expiry"] = (strip["expiration"].dt.normalize() - strip["date"]).dt.days
strip = strip[strip["days_to_expiry"] >= 0].copy()
strip = strip.sort_values(["date", "expiration", "instrument_id"]).reset_index(drop=True)
strip["contract_rank"] = strip.groupby("date")["expiration"].rank(method="dense").astype(int)

strip_out = strip[[
    "date",
    "instrument_id",
    "raw_symbol",
    "expiration",
    "days_to_expiry",
    "contract_rank",
    "settlement",
    "open_interest",
]].copy()

strip_out["date"] = strip_out["date"].dt.strftime("%Y-%m-%d")
strip_out["expiration"] = pd.to_datetime(strip_out["expiration"]).dt.strftime("%Y-%m-%d")
strip_out.to_csv(OUT_STRIP, index=False)

# =========================
# Daily continuous curve
# =========================
cont = (
    strip.sort_values(["date", "contract_rank", "expiration"])
         .drop_duplicates(subset=["date"], keep="first")
         .copy()
)

cont = cont[[
    "date",
    "instrument_id",
    "raw_symbol",
    "expiration",
    "days_to_expiry",
    "settlement",
    "open_interest",
    "contract_rank",
]].rename(columns={"settlement": "price"})

cont = cont.sort_values("date").reset_index(drop=True)
cont = compute_features(cont, "price")
cont["roll_changed"] = cont["instrument_id"] != cont["instrument_id"].shift(1)

cont_out = cont.copy()
cont_out["date"] = cont_out["date"].dt.strftime("%Y-%m-%d")
cont_out["expiration"] = pd.to_datetime(cont_out["expiration"]).dt.strftime("%Y-%m-%d")
cont_out.to_csv(OUT_CONT, index=False)

# =========================
# Weekly Friday sample for futures
# =========================
weekly = make_friday_sample(cont, "date")

weekly_out = weekly[[
    "friday_week",
    "date",
    "price",
    "instrument_id",
    "raw_symbol",
    "expiration",
    "days_to_expiry",
    "open_interest",
    "contract_rank",
    "log_ret_1d",
    "log_ret_5d",
    "log_ret_20d",
    "vol_20d",
    "vol_60d",
    "roll_changed",
]].rename(columns={"date": "source_date"})

weekly_out["friday_week"] = pd.to_datetime(weekly_out["friday_week"]).dt.strftime("%Y-%m-%d")
weekly_out["source_date"] = pd.to_datetime(weekly_out["source_date"]).dt.strftime("%Y-%m-%d")
weekly_out["expiration"] = pd.to_datetime(weekly_out["expiration"]).dt.strftime("%Y-%m-%d")
weekly_out.to_csv(OUT_WEEKLY, index=False)

# =========================
# FX daily and weekly
# =========================
print("Starting FX download / cache load...", flush=True)
fx = fetch_fred_series_csv(FX_SERIES)

date_col = pick_col(fx, ["DATE", "date"])
value_col = pick_col(fx, [FX_SERIES, "value"])

fx = fx[[date_col, value_col]].copy()
fx.columns = ["date", "brl_per_usd"]
fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
fx["brl_per_usd"] = pd.to_numeric(fx["brl_per_usd"], errors="coerce")
fx = fx.dropna(subset=["date", "brl_per_usd"]).copy()
fx = fx[fx["date"] >= pd.Timestamp(FX_START)].sort_values("date").reset_index(drop=True)

fx = compute_features(fx, "brl_per_usd")

fx_out = fx.copy()
fx_out["date"] = fx_out["date"].dt.strftime("%Y-%m-%d")
fx_out.to_csv(OUT_FX_DAILY, index=False)

fx_weekly = make_friday_sample(fx, "date").rename(columns={"date": "source_date"})
fx_weekly_out = fx_weekly.copy()
fx_weekly_out["friday_week"] = fx_weekly_out["friday_week"].dt.strftime("%Y-%m-%d")
fx_weekly_out["source_date"] = fx_weekly_out["source_date"].dt.strftime("%Y-%m-%d")
fx_weekly_out.to_csv(OUT_FX_WEEKLY, index=False)

# =========================
# Join futures + FX
# =========================
weekly_joined = weekly_out.merge(
    fx_weekly_out[[
        "friday_week",
        "source_date",
        "brl_per_usd",
        "log_ret_1d",
        "log_ret_5d",
        "log_ret_20d",
        "vol_20d",
        "vol_60d",
    ]],
    on="friday_week",
    how="left",
    suffixes=("", "_fx"),
)

weekly_joined = weekly_joined.rename(columns={
    "source_date_fx": "fx_source_date",
    "log_ret_1d_fx": "fx_log_ret_1d",
    "log_ret_5d_fx": "fx_log_ret_5d",
    "log_ret_20d_fx": "fx_log_ret_20d",
    "vol_20d_fx": "fx_vol_20d",
    "vol_60d_fx": "fx_vol_60d",
})

weekly_joined.to_csv(OUT_WEEKLY_JOINED, index=False)

weekly_joined_overlap = weekly_joined.loc[
    weekly_joined["price"].notna() & weekly_joined["brl_per_usd"].notna()
].copy()

weekly_joined_overlap = weekly_joined_overlap.sort_values("friday_week").reset_index(drop=True)
weekly_joined_overlap.to_csv(OUT_WEEKLY_JOINED_OVERLAP, index=False)

# =========================
# COT weekly
# =========================
print("Starting COT download / cache load...", flush=True)
cot_raw = load_cot_legacy_futures_only(COT_START_YEAR, COT_END_YEAR)
cot_weekly = transform_coffee_cot_legacy(cot_raw)

cot_out = cot_weekly.copy()
cot_out["position_date"] = cot_out["position_date"].dt.strftime("%Y-%m-%d")
cot_out["release_date"] = cot_out["release_date"].dt.strftime("%Y-%m-%d")
cot_out.to_csv(OUT_COT_WEEKLY, index=False)

# =========================
# Merge COT by release date
# =========================
weekly_with_cot = weekly_joined.copy()
weekly_with_cot["friday_week"] = pd.to_datetime(weekly_with_cot["friday_week"])

cot_merge = cot_weekly.copy()
cot_merge["release_date"] = pd.to_datetime(cot_merge["release_date"])

weekly_with_cot = weekly_with_cot.merge(
    cot_merge,
    left_on="friday_week",
    right_on="release_date",
    how="left",
)

for col in ["friday_week", "source_date", "fx_source_date", "position_date", "release_date", "expiration"]:
    if col in weekly_with_cot.columns:
        weekly_with_cot[col] = pd.to_datetime(weekly_with_cot[col], errors="coerce").dt.strftime("%Y-%m-%d")

weekly_with_cot.to_csv(OUT_WEEKLY_WITH_COT, index=False)

weekly_with_cot_overlap = weekly_with_cot.loc[
    weekly_with_cot["price"].notna()
    & weekly_with_cot["brl_per_usd"].notna()
    & weekly_with_cot["spec_net"].notna()
].copy()

weekly_with_cot_overlap = weekly_with_cot_overlap.sort_values("friday_week").reset_index(drop=True)
weekly_with_cot_overlap.to_csv(OUT_WEEKLY_WITH_COT_OVERLAP, index=False)

# =========================
# Weather historical daily + Friday alignment
# =========================
print("Starting historical weather download / cache load...", flush=True)
weather_hist_end = WEATHER_HIST_END or pd.Timestamp.today().strftime("%Y-%m-%d")
weather_daily = fetch_open_meteo_historical_daily(
    latitude=WEATHER_LAT,
    longitude=WEATHER_LON,
    start_date=WEATHER_HIST_START,
    end_date=weather_hist_end,
)

weather_daily_out = weather_daily.copy()
weather_daily_out["date"] = weather_daily_out["date"].dt.strftime("%Y-%m-%d")
weather_daily_out.to_csv(OUT_WEATHER_DAILY, index=False)

weather_weekly = build_weather_weekly_friday(weather_daily)
weather_weekly_out = weather_weekly.copy()
weather_weekly_out["friday_week"] = pd.to_datetime(weather_weekly_out["friday_week"]).dt.strftime("%Y-%m-%d")
weather_weekly_out["weather_source_date"] = pd.to_datetime(weather_weekly_out["weather_source_date"]).dt.strftime("%Y-%m-%d")
weather_weekly_out.to_csv(OUT_WEATHER_WEEKLY, index=False)

# =========================
# Weather forecast archive at Friday
# =========================
print("Starting historical forecast download / cache load...", flush=True)
weather_fcst_end = WEATHER_FORECAST_END or pd.Timestamp.today().strftime("%Y-%m-%d")
weather_forecast_long = fetch_open_meteo_historical_forecast_daily(
    latitude=WEATHER_LAT,
    longitude=WEATHER_LON,
    start_date=WEATHER_FORECAST_START,
    end_date=weather_fcst_end,
    max_horizon_days=WEATHER_FORECAST_MAX_HORIZON_DAYS,
    model=WEATHER_FORECAST_MODEL,
)

weather_forecast_out = weather_forecast_long.copy()
for col in ["friday_week", "issue_date", "forecast_source_date", "target_date"]:
    weather_forecast_out[col] = pd.to_datetime(weather_forecast_out[col]).dt.strftime("%Y-%m-%d")
weather_forecast_out.to_csv(OUT_WEATHER_FORECAST_FRIDAY, index=False)

# =========================
# Merge weather into weekly panel
# =========================
weekly_with_weather = weekly_with_cot.copy()
weekly_with_weather["friday_week"] = pd.to_datetime(weekly_with_weather["friday_week"])

weather_weekly_merge = weather_weekly.copy()
weather_weekly_merge["friday_week"] = pd.to_datetime(weather_weekly_merge["friday_week"])

weekly_with_weather = weekly_with_weather.merge(
    weather_weekly_merge,
    on="friday_week",
    how="left",
)

for col in [
    "friday_week",
    "source_date",
    "fx_source_date",
    "position_date",
    "release_date",
    "expiration",
    "weather_source_date",
]:
    if col in weekly_with_weather.columns:
        weekly_with_weather[col] = pd.to_datetime(weekly_with_weather[col], errors="coerce").dt.strftime("%Y-%m-%d")

weekly_with_weather.to_csv(OUT_WEEKLY_WITH_WEATHER, index=False)

weekly_with_weather_overlap = weekly_with_weather.loc[
    weekly_with_weather["price"].notna()
    & weekly_with_weather["brl_per_usd"].notna()
    & weekly_with_weather["spec_net"].notna()
    & weekly_with_weather["rain_30d"].notna()
].copy()

weekly_with_weather_overlap = weekly_with_weather_overlap.sort_values("friday_week").reset_index(drop=True)
weekly_with_weather_overlap.to_csv(OUT_WEEKLY_WITH_WEATHER_OVERLAP, index=False)



# =========================
# Advanced curve / model panel helpers
# =========================
def _interp_price_for_target_dte(group: pd.DataFrame, target_dte: float) -> float:
    g = (
        group[["days_to_expiry", "settlement"]]
        .dropna()
        .drop_duplicates("days_to_expiry")
        .sort_values("days_to_expiry")
    )
    if g.empty:
        return np.nan

    dtes = g["days_to_expiry"].to_numpy(dtype=float)
    prices = g["settlement"].to_numpy(dtype=float)

    if target_dte in dtes:
        return float(prices[np.where(dtes == target_dte)[0][0]])

    lower_mask = dtes < target_dte
    upper_mask = dtes > target_dte

    if lower_mask.any() and upper_mask.any():
        d0 = dtes[lower_mask].max()
        d1 = dtes[upper_mask].min()
        p0 = prices[dtes == d0][0]
        p1 = prices[dtes == d1][0]
        w = (target_dte - d0) / (d1 - d0)
        return float(p0 + w * (p1 - p0))

    idx = np.argmin(np.abs(dtes - target_dte))
    return float(prices[idx])


def build_curve_features_daily(strip_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in strip_df.groupby("date", sort=True):
        g = g.sort_values(["days_to_expiry", "expiration", "instrument_id"]).copy()
        row = {"date": pd.to_datetime(date)}

        for r in range(1, 7):
            gr = g[g["contract_rank"].eq(r)]
            row[f"rank{r}_price"] = gr["settlement"].iloc[0] if not gr.empty else np.nan
            row[f"rank{r}_dte"] = gr["days_to_expiry"].iloc[0] if not gr.empty else np.nan
            row[f"rank{r}_oi"] = gr["open_interest"].iloc[0] if not gr.empty else np.nan

        for target_dte, label in [(30, "cm_1m"), (90, "cm_3m"), (180, "cm_6m")]:
            row[f"{label}_price"] = _interp_price_for_target_dte(g, target_dte)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["spread_1_2"] = out["rank1_price"] - out["rank2_price"]
    out["spread_1_3"] = out["rank1_price"] - out["rank3_price"]
    out["spread_3_6"] = out["cm_3m_price"] - out["cm_6m_price"]
    out["basis_cm1_spotproxy"] = out["cm_1m_price"] - out["rank1_price"]
    out["curve_slope_1_6"] = out["cm_6m_price"] - out["rank1_price"]
    out["curve_curvature_1_3_6"] = 2 * out["cm_3m_price"] - out["rank1_price"] - out["cm_6m_price"]

    for col in ["rank1_price", "cm_3m_price", "cm_6m_price", "curve_slope_1_6", "curve_curvature_1_3_6"]:
        out[f"{col}_ret_5d"] = safe_log_return_series(out[col], 5)
        out[f"{col}_ret_20d"] = safe_log_return_series(out[col], 20)

    return out


def add_weather_anomalies(weather_daily_df: pd.DataFrame) -> pd.DataFrame:
    out = weather_daily_df.copy().sort_values("date").reset_index(drop=True)
    out["doy"] = out["date"].dt.dayofyear.clip(upper=365)

    clim = (
        out.groupby("doy", as_index=False)
           .agg(
               clim_tavg_c=("weather_tavg_c", "mean"),
               clim_tmin_c=("weather_tmin_c", "mean"),
               clim_rain_mm=("weather_rain_mm", "mean"),
           )
    )

    out = out.merge(clim, on="doy", how="left")
    out["tavg_anom"] = out["weather_tavg_c"] - out["clim_tavg_c"]
    out["tmin_anom"] = out["weather_tmin_c"] - out["clim_tmin_c"]
    out["rain_anom"] = out["weather_rain_mm"] - out["clim_rain_mm"]

    out["tavg_anom_30d"] = out["tavg_anom"].rolling(30, min_periods=30).mean()
    out["tmin_anom_30d"] = out["tmin_anom"].rolling(30, min_periods=30).mean()
    out["rain_anom_30d"] = out["rain_anom"].rolling(30, min_periods=30).sum()
    out["rain_7d"] = out["weather_rain_mm"].rolling(7, min_periods=7).sum()
    out["tmin_7d_min"] = out["weather_tmin_c"].rolling(7, min_periods=7).min()
    out["frost_7d_flag"] = (out["tmin_7d_min"] <= 2.0).astype(float)
    out["heat_7d_flag"] = (out["weather_tmax_c"].rolling(7, min_periods=7).max() >= 34.0).astype(float)

    denom = out["clim_rain_mm"].rolling(30, min_periods=30).sum().replace(0, np.nan)
    out["rain_ratio_30d"] = out["rain_30d"] / denom
    return out


def add_lagged_and_interaction_features(model_panel: pd.DataFrame) -> pd.DataFrame:
    out = model_panel.copy().sort_values("friday_week").reset_index(drop=True)

    base_cols = [
        col for col in [
            "price", "brl_per_usd", "spec_net", "comm_net", "spec_net_pct_oi", "comm_net_pct_oi",
            "rain_30d", "rain_7d", "tavg_anom_30d", "tmin_anom_30d", "rain_ratio_30d",
            "rank1_price", "cm_3m_price", "cm_6m_price", "curve_slope_1_6", "curve_curvature_1_3_6",
            "spot_proxy_price",
        ]
        if col in out.columns
    ]

    lag_steps = [1, 4, 13, 26]
    for col in base_cols:
        for lag in lag_steps:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    interaction_pairs = [
        ("price", "brl_per_usd"),
        ("price", "spec_net"),
        ("price", "rain_30d"),
        ("rank1_price", "brl_per_usd"),
        ("rank1_price", "spec_net"),
        ("cm_3m_price", "cm_6m_price"),
    ]
    for left, right in interaction_pairs:
        if left in out.columns and right in out.columns:
            out[f"{left}_{right}_interaction"] = out[left] * out[right]

    return out


def add_forward_targets(model_panel: pd.DataFrame) -> pd.DataFrame:
    out = model_panel.copy().sort_values("friday_week").reset_index(drop=True)
    price_col = "spot_proxy_price" if "spot_proxy_price" in out.columns else "price"
    if price_col not in out.columns:
        raise KeyError("Expected spot_proxy_price or price column for forward targets.")

    future_windows = [1, 4, 13, 26]
    base_px = _safe_series_for_log(out[price_col])
    for horizon in future_windows:
        future_px = _safe_series_for_log(out[price_col].shift(-horizon))
        out[f"target_ret_{horizon}w"] = np.log(future_px / base_px)

    return out


OUT_WEATHER_FORECAST_WEEKLY = out_path(f"{WEATHER_NAME}_weather_forecast_friday_summary.csv")
OUT_MODEL_PANEL = out_path("kc_model_panel_weekly.csv")
OUT_MODEL_PANEL_OVERLAP = out_path("kc_model_panel_weekly_overlap_only.csv")

curve_daily = build_curve_features_daily(strip)
curve_daily_out = curve_daily.copy()
curve_daily_out["date"] = curve_daily_out["date"].dt.strftime("%Y-%m-%d")
curve_daily_out.to_csv(OUT_CURVE_DAILY, index=False)

curve_weekly = make_friday_sample(curve_daily, "date").rename(columns={"date": "curve_source_date"})
curve_weekly_out = curve_weekly.copy()
curve_weekly_out["friday_week"] = pd.to_datetime(curve_weekly_out["friday_week"]).dt.strftime("%Y-%m-%d")
curve_weekly_out["curve_source_date"] = pd.to_datetime(curve_weekly_out["curve_source_date"]).dt.strftime("%Y-%m-%d")
curve_weekly_out.to_csv(OUT_CURVE_WEEKLY, index=False)

weather_daily_enriched = add_weather_anomalies(weather_daily)
weather_daily_enriched_out = weather_daily_enriched.copy()
weather_daily_enriched_out["date"] = weather_daily_enriched_out["date"].dt.strftime("%Y-%m-%d")
weather_daily_enriched_out.to_csv(OUT_WEATHER_DAILY_ENRICHED, index=False)

weather_weekly_enriched = make_friday_sample(weather_daily_enriched, "date").rename(columns={"date": "weather_source_date"})
weather_weekly_enriched_out = weather_weekly_enriched.copy()
weather_weekly_enriched_out["friday_week"] = pd.to_datetime(weather_weekly_enriched_out["friday_week"]).dt.strftime("%Y-%m-%d")
weather_weekly_enriched_out["weather_source_date"] = pd.to_datetime(weather_weekly_enriched_out["weather_source_date"]).dt.strftime("%Y-%m-%d")
weather_weekly_enriched_out.to_csv(OUT_WEATHER_WEEKLY_ENRICHED, index=False)

print("Summarizing weekly weather forecast features...")
weather_forecast_weekly = summarize_weather_forecast_friday(weather_forecast_long)
weather_forecast_weekly_out = weather_forecast_weekly.copy()
weather_forecast_weekly_out["friday_week"] = pd.to_datetime(weather_forecast_weekly_out["friday_week"]).dt.strftime("%Y-%m-%d")
weather_forecast_weekly_out.to_csv(OUT_WEATHER_FORECAST_WEEKLY, index=False)

print("Building final model panel...")
model_panel = weekly_with_weather.copy()
model_panel["friday_week"] = pd.to_datetime(model_panel["friday_week"])

curve_weekly_merge = curve_weekly.copy()
curve_weekly_merge["friday_week"] = pd.to_datetime(curve_weekly_merge["friday_week"])
model_panel = model_panel.merge(
    curve_weekly_merge,
    on="friday_week",
    how="left",
    suffixes=("", "_curve")
)

weather_weekly_enriched_merge = weather_weekly_enriched.copy()
weather_weekly_enriched_merge["friday_week"] = pd.to_datetime(weather_weekly_enriched_merge["friday_week"])
keep_weather_cols = [
    "friday_week", "weather_source_date", "weather_tavg_c", "weather_tmax_c", "weather_tmin_c",
    "weather_precip_mm", "weather_rain_mm", "rain_30d", "rain_7d", "rain_anom_30d",
    "tavg_anom_30d", "tmin_anom_30d", "rain_ratio_30d", "frost_7d_flag", "heat_7d_flag",
]
model_panel = model_panel.drop(columns=[c for c in keep_weather_cols if c in model_panel.columns and c != "friday_week"], errors="ignore")
model_panel = model_panel.merge(
    weather_weekly_enriched_merge[keep_weather_cols],
    on="friday_week",
    how="left",
)

weather_forecast_weekly_merge = weather_forecast_weekly.copy()
weather_forecast_weekly_merge["friday_week"] = pd.to_datetime(weather_forecast_weekly_merge["friday_week"])
model_panel = model_panel.merge(weather_forecast_weekly_merge, on="friday_week", how="left")

if "rank1_price" in model_panel.columns:
    model_panel["spot_proxy_price"] = model_panel["rank1_price"]

model_panel = add_lagged_and_interaction_features(model_panel)
model_panel = add_forward_targets(model_panel)

model_panel_out = model_panel.copy()
for col in [
    "friday_week", "source_date", "fx_source_date", "position_date", "release_date",
    "expiration", "weather_source_date", "curve_source_date",
]:
    if col in model_panel_out.columns:
        model_panel_out[col] = pd.to_datetime(model_panel_out[col], errors="coerce").dt.strftime("%Y-%m-%d")
model_panel_out.to_csv(OUT_MODEL_PANEL, index=False)

required_cols = [
    "price", "brl_per_usd", "spec_net", "rain_30d", "rank1_price", "cm_3m_price", "cm_6m_price", "target_ret_26w"
]
required_present = [c for c in required_cols if c in model_panel.columns]
model_panel_overlap = model_panel.loc[model_panel[required_present].notna().all(axis=1)].copy()
model_panel_overlap = model_panel_overlap.sort_values("friday_week").reset_index(drop=True)
model_panel_overlap_out = model_panel_overlap.copy()
for col in [
    "friday_week", "source_date", "fx_source_date", "position_date", "release_date",
    "expiration", "weather_source_date", "curve_source_date",
]:
    if col in model_panel_overlap_out.columns:
        model_panel_overlap_out[col] = pd.to_datetime(model_panel_overlap_out[col], errors="coerce").dt.strftime("%Y-%m-%d")
model_panel_overlap_out.to_csv(OUT_MODEL_PANEL_OVERLAP, index=False)

# =========================
# Summary
# =========================
print("Saved:")
print(f"  {OUT_STRIP}")
print(f"  {OUT_CONT}")
print(f"  {OUT_WEEKLY}")
print(f"  {OUT_FX_DAILY}")
print(f"  {OUT_FX_WEEKLY}")
print(f"  {OUT_COT_WEEKLY}")
print(f"  {OUT_WEATHER_DAILY}")
print(f"  {OUT_WEATHER_WEEKLY}")
print(f"  {OUT_WEATHER_FORECAST_FRIDAY}")
print(f"  {OUT_WEEKLY_JOINED}")
print(f"  {OUT_WEEKLY_JOINED_OVERLAP}")
print(f"  {OUT_WEEKLY_WITH_COT}")
print(f"  {OUT_WEEKLY_WITH_COT_OVERLAP}")
print(f"  {OUT_WEEKLY_WITH_WEATHER}")
print(f"  {OUT_WEEKLY_WITH_WEATHER_OVERLAP}")

if not weekly_joined_overlap.empty:
    overlap_start = weekly_joined_overlap["friday_week"].min()
    overlap_end = weekly_joined_overlap["friday_week"].max()
    print(f"\nCoffee + FX overlap window: {overlap_start} -> {overlap_end}")

if not weekly_with_cot_overlap.empty:
    overlap_start = weekly_with_cot_overlap["friday_week"].min()
    overlap_end = weekly_with_cot_overlap["friday_week"].max()
    print(f"Coffee + FX + COT overlap window: {overlap_start} -> {overlap_end}")

if not weekly_with_weather_overlap.empty:
    overlap_start = weekly_with_weather_overlap["friday_week"].min()
    overlap_end = weekly_with_weather_overlap["friday_week"].max()
    print(f"Coffee + FX + COT + Weather overlap window: {overlap_start} -> {overlap_end}")

contracts_per_day = strip.groupby("date")["instrument_id"].nunique()
print("\nContracts per day summary:")
print(contracts_per_day.describe())

print("\nSample weather weekly rows:")
print(weather_weekly.head(10).to_string(index=False))

print("\nSample weather forecast rows:")
print(weather_forecast_long.head(10).to_string(index=False))

print("\nSample final overlap rows:")
print(weekly_with_weather_overlap.head(10).to_string(index=False))
