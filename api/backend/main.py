from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from backend.config import (
    BASE_DIR,
    BLOB_TOKEN,
    DATA_DIR,
    DATA_FETCH_TIMEOUT_SECONDS,
    LOGDATA_DIR,
    OUTPUT_DIR,
)
from backend.ml import coffee_data_merged, coffee_xgboost_train
from backend.pipeline import PipelineError, get_prediction
from backend.schemas import (
    BriefResponse,
    ContractRow,
    HealthResponse,
    NewsItem,
    PredictionResponse,
    ProjectedSpotCsvFiles,
    ProjectedSpotCsvResponse,
    SnapshotResponse,
)
from backend.storage.cache import CacheStore

app = FastAPI(
    title='Coffee XGBoost Prediction API',
    version='1.0.0',
    description='FastAPI app for Vercel that runs your coffee data pipeline, builds projections, and caches results online.',
)


PROJECT_ROOT = BASE_DIR.parent
WEBSITE_PUBLIC_DATA_DIR = PROJECT_ROOT / 'website' / 'public' / 'data'
ROOT_OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
DATA_FETCH_SCRIPT_PATH = BASE_DIR / 'data_fetch.py'
MARKET_JSON_DIRS = [DATA_DIR, WEBSITE_PUBLIC_DATA_DIR]
MARKET_CSV_DIRS = [WEBSITE_PUBLIC_DATA_DIR, OUTPUT_DIR, ROOT_OUTPUTS_DIR, DATA_DIR]

_REFRESH_LOCK = threading.Lock()


def _read_json_file(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'Missing file: {path.name}') from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f'Invalid JSON in {path.name}') from exc


def _read_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'Missing file: {path.name}') from exc
    if not text.strip():
        raise HTTPException(status_code=500, detail=f'File is empty: {path.name}')
    return text


def _first_existing_path(file_name: str, candidate_dirs: list[Path]) -> Path | None:
    for directory in candidate_dirs:
        candidate = directory / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _require_data_file(file_name: str, candidate_dirs: list[Path]) -> Path:
    path = _first_existing_path(file_name, candidate_dirs)
    if path is None:
        raise HTTPException(status_code=404, detail=f'{file_name} not found in configured data locations')
    return path


def _last_friday(d: date) -> date:
    return d - timedelta(days=(d.weekday() - 4) % 7)


def _file_is_stale_since_last_friday(path: Path, cutoff_friday: date) -> bool:
    mtime_date = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    return mtime_date < cutoff_friday


def _is_refresh_needed_since_last_friday(
    required_files: list[str],
    *,
    candidate_dirs: list[Path],
    cutoff_friday: date,
) -> bool:
    for file_name in required_files:
        found = _first_existing_path(file_name, candidate_dirs)
        if found is None:
            return True
        if _file_is_stale_since_last_friday(found, cutoff_friday):
            return True
    return False


def _run_data_fetch_script() -> None:
    if not DATA_FETCH_SCRIPT_PATH.exists():
        raise HTTPException(status_code=500, detail='data_fetch.py is missing; cannot refresh market data')

    cmd = ['python', str(DATA_FETCH_SCRIPT_PATH)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=DATA_FETCH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f'data_fetch.py timed out after {DATA_FETCH_TIMEOUT_SECONDS}s',
        ) from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or '').strip()
        stdout = (proc.stdout or '').strip()
        message = stderr or stdout or 'unknown error'
        raise HTTPException(status_code=500, detail=f'data_fetch.py failed: {message}')


def _run_xgboost_refresh_workflow(as_of_date: date) -> None:
    try:
        dataset = coffee_data_merged.build_dataset()
        if dataset.empty:
            raise RuntimeError('Training dataset is empty')

        feature_df = coffee_xgboost_train.build_feature_dataset(dataset)
        projections, weekly_path = coffee_xgboost_train.fit_final_models_and_project(feature_df)

        history = dataset[['Date', 'coffee_c']].copy()
        history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%Y-%m-%d')
        history = history.dropna(subset=['Date', 'coffee_c'])

        history_path = OUTPUT_DIR / 'coffee_xgb_proj4_history.csv'
        weekly_path_path = OUTPUT_DIR / 'coffee_xgb_proj4_rolling_path.csv'
        projection_path = OUTPUT_DIR / 'coffee_xgb_proj4_latest_projection.csv'

        history.to_csv(history_path, index=False)
        weekly_path.to_csv(weekly_path_path, index=False)
        projections.to_csv(projection_path, index=False)

        WEBSITE_PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_path, WEBSITE_PUBLIC_DATA_DIR / history_path.name)
        shutil.copy2(weekly_path_path, WEBSITE_PUBLIC_DATA_DIR / weekly_path_path.name)
        shutil.copy2(projection_path, WEBSITE_PUBLIC_DATA_DIR / projection_path.name)

        # Keep the prediction cache synchronized with the same refresh date.
        get_prediction(requested_date=as_of_date.isoformat(), force_refresh=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'XGBoost refresh failed: {exc}') from exc


def _ensure_market_data(
    required_files: list[str],
    *,
    candidate_dirs: list[Path],
    cutoff_friday: date,
) -> None:
    if not _is_refresh_needed_since_last_friday(
        required_files,
        candidate_dirs=candidate_dirs,
        cutoff_friday=cutoff_friday,
    ):
        return

    with _REFRESH_LOCK:
        if not _is_refresh_needed_since_last_friday(
            required_files,
            candidate_dirs=candidate_dirs,
            cutoff_friday=cutoff_friday,
        ):
            return

        _run_data_fetch_script()
        _run_xgboost_refresh_workflow(cutoff_friday)

        if _is_refresh_needed_since_last_friday(
            required_files,
            candidate_dirs=candidate_dirs,
            cutoff_friday=cutoff_friday,
        ):
            raise HTTPException(
                status_code=503,
                detail='Market data refresh completed but required files are still stale or missing',
            )


def _market_cache_key(endpoint: str, cutoff_friday: date) -> str:
    return f'market-{endpoint}-{cutoff_friday.isoformat()}'


def _read_market_cache(endpoint: str, cutoff_friday: date) -> Any | None:
    cache = CacheStore()
    entry = cache.read(_market_cache_key(endpoint, cutoff_friday))
    if not isinstance(entry, dict):
        return None
    return entry.get('payload')


def _write_market_cache(endpoint: str, cutoff_friday: date, payload: Any) -> None:
    cache = CacheStore()
    cache.write(
        _market_cache_key(endpoint, cutoff_friday),
        {
            'cutoff_friday': cutoff_friday.isoformat(),
            'generated_at': datetime.now(UTC).isoformat(),
            'payload': payload,
        },
    )


def _extract_as_of_date(forecast_csv: str) -> str | None:
    lines = [line for line in forecast_csv.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header = [value.strip().lower() for value in lines[0].split(',')]
    as_of_idx = -1
    for idx, value in enumerate(header):
        if value in {'as_of_date', 'asofdate'}:
            as_of_idx = idx
            break

    if as_of_idx < 0:
        return None

    first_row = [value.strip() for value in lines[1].split(',')]
    if as_of_idx >= len(first_row):
        return None
    return first_row[as_of_idx] or None


@app.get('/', tags=['meta'])
def root() -> dict[str, str]:
    return {
        'message': 'Coffee XGBoost Prediction API is running.',
        'docs': '/docs',
        'health': '/health',
        'predict': '/predict?date=2026-04-17',
        'projected_spot': '/projected-spot',
        'contracts': '/contracts',
        'snapshot': '/snapshot',
        'news': '/news',
        'brief': '/brief',
        'market_report': '/market-report',
    }


@app.get('/health', response_model=HealthResponse, tags=['meta'])
def health() -> HealthResponse:
    cache = CacheStore()
    return HealthResponse(
        status='ok',
        storage_backend=cache.backend_name,
        has_blob_token=bool(BLOB_TOKEN),
        has_data_dir=True,  # In Vercel, data is bundled; local check not applicable
        has_logdata_dir=True,  # In Vercel, data is bundled; local check not applicable
    )


@app.get('/predict', response_model=PredictionResponse, tags=['prediction'])
def predict(
    date: str | None = Query(default=None, description='Optional as-of date in YYYY-MM-DD format.'),
    force_refresh: bool | None = Query(default=None, description='Force recomputing even if cached data exists.'),
) -> PredictionResponse:
    try:
        payload = get_prediction(requested_date=date, force_refresh=force_refresh)
        return PredictionResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f'Missing file: {exc}') from exc


@app.get('/projected-spot', response_model=ProjectedSpotCsvResponse, tags=['market-data'])
def projected_spot() -> ProjectedSpotCsvResponse:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('projected-spot', cutoff_friday)
    if isinstance(cached, dict) and 'historyCsv' in cached and 'forecastCsv' in cached:
        return ProjectedSpotCsvResponse(**cached)

    _ensure_market_data(
        ['coffee_xgb_proj4_history.csv', 'coffee_xgb_proj4_rolling_path.csv'],
        candidate_dirs=MARKET_CSV_DIRS,
        cutoff_friday=cutoff_friday,
    )

    history_path = _require_data_file(
        'coffee_xgb_proj4_history.csv',
        MARKET_CSV_DIRS,
    )
    forecast_path = _require_data_file(
        'coffee_xgb_proj4_rolling_path.csv',
        MARKET_CSV_DIRS,
    )

    history_csv = _read_text_file(history_path)
    forecast_csv = _read_text_file(forecast_path)

    payload = ProjectedSpotCsvResponse(
        files=ProjectedSpotCsvFiles(
            history=history_path.name,
            forecast=forecast_path.name,
        ),
        asOfDate=_extract_as_of_date(forecast_csv),
        historyCsv=history_csv,
        forecastCsv=forecast_csv,
    )

    _write_market_cache('projected-spot', cutoff_friday, payload.model_dump())
    return payload


@app.get('/contracts', response_model=list[ContractRow], tags=['market-data'])
def contracts() -> list[ContractRow]:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('contracts', cutoff_friday)
    if isinstance(cached, list):
        return [ContractRow(**row) for row in cached]

    _ensure_market_data(
        ['contracts.json'],
        candidate_dirs=MARKET_JSON_DIRS,
        cutoff_friday=cutoff_friday,
    )

    path = _require_data_file('contracts.json', MARKET_JSON_DIRS)
    data = _read_json_file(path)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail='contracts.json must contain a JSON array')
    _write_market_cache('contracts', cutoff_friday, data)
    return [ContractRow(**row) for row in data]


@app.get('/snapshot', response_model=SnapshotResponse, tags=['market-data'])
def snapshot() -> SnapshotResponse:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('snapshot', cutoff_friday)
    if isinstance(cached, dict):
        return SnapshotResponse(**cached)

    _ensure_market_data(
        ['snapshot.json'],
        candidate_dirs=MARKET_JSON_DIRS,
        cutoff_friday=cutoff_friday,
    )

    path = _require_data_file('snapshot.json', MARKET_JSON_DIRS)
    data = _read_json_file(path)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail='snapshot.json must contain a JSON object')
    _write_market_cache('snapshot', cutoff_friday, data)
    return SnapshotResponse(**data)


@app.get('/news', response_model=list[NewsItem], tags=['market-data'])
def news(limit: int = Query(default=3, ge=1, le=20)) -> list[NewsItem]:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('news', cutoff_friday)
    if isinstance(cached, list):
        return [NewsItem(**row) for row in cached[:limit]]

    _ensure_market_data(
        ['news.json'],
        candidate_dirs=MARKET_JSON_DIRS,
        cutoff_friday=cutoff_friday,
    )

    path = _require_data_file('news.json', MARKET_JSON_DIRS)
    data = _read_json_file(path)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail='news.json must contain a JSON array')
    _write_market_cache('news', cutoff_friday, data)
    return [NewsItem(**row) for row in data[:limit]]


@app.get('/brief', response_model=BriefResponse, tags=['market-data'])
def brief() -> BriefResponse:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('brief', cutoff_friday)
    if isinstance(cached, dict):
        return BriefResponse(**cached)

    _ensure_market_data(
        ['roaster_brief.json'],
        candidate_dirs=MARKET_JSON_DIRS,
        cutoff_friday=cutoff_friday,
    )

    path = _require_data_file('roaster_brief.json', MARKET_JSON_DIRS)
    data = _read_json_file(path)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail='roaster_brief.json must contain a JSON object')
    _write_market_cache('brief', cutoff_friday, data)
    return BriefResponse(**data)


@app.get('/market-report', tags=['market-data'])
def market_report() -> dict[str, object]:
    cutoff_friday = _last_friday(datetime.now(UTC).date())

    cached = _read_market_cache('market-report', cutoff_friday)
    if isinstance(cached, dict):
        return cached

    _ensure_market_data(
        ['latest_market_report.json'],
        candidate_dirs=MARKET_JSON_DIRS,
        cutoff_friday=cutoff_friday,
    )

    path = _require_data_file('latest_market_report.json', MARKET_JSON_DIRS)
    data = _read_json_file(path)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail='latest_market_report.json must contain a JSON object')
    _write_market_cache('market-report', cutoff_friday, data)
    return data
