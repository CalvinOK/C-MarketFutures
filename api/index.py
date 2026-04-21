from __future__ import annotations

import csv
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIRS = [
    BASE_DIR / 'public' / 'data',
    PROJECT_ROOT / 'website' / 'public' / 'data',
    PROJECT_ROOT / 'old' / 'api' / 'data',
]

OUTPUT_DIRS = [
    BASE_DIR / 'outputs',
    PROJECT_ROOT / 'website' / 'public' / 'data',
    PROJECT_ROOT / 'old' / 'api' / 'outputs',
]

CACHE_DIR = BASE_DIR / 'outputs' / 'cache'
LOGDATA_DIRS = [
    BASE_DIR / 'logdata',
    PROJECT_ROOT / 'old' / 'api' / 'logdata',
]

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing_path(file_name: str, directories: list[Path]) -> Path | None:
    for directory in directories:
        candidate = directory / file_name
        if candidate.is_file():
            return candidate
    return None


def _read_text(file_name: str, directories: list[Path]) -> str:
    path = _first_existing_path(file_name, directories)
    if path is None:
        raise FileNotFoundError(file_name)
    return path.read_text(encoding='utf-8')


def _read_json(file_name: str, directories: list[Path]) -> Any:
    return json.loads(_read_text(file_name, directories))


def _parse_bool(value: str | None, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _parse_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(UTC).date()


def _is_refresh_day(value: date) -> bool:
    return value.weekday() == 4


def _last_friday(value: date) -> date:
    return value - timedelta(days=(value.weekday() - 4) % 7)


def _json_response(payload: Any, status_code: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        status=status_code,
        mimetype='application/json; charset=utf-8',
    )


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f'{key}.json'


def _read_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _write_cache(key: str, payload: dict[str, Any]) -> None:
    _cache_path(key).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _latest_cache_on_or_before(requested_date: str) -> dict[str, Any] | None:
    candidates: list[tuple[str, Path]] = []
    for directory in [CACHE_DIR, PROJECT_ROOT / 'old' / 'api' / 'outputs' / 'cache']:
        if not directory.is_dir():
            continue
        for path in directory.glob('*.json'):
            if path.stem <= requested_date:
                candidates.append((path.stem, path))
    if not candidates:
        return None
    _, best_path = sorted(candidates)[-1]
    try:
        payload = json.loads(best_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_market_data(file_name: str) -> Any:
    return _read_json(file_name, DATA_DIRS)


def _load_projection_rows() -> list[dict[str, Any]]:
    path = _first_existing_path('coffee_xgb_proj4_latest_projection.csv', OUTPUT_DIRS)
    if path is None:
        path = _first_existing_path('coffee_spot_projection_6m.csv', OUTPUT_DIRS)
    if path is None:
        raise FileNotFoundError('coffee_xgb_proj4_latest_projection.csv')

    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    normalized: list[dict[str, Any]] = []
    if path.name == 'coffee_xgb_proj4_latest_projection.csv':
        for row in rows:
            normalized.append(
                {
                    'as_of_date': row.get('as_of_date'),
                    'horizon_weeks': int(float(row['horizon_weeks'])),
                    'current_price': float(row['current_price']),
                    'predicted_log_change': float(row['predicted_log_change']),
                    'projected_price': float(row['projected_price']),
                    'n_features_used': int(float(row['n_features_used'])),
                }
            )
    else:
        for row in rows:
            normalized.append(
                {
                    'date': row.get('date'),
                    'forecast': float(row['forecast']),
                    'lower_95': float(row['lower_95']),
                    'upper_95': float(row['upper_95']),
                    'step': int(float(row['step'])),
                }
            )
    return normalized


def _load_history_csv() -> tuple[str, str | None]:
    path = _first_existing_path('coffee_xgb_proj4_history.csv', OUTPUT_DIRS)
    if path is not None:
        text = path.read_text(encoding='utf-8')
        as_of_date = None
        rows = text.splitlines()
        if len(rows) > 1:
            header = [value.strip().lower() for value in rows[0].split(',')]
            if 'date' in header:
                date_idx = header.index('date')
                first_row = rows[1].split(',')
                if date_idx < len(first_row):
                    as_of_date = first_row[date_idx] or None
        return text, as_of_date

    merged_path = _first_existing_path('coffee_model_dataset_merged.csv', OUTPUT_DIRS)
    if merged_path is None:
        raise FileNotFoundError('coffee_xgb_proj4_history.csv')

    lines = merged_path.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise FileNotFoundError('coffee_xgb_proj4_history.csv')

    reader = csv.DictReader(lines)
    output_lines = ['Date,coffee_c']
    first_date: str | None = None
    for row in reader:
        date_value = row.get('Date') or row.get('date')
        coffee_value = row.get('coffee_c') or row.get('coffee_c ') or row.get('coffee')
        if not date_value or not coffee_value:
            continue
        if first_date is None:
            first_date = date_value
        output_lines.append(f'{date_value},{coffee_value}')
    return '\n'.join(output_lines), first_date


def _load_rolling_path_csv() -> tuple[str, str | None]:
    path = _first_existing_path('coffee_xgb_proj4_rolling_path.csv', OUTPUT_DIRS)
    if path is None:
        raise FileNotFoundError('coffee_xgb_proj4_rolling_path.csv')
    text = path.read_text(encoding='utf-8')
    as_of_date = None
    rows = text.splitlines()
    if len(rows) > 1:
        header = [value.strip().lower() for value in rows[0].split(',')]
        if 'as_of_date' in header:
            as_of_idx = header.index('as_of_date')
            first_row = rows[1].split(',')
            if as_of_idx < len(first_row):
                as_of_date = first_row[as_of_idx] or None
    return text, as_of_date


def _load_prediction_artifact(requested_date: date) -> dict[str, Any]:
    projections = _load_projection_rows()
    history_csv, history_as_of = _load_history_csv()
    weekly_path_csv, weekly_as_of = _load_rolling_path_csv()

    resolved_as_of = history_as_of or weekly_as_of
    latest_price: float | None = None
    if projections:
        first_projection = projections[0]
        latest_price = first_projection.get('current_price') or first_projection.get('forecast')
        if resolved_as_of is None:
            resolved_as_of = first_projection.get('as_of_date') or first_projection.get('date')

    payload: dict[str, Any] = {
        'requested_date': requested_date.isoformat(),
        'resolved_as_of_date': resolved_as_of or requested_date.isoformat(),
        'cache_hit': False,
        'refresh_reason': 'fresh_compute',
        'generated_at': datetime.now(UTC).isoformat(),
        'storage_backend': 'local_ephemeral',
        'source': 'xgboost_training_script',
        'latest_price': latest_price,
        'projections': projections,
        'metadata': {
            'artifact_source': 'checked-in csv outputs',
            'history_csv_rows': max(len(history_csv.splitlines()) - 1, 0),
            'weekly_path_rows': max(len(weekly_path_csv.splitlines()) - 1, 0),
        },
    }
    return payload


def _health_payload() -> dict[str, Any]:
    return {
        'status': 'ok',
        'storage_backend': 'local_ephemeral',
        'has_blob_token': bool(os.getenv('BLOB_READ_WRITE_TOKEN')),
        'has_data_dir': any(directory.is_dir() for directory in DATA_DIRS),
        'has_logdata_dir': any(directory.is_dir() for directory in LOGDATA_DIRS),
    }


def _market_cache_key(endpoint: str, cutoff_friday: date) -> str:
    return f'{endpoint}-{cutoff_friday.isoformat()}'


@app.get('/')
def root() -> Response:
    return _json_response(
        {
            'message': 'Coffee market API is running.',
            'health': '/health',
            'predict': '/predict?date=2026-04-17',
            'projected_spot': '/projected-spot',
            'contracts': '/contracts',
            'snapshot': '/snapshot',
            'news': '/news',
            'brief': '/brief',
            'market_report': '/market-report',
        }
    )


@app.get('/health')
def health() -> Response:
    return _json_response(_health_payload())


@app.get('/predict')
def predict() -> Response:
    try:
        requested_date = _parse_date(request.args.get('date'))
    except ValueError:
        return _json_response({'error': 'date must be in YYYY-MM-DD format'}, status_code=400)
    force_refresh = _parse_bool(request.args.get('force_refresh'))
    cache_key = requested_date.isoformat()

    exact = _read_cache(cache_key)
    if exact and not force_refresh:
        exact['cache_hit'] = True
        exact['refresh_reason'] = 'exact_cache_hit'
        exact['storage_backend'] = 'local_ephemeral'
        return _json_response(exact)

    if not _is_refresh_day(requested_date) and not force_refresh:
        historical = _latest_cache_on_or_before(cache_key)
        if historical:
            historical['cache_hit'] = True
            historical['refresh_reason'] = 'historical_cache_hit'
            historical['storage_backend'] = 'local_ephemeral'
            return _json_response(historical)

    try:
        payload = _load_prediction_artifact(requested_date)
    except FileNotFoundError as exc:
        return _json_response({'error': f'missing artifact: {exc.args[0]}'}, status_code=503)

    _write_cache(cache_key, payload)
    return _json_response(payload)


@app.get('/projected-spot')
def projected_spot() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('projected-spot', cutoff_friday)
    cached = _read_cache(cache_key)
    if isinstance(cached, dict) and 'historyCsv' in cached and 'forecastCsv' in cached:
        return _json_response(cached)

    try:
        history_csv, _ = _load_history_csv()
        forecast_csv, forecast_as_of = _load_rolling_path_csv()
    except FileNotFoundError as exc:
        return _json_response({'error': f'missing artifact: {exc.args[0]}'}, status_code=503)

    as_of_date = None
    forecast_lines = [line for line in forecast_csv.splitlines() if line.strip()]
    if len(forecast_lines) > 1:
        header = [value.strip().lower() for value in forecast_lines[0].split(',')]
        if 'as_of_date' in header:
            as_of_idx = header.index('as_of_date')
            first_row = forecast_lines[1].split(',')
            if as_of_idx < len(first_row):
                as_of_date = first_row[as_of_idx] or None
    as_of_date = as_of_date or forecast_as_of

    payload = {
        'format': 'projected-spot-csv.v1',
        'files': {
            'history': 'coffee_xgb_proj4_history.csv',
            'forecast': 'coffee_xgb_proj4_rolling_path.csv',
        },
        'asOfDate': as_of_date,
        'historyCsv': history_csv,
        'forecastCsv': forecast_csv,
    }
    _write_cache(cache_key, payload)
    return _json_response(payload)


@app.get('/contracts')
def contracts() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('contracts', cutoff_friday)
    cached = _read_cache(cache_key)
    if isinstance(cached, list):
        return _json_response(cached)

    payload = _load_market_data('contracts.json')
    if not isinstance(payload, list):
        return _json_response({'error': 'contracts.json must contain a JSON array'}, status_code=500)

    _write_cache(cache_key, payload)
    return _json_response(payload)


@app.get('/snapshot')
def snapshot() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('snapshot', cutoff_friday)
    cached = _read_cache(cache_key)
    if isinstance(cached, dict):
        return _json_response(cached)

    payload = _load_market_data('snapshot.json')
    if not isinstance(payload, dict):
        return _json_response({'error': 'snapshot.json must contain a JSON object'}, status_code=500)

    _write_cache(cache_key, payload)
    return _json_response(payload)


@app.get('/news')
def news() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('news', cutoff_friday)
    cached = _read_cache(cache_key)
    limit = request.args.get('limit', default=3, type=int)
    if not isinstance(limit, int):
        limit = 3
    if cached and isinstance(cached, list):
        return _json_response(cached[: max(1, min(limit, 20))])

    payload = _load_market_data('news.json')
    if not isinstance(payload, list):
        return _json_response({'error': 'news.json must contain a JSON array'}, status_code=500)

    _write_cache(cache_key, payload)
    return _json_response(payload[: max(1, min(limit, 20))])


@app.get('/brief')
def brief() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('brief', cutoff_friday)
    cached = _read_cache(cache_key)
    if isinstance(cached, dict):
        return _json_response(cached)

    payload = _load_market_data('roaster_brief.json')
    if not isinstance(payload, dict):
        return _json_response({'error': 'roaster_brief.json must contain a JSON object'}, status_code=500)

    _write_cache(cache_key, payload)
    return _json_response(payload)


@app.get('/market-report')
def market_report() -> Response:
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cache_key = _market_cache_key('market-report', cutoff_friday)
    cached = _read_cache(cache_key)
    if isinstance(cached, dict):
        return _json_response(cached)

    payload = _load_market_data('latest_market_report.json')
    if not isinstance(payload, dict):
        return _json_response({'error': 'latest_market_report.json must contain a JSON object'}, status_code=500)

    _write_cache(cache_key, payload)
    return _json_response(payload)
