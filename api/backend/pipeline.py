from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import DATA_DIR, DEFAULT_FORCE_ON_REFRESH_DAY, LOGDATA_DIR, OUTPUT_DIR, REFRESH_WEEKDAY
from backend.storage.cache import CacheStore
from backend.ml import coffee_data_merged, coffee_xgboost_train


class PipelineError(RuntimeError):
    pass


def _validate_input_dirs() -> None:
    if not DATA_DIR.exists() or not LOGDATA_DIR.exists():
        raise PipelineError(
            'Expected ./data and ./logdata folders with CSV files. '
            'In Vercel, ensure data files are in api/data and api/logdata directories and committed to git, '
            'so add those source files before deploying.'
        )


def _resolve_request_date(requested_date: str | None) -> date:
    if requested_date:
        return date.fromisoformat(requested_date)
    return datetime.now(UTC).date()


def _is_refresh_day(d: date) -> bool:
    return d.weekday() == REFRESH_WEEKDAY


def _normalize_projection_rows(projections: pd.DataFrame) -> list[dict[str, Any]]:
    rows = projections.copy()
    for col in rows.columns:
        if pd.api.types.is_datetime64_any_dtype(rows[col]):
            rows[col] = rows[col].dt.strftime('%Y-%m-%d')
    return rows.to_dict(orient='records')


def _build_fresh_prediction(as_of_date: date) -> dict[str, Any]:
    _validate_input_dirs()

    dataset = coffee_data_merged.build_dataset()
    dataset = dataset.loc[pd.to_datetime(dataset['Date']).dt.date <= as_of_date].copy()
    if dataset.empty:
        raise PipelineError(f'No rows available on or before {as_of_date.isoformat()}.')

    feature_df = coffee_xgboost_train.build_feature_dataset(dataset)
    projections, weekly_path = coffee_xgboost_train.fit_final_models_and_project(feature_df)

    latest_row = dataset.sort_values('Date').iloc[-1]

    outputs_dir = OUTPUT_DIR / as_of_date.isoformat()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    projections.to_csv(outputs_dir / 'projection.csv', index=False)
    weekly_path.to_csv(outputs_dir / 'weekly_path.csv', index=False)

    return {
        'requested_date': as_of_date.isoformat(),
        'resolved_as_of_date': pd.to_datetime(latest_row['Date']).date().isoformat(),
        'cache_hit': False,
        'refresh_reason': 'fresh_compute',
        'generated_at': datetime.now(UTC).isoformat(),
        'storage_backend': '',
        'source': 'xgboost_training_script',
        'latest_price': float(latest_row['coffee_c']) if pd.notna(latest_row['coffee_c']) else None,
        'projections': _normalize_projection_rows(projections),
        'metadata': {
            'dataset_rows_used': int(len(dataset)),
            'feature_rows_used': int(len(feature_df)),
            'weekly_path_rows': int(len(weekly_path)),
        },
    }


def get_prediction(requested_date: str | None, force_refresh: bool | None = None) -> dict[str, Any]:
    cache = CacheStore()
    req_date = _resolve_request_date(requested_date)
    key = req_date.isoformat()

    if force_refresh is None:
        force_refresh = DEFAULT_FORCE_ON_REFRESH_DAY and _is_refresh_day(req_date)

    exact = cache.read(key)
    if exact and not force_refresh:
        exact['cache_hit'] = True
        exact['refresh_reason'] = 'exact_cache_hit'
        exact['storage_backend'] = cache.backend_name
        return exact

    if not _is_refresh_day(req_date) and not force_refresh:
        historical = cache.latest_on_or_before(key)
        if historical:
            historical['cache_hit'] = True
            historical['refresh_reason'] = 'historical_cache_hit'
            historical['storage_backend'] = cache.backend_name
            return historical

    payload = _build_fresh_prediction(req_date)
    write_info = cache.write(key, payload)
    payload['storage_backend'] = write_info['storage_backend']
    if 'url' in write_info:
        payload['metadata']['blob_url'] = write_info['url']
    if 'path' in write_info:
        payload['metadata']['cache_path'] = write_info['path']
    return payload
