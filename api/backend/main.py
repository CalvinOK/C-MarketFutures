from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from backend.config import BLOB_TOKEN, DATA_DIR, LOGDATA_DIR
from backend.pipeline import PipelineError, get_prediction
from backend.schemas import HealthResponse, PredictionResponse
from backend.storage.cache import CacheStore

app = FastAPI(
    title='Coffee XGBoost Prediction API',
    version='1.0.0',
    description='FastAPI app for Vercel that runs your coffee data pipeline, builds projections, and caches results online.',
)


@app.get('/', tags=['meta'])
def root() -> dict[str, str]:
    return {
        'message': 'Coffee XGBoost Prediction API is running.',
        'docs': '/docs',
        'health': '/health',
        'predict': '/predict?date=2026-04-17',
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
