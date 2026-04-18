from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    requested_date: str
    resolved_as_of_date: str
    cache_hit: bool
    refresh_reason: str
    generated_at: str
    storage_backend: str
    source: str
    latest_price: float | None = None
    projections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    storage_backend: str
    has_blob_token: bool
    has_data_dir: bool
    has_logdata_dir: bool
