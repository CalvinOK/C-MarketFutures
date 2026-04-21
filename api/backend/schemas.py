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


class ContractRow(BaseModel):
    symbol: str
    expiry_date: str | None = None
    last_price: float
    price_change: float = 0.0
    price_change_pct: float = 0.0
    volume: int = 0
    open_interest: int = 0
    captured_at: str | None = None


class SnapshotResponse(BaseModel):
    frontPrice: float
    curveShape: str
    totalVolume: int
    totalOpenInterest: int
    frontSymbol: str | None = None
    asOf: str | None = None


class NewsItem(BaseModel):
    category: str
    text: str
    source: str
    url: str | None = None
    timestamp: str


class BriefResponse(BaseModel):
    headline: str
    source_report: str
    report_url: str
    market_bias: str
    key_takeaways: list[str] = Field(default_factory=list)
    generated_at: str | None = None


class ProjectedSpotCsvFiles(BaseModel):
    history: str
    forecast: str


class ProjectedSpotCsvResponse(BaseModel):
    format: str = 'projected-spot-csv.v1'
    files: ProjectedSpotCsvFiles
    asOfDate: str | None = None
    historyCsv: str
    forecastCsv: str
