import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, Response
import requests

from .runner import run_local_script

app = Flask(__name__)

API_DIR = Path(__file__).resolve().parent
API_ROOT = API_DIR.parent
PROJECT_ROOT = API_ROOT.parent
RUNTIME_DATA_DIR = Path(os.getenv("RUNTIME_DATA_DIR", "/tmp/coffee-market-data"))

JSON_DATA_DIRS = [
    RUNTIME_DATA_DIR,
]

CSV_DATA_DIRS = [
    RUNTIME_DATA_DIR,
    API_ROOT / "outputs",
    PROJECT_ROOT / "outputs",
]

MARKET_CACHE: dict[str, dict] = {}
FORECAST_CACHE: dict[str, dict] = {}
DEFAULT_CONTRACTS_SCRIPT = "barchart_scraper/scraper.py"
DEFAULT_PROJECTION_SCRIPT = "scripts/run_old_projection_pipeline.py"
DEFAULT_NEWS_SCRIPT = "scripts/news_scraper.py"
DEFAULT_BRIEF_SCRIPT = "scripts/sucafina_scraper.py"

_FRESHNESS_THRESHOLDS: dict[str, timedelta] = {
    "contracts":      timedelta(hours=1),
    "snapshot":       timedelta(hours=1),
    "news":           timedelta(days=1),
    "brief":          timedelta(days=7),
    "projected-spot": timedelta(days=7),
}


def _check_freshness(endpoint: str, timestamp_str: str | None) -> dict:
    if not timestamp_str:
        return {"stale": True, "reason": "no_timestamp"}
    threshold = _FRESHNESS_THRESHOLDS.get(endpoint)
    if not threshold:
        return {"stale": False}
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age = datetime.now(UTC) - ts
        return {
            "stale": age > threshold,
            "age_hours": round(age.total_seconds() / 3600, 1),
            "threshold_hours": threshold.total_seconds() / 3600,
        }
    except (ValueError, TypeError):
        return {"stale": True, "reason": "unparseable_timestamp"}


def _freshness_error(endpoint: str, freshness: dict):
    return jsonify(
        {
            "error": f"{endpoint} data is stale",
            "_freshness": freshness,
        }
    ), 503


def _refresh_error(endpoint: str, refresh_result: dict):
    return jsonify(
        {
            "error": f"{endpoint} refresh failed",
            "detail": refresh_result,
        }
    ), 503


def _last_friday(d: date) -> date:
    return d - timedelta(days=(d.weekday() - 4) % 7)


def _first_existing_path(file_name: str, candidate_dirs: list[Path]) -> Path | None:
    for directory in candidate_dirs:
        candidate = directory / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _cache_key(endpoint: str, cutoff_friday: date) -> str:
    return f"{endpoint}:{cutoff_friday.isoformat()}"


def _read_cached(endpoint: str, cutoff_friday: date):
    return MARKET_CACHE.get(_cache_key(endpoint, cutoff_friday))


def _write_cached(endpoint: str, cutoff_friday: date, payload):
    MARKET_CACHE[_cache_key(endpoint, cutoff_friday)] = payload


def _read_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise FileNotFoundError(f"File is empty: {path.name}")
    return text


def _read_json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_as_of_date(forecast_csv: str) -> str | None:
    lines = [line for line in forecast_csv.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header = [v.strip().lower() for v in lines[0].split(",")]
    try:
        idx = header.index("as_of_date")
    except ValueError:
        return None

    first_row = [v.strip() for v in lines[1].split(",")]
    if idx >= len(first_row):
        return None
    return first_row[idx] or None


def _file_is_stale_since_last_friday(path: Path, cutoff_friday: date) -> bool:
    """File-mtime check — appropriate for JSON files written fresh each run."""
    mtime_date = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    return mtime_date < cutoff_friday


def _forecast_is_stale(forecast_path: Path | None, cutoff_friday: date) -> bool:
    """Check staleness using the as_of_date embedded in the CSV, not file mtime.

    File mtime is unreliable: the pipeline can write a file after the last
    Friday while the actual model data ends weeks earlier (e.g. ran April 22
    but logdata ends April 2). The as_of_date column reflects the true last
    training date, so we compare that against the most recent Friday.
    """
    if forecast_path is None:
        return True
    try:
        text = forecast_path.read_text(encoding="utf-8")
        as_of_str = _extract_as_of_date(text)
        if not as_of_str:
            return True
        as_of = date.fromisoformat(as_of_str[:10])
        return as_of < cutoff_friday
    except Exception:
        return True


def _maybe_run_refresh_script(script_path: str | None):
    if not script_path:
        return None
    timeout_seconds = int(os.getenv("REFRESH_SCRIPT_TIMEOUT_SECONDS", "1800"))
    return run_local_script(script_path, timeout_seconds=timeout_seconds)


def _require_file(file_name: str, candidate_dirs: list[Path]) -> Path:
    path = _first_existing_path(file_name, candidate_dirs)
    if path is None:
        raise FileNotFoundError(file_name)
    return path


def _get_contracts_script_path() -> str:
    return os.getenv("CONTRACTS_SCRIPT", DEFAULT_CONTRACTS_SCRIPT)


def _market_api_auth_error():
    expected = os.getenv("MARKET_API_TOKEN", "").strip()
    if not expected:
        return None
    provided = request.headers.get("Authorization", "")
    if provided != f"Bearer {expected}":
        return jsonify({"error": "Unauthorized"}), 401
    return None

@app.route("/")
def root():
    return jsonify({
        "message": "Coffee market API running on Flask",
        "endpoints": [
            "/api/hello",
            "/api/projected-spot",
            "/api/coffee/history/latest.csv",
            "/api/coffee/forecast",
            "/api/contracts",
            "/api/snapshot",
            "/api/news",
            "/api/brief",
        ],
    })

@app.route("/api/hello", methods=["GET"])
@app.route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "world")
    return jsonify({
        "message": f"Hello, {name}!"
    })

@app.route("/api/echo", methods=["POST"])
@app.route("/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({
        "you_sent": data
    })


@app.route("/api/projected-spot", methods=["GET"])
@app.route("/projected-spot", methods=["GET"])
def projected_spot():
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    cached = _read_cached("projected-spot", cutoff_friday)
    if isinstance(cached, dict):
        freshness = cached.get("_freshness", {})
        if not freshness.get("stale", True):
            if request.args.get("format") == "csv":
                return Response(cached["forecastCsv"], mimetype="text/csv")
            return jsonify(cached)

    refresh_result = None
    run_refresh = request.args.get("run", "false").lower() in {"1", "true", "yes"}
    script = request.args.get(
        "script",
        os.getenv("PROJECTION_SCRIPT", DEFAULT_PROJECTION_SCRIPT),
    )

    history_path = _first_existing_path("coffee_xgb_proj4_history.csv", CSV_DATA_DIRS)
    forecast_path = _first_existing_path("coffee_xgb_proj4_rolling_path.csv", CSV_DATA_DIRS)

    stale = _forecast_is_stale(forecast_path, cutoff_friday)
    needs_refresh = run_refresh or history_path is None or forecast_path is None or stale

    if needs_refresh and script:
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return _refresh_error("Projection", refresh_result)

    try:
        history_path = _require_file("coffee_xgb_proj4_history.csv", CSV_DATA_DIRS)
        forecast_path = _require_file("coffee_xgb_proj4_rolling_path.csv", CSV_DATA_DIRS)
        history_csv = _read_text_file(history_path)
        forecast_csv = _read_text_file(forecast_path)
    except FileNotFoundError as exc:
        detail = {"pipelineRun": refresh_result} if refresh_result else {}
        return jsonify({"error": f"Missing required CSV: {exc}", **detail}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid CSV file encoding"}), 500

    if request.args.get("format") == "csv":
        return Response(forecast_csv, mimetype="text/csv")

    as_of_date = _extract_as_of_date(forecast_csv)
    payload = {
        "format": "projected-spot-csv.v1",
        "files": {
            "history": history_path.name,
            "forecast": forecast_path.name,
        },
        "asOfDate": as_of_date,
        "historyCsv": history_csv,
        "forecastCsv": forecast_csv,
        "_freshness": _check_freshness("projected-spot", as_of_date),
    }
    if refresh_result is not None:
        payload["scriptRun"] = refresh_result
        if not refresh_result.get("ok"):
            return _refresh_error("Projection", refresh_result)

    if payload["_freshness"].get("stale", True):
        return _freshness_error("Projection", payload["_freshness"])

    _write_cached("projected-spot", cutoff_friday, payload)
    return jsonify(payload)


@app.route("/api/coffee/history/latest.csv", methods=["GET"])
@app.route("/coffee/history/latest.csv", methods=["GET"])
def latest_coffee_history_csv():
    """Return recent Coffee C daily OHLCV rows from the existing Databento path."""
    auth_error = _market_api_auth_error()
    if auth_error:
        return auth_error
    try:
        from scripts.fetch_logdata import build_logdata_csv, fetch_databento_history

        end = datetime.now(UTC).date() - timedelta(days=1)
        start = end - timedelta(days=10)
        raw = fetch_databento_history(
            "coffee",
            "IFUS.IMPACT",
            "KC.c.0",
            "ohlcv-1d",
            start,
            end,
            stype_in="continuous",
        )
        csv_data = build_logdata_csv(raw).to_csv(index=False)
        return Response(csv_data, mimetype="text/csv")
    except Exception as exc:
        print(f"[coffee-history-latest] provider failure: {type(exc).__name__}: {exc}")
        return jsonify({"error": "Unable to fetch latest Coffee C daily data", "code": "provider_data_error"}), 502


def _fetch_supabase_history_for_forecast() -> list[dict]:
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY", "")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required for forecast")

    rows: list[dict] = []
    for offset in range(0, 5000, 1000):
        response = requests.get(
            f'{supabase_url}/rest/v1/Coffee%20C%20Historical%20Data',
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            params={"select": '"Date","Price"', "limit": 1000, "offset": offset},
            timeout=20,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError("Supabase forecast history response was not an array")
        rows.extend(page)
        if len(page) < 1000:
            break
    return rows


@app.route("/api/coffee/forecast", methods=["GET"])
@app.route("/coffee/forecast", methods=["GET"])
def coffee_forecast():
    auth_error = _market_api_auth_error()
    if auth_error:
        return auth_error
    try:
        from backend.ml.coffee_daily_forecast import train_direct_forecast

        history = _fetch_supabase_history_for_forecast()
        if not history:
            return jsonify({"error": "Coffee C history is empty", "code": "insufficient_history"}), 422
        result = train_direct_forecast(history)
        payload = {
            "format": "coffee-daily-forecast.v1",
            "asOf": result.as_of,
            "currentPrice": result.current_price,
            "unit": result.unit,
            "forecast": result.forecast,
            "validation": result.validation,
            "featureCount": result.feature_count,
            "rowsUsed": result.rows_used,
        }
        FORECAST_CACHE.clear()
        FORECAST_CACHE[result.as_of] = payload
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "invalid_or_insufficient_history"}), 422
    except requests.RequestException as exc:
        print(f"[coffee-forecast] database failure: {type(exc).__name__}: {exc}")
        return jsonify({"error": "Unable to retrieve Coffee C history", "code": "database_error"}), 502
    except ImportError as exc:
        print(f"[coffee-forecast] dependency failure: {exc}")
        return jsonify({"error": "Forecast dependencies are unavailable", "code": "dependency_error"}), 503
    except Exception as exc:
        print(f"[coffee-forecast] model failure: {type(exc).__name__}: {exc}")
        return jsonify({"error": "Coffee forecast training failed", "code": "model_error"}), 500


@app.route("/api/contracts", methods=["GET"])
@app.route("/contracts", methods=["GET"])
def contracts():
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    run_refresh = request.args.get("run", "false").lower() in {"1", "true", "yes"}

    script = request.args.get(
        "script",
        _get_contracts_script_path(),
    )
    refresh_result = None

    contracts_path = _first_existing_path("contracts.json", JSON_DATA_DIRS)
    is_stale = bool(contracts_path and _file_is_stale_since_last_friday(contracts_path, cutoff_friday))
    needs_refresh = run_refresh or contracts_path is None or is_stale

    cached = _read_cached("contracts", cutoff_friday)
    if isinstance(cached, dict) and "data" in cached and not needs_refresh:
        return jsonify(cached)

    if needs_refresh and script:
        refresh_result = _maybe_run_refresh_script(script)

    if refresh_result and not refresh_result.get("ok", False):
        return _refresh_error("Contracts", refresh_result)

    try:
        contracts_path = _require_file("contracts.json", JSON_DATA_DIRS)
        rows = _read_json_file(contracts_path)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing required JSON: {exc}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in contracts.json"}), 500

    if not isinstance(rows, list):
        return jsonify({"error": "contracts.json must contain a JSON array"}), 500

    latest_ts = max((r.get("captured_at", "") for r in rows), default=None)
    payload = {"data": rows, "_freshness": _check_freshness("contracts", latest_ts)}
    if refresh_result and not refresh_result.get("ok", False):
        return _refresh_error("Contracts", refresh_result)

    if payload["_freshness"].get("stale", True):
        return _freshness_error("Contracts", payload["_freshness"])

    _write_cached("contracts", cutoff_friday, payload)
    response = jsonify(payload)
    if refresh_result is not None:
        response.headers["X-Script-Run"] = "ok"
    return response


@app.route("/api/snapshot", methods=["GET"])
@app.route("/snapshot", methods=["GET"])
def snapshot():
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    run_refresh = request.args.get("run", "false").lower() in {"1", "true", "yes"}
    script = request.args.get(
        "script",
        _get_contracts_script_path(),
    )
    refresh_result = None
    snapshot_path = _first_existing_path("snapshot.json", JSON_DATA_DIRS)
    is_stale = bool(snapshot_path and _file_is_stale_since_last_friday(snapshot_path, cutoff_friday))
    needs_refresh = run_refresh or snapshot_path is None or is_stale

    cached = _read_cached("snapshot", cutoff_friday)
    if isinstance(cached, dict) and not needs_refresh:
        return jsonify(cached)

    if needs_refresh and script:
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return _refresh_error("Snapshot", refresh_result)

    try:
        path = _require_file("snapshot.json", JSON_DATA_DIRS)
        payload = _read_json_file(path)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing required JSON: {exc}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in snapshot.json"}), 500

    if not isinstance(payload, dict):
        return jsonify({"error": "snapshot.json must contain a JSON object"}), 500

    payload["_freshness"] = _check_freshness("snapshot", payload.get("asOf"))
    if refresh_result and not refresh_result.get("ok", False):
        return _refresh_error("Snapshot", refresh_result)
    if payload["_freshness"].get("stale", True):
        return _freshness_error("Snapshot", payload["_freshness"])
    _write_cached("snapshot", cutoff_friday, payload)
    return jsonify(payload)


@app.route("/api/news", methods=["GET"])
@app.route("/news", methods=["GET"])
def news():
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    run_refresh = request.args.get("run", "false").lower() in {"1", "true", "yes"}

    news_path = _first_existing_path("news.json", JSON_DATA_DIRS)
    # News goes stale faster than the weekly cache key — check actual file age.
    file_too_old = bool(
        news_path and
        (datetime.now(UTC) - datetime.fromtimestamp(news_path.stat().st_mtime, tz=UTC))
        > _FRESHNESS_THRESHOLDS["news"]
    )

    cached = _read_cached("news", cutoff_friday)
    if (
        isinstance(cached, dict)
        and "data" in cached
        and not cached.get("_freshness", {}).get("stale", True)
        and not run_refresh
        and not file_too_old
    ):
        return jsonify(cached)

    needs_refresh = run_refresh or news_path is None or file_too_old
    if needs_refresh:
        script = request.args.get("script", os.getenv("NEWS_SCRIPT", DEFAULT_NEWS_SCRIPT))
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return _refresh_error("News", refresh_result)

    try:
        path = _require_file("news.json", JSON_DATA_DIRS)
        items = _read_json_file(path)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing required JSON: {exc}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in news.json"}), 500

    if not isinstance(items, list):
        return jsonify({"error": "news.json must contain a JSON array"}), 500

    limit = request.args.get("limit", default=3, type=int)
    if limit is None or limit < 1:
        limit = 3
    items = items[: min(limit, 20)]

    latest_ts = max((item.get("timestamp", "") for item in items), default=None)
    payload = {"data": items, "_freshness": _check_freshness("news", latest_ts)}

    _write_cached("news", cutoff_friday, payload)
    return jsonify(payload)


@app.route("/api/brief", methods=["GET"])
@app.route("/brief", methods=["GET"])
def brief():
    cutoff_friday = _last_friday(datetime.now(UTC).date())
    run_refresh = request.args.get("run", "false").lower() in {"1", "true", "yes"}
    cached = _read_cached("brief", cutoff_friday)
    if (
        isinstance(cached, dict)
        and not cached.get("_freshness", {}).get("stale", True)
        and not run_refresh
    ):
        return jsonify(cached)

    brief_path = _first_existing_path("roaster_brief.json", JSON_DATA_DIRS)
    brief_stale = bool(
        brief_path and
        (datetime.now(UTC) - datetime.fromtimestamp(brief_path.stat().st_mtime, tz=UTC))
        > _FRESHNESS_THRESHOLDS["brief"]
    )
    needs_brief_refresh = run_refresh or brief_path is None or brief_stale

    if needs_brief_refresh:
        script = request.args.get("script", os.getenv("BRIEF_SCRIPT", DEFAULT_BRIEF_SCRIPT))
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return _refresh_error("Brief", refresh_result)

    try:
        path = _require_file("roaster_brief.json", JSON_DATA_DIRS)
        payload = _read_json_file(path)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing required JSON: {exc}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in roaster_brief.json"}), 500

    if not isinstance(payload, dict):
        return jsonify({"error": "roaster_brief.json must contain a JSON object"}), 500

    payload["_freshness"] = _check_freshness("brief", payload.get("generated_at"))
    if payload["_freshness"].get("stale", True):
        return _freshness_error("Brief", payload["_freshness"])

    # Inject current snapshot so callers always get live prices alongside the narrative.
    snapshot_path = _first_existing_path("snapshot.json", JSON_DATA_DIRS)
    if snapshot_path:
        try:
            live_snap = _read_json_file(snapshot_path)
            if isinstance(live_snap, dict):
                payload["_live_snapshot"] = live_snap
        except (json.JSONDecodeError, OSError):
            pass

    _write_cached("brief", cutoff_friday, payload)
    return jsonify(payload)


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "coffee-market-api",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "jsonSearchDirs": [str(path) for path in JSON_DATA_DIRS],
            "csvSearchDirs": [str(path) for path in CSV_DATA_DIRS],
        }
    )