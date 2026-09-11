import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request
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

MARKET_CACHE: dict[str, dict] = {}
DEFAULT_CONTRACTS_SCRIPT = "barchart_scraper/scraper.py"
DEFAULT_NEWS_SCRIPT = "scripts/news_scraper.py"
DEFAULT_BRIEF_SCRIPT = "scripts/sucafina_scraper.py"

_FRESHNESS_THRESHOLDS: dict[str, timedelta] = {
    "contracts":      timedelta(hours=1),
    "snapshot":       timedelta(hours=1),
    "news":           timedelta(days=1),
    "brief":          timedelta(days=7),
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


def _read_json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_is_stale_since_last_friday(path: Path, cutoff_friday: date) -> bool:
    """File-mtime check — appropriate for JSON files written fresh each run."""
    mtime_date = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    return mtime_date < cutoff_friday


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


def _fetch_supabase_history_for_forecast() -> list[dict]:
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY", "")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required for forecast")

    rows: list[dict] = []
    for offset in range(0, 10000, 1000):
        response = requests.get(
            f'{supabase_url}/rest/v1/Coffee%20C%20Historical%20Data',
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            params={"select": '\"Date\",\"Price\"', "limit": 1000, "offset": offset},
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
@app.route("/api/projected-spot", methods=["GET"])
@app.route("/projected-spot", methods=["GET"])
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
    auth_error = _market_api_auth_error()
    if auth_error:
        return auth_error
    try:
        from databento_contracts import fetch_contracts
        return jsonify(fetch_contracts())
    except Exception as exc:
        print(f"[contracts] Databento failure: {type(exc).__name__}: {exc}")
        return jsonify({"error": "Unable to retrieve Coffee C contracts", "code": "contracts_provider_error"}), 502

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
        }
    )