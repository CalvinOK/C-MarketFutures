import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, Response

from api.runner import run_local_script

app = Flask(__name__)

API_DIR = Path(__file__).resolve().parent
API_ROOT = API_DIR.parent
PROJECT_ROOT = API_ROOT.parent
RUNTIME_DATA_DIR = Path(os.getenv("RUNTIME_DATA_DIR", "/tmp/coffee-market-data"))

JSON_DATA_DIRS = [
    RUNTIME_DATA_DIR,
    API_ROOT / "public" / "data",
    API_ROOT / "data",
    PROJECT_ROOT / "website" / "public" / "data",
    PROJECT_ROOT / "old" / "api 2" / "data",
]

CSV_DATA_DIRS = [
    RUNTIME_DATA_DIR,
    API_ROOT / "public" / "data",
    API_ROOT / "outputs",
    PROJECT_ROOT / "website" / "public" / "data",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "old" / "api 2" / "outputs",
]

MARKET_CACHE: dict[str, dict] = {}
DEFAULT_CONTRACTS_SCRIPT = "barchart_scraper/scraper.py"
DEFAULT_PROJECTION_SCRIPT = "scripts/run_old_projection_pipeline.py"

_FRESHNESS_THRESHOLDS: dict[str, timedelta] = {
    "contracts":      timedelta(hours=1),
    "snapshot":       timedelta(hours=1),
    "news":           timedelta(days=2),
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

@app.route("/")
def root():
    return jsonify({
        "message": "Coffee market API running on Flask",
        "endpoints": [
            "/api/hello",
            "/api/projected-spot",
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

    stale = False
    if history_path and _file_is_stale_since_last_friday(history_path, cutoff_friday):
        stale = True
    if forecast_path and _file_is_stale_since_last_friday(forecast_path, cutoff_friday):
        stale = True

    needs_refresh = run_refresh or history_path is None or forecast_path is None or stale

    if needs_refresh and script:
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            if run_refresh:
                return jsonify({"error": "Projection refresh script failed", "detail": refresh_result}), 500

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
            payload["_freshness"]["pipeline_error"] = refresh_result.get("stderr", "")

    _write_cached("projected-spot", cutoff_friday, payload)
    return jsonify(payload)


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
        return jsonify({"error": "Contracts refresh script failed", "detail": refresh_result}), 500

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
    snapshot_path = _first_existing_path("snapshot.json", JSON_DATA_DIRS)
    is_stale = bool(snapshot_path and _file_is_stale_since_last_friday(snapshot_path, cutoff_friday))
    needs_refresh = run_refresh or snapshot_path is None or is_stale

    cached = _read_cached("snapshot", cutoff_friday)
    if isinstance(cached, dict) and not needs_refresh:
        return jsonify(cached)

    if needs_refresh and script:
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return jsonify({"error": "Snapshot refresh script failed", "detail": refresh_result}), 500

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
    if isinstance(cached, dict) and "data" in cached and not run_refresh and not file_too_old:
        return jsonify(cached)

    if run_refresh:
        script = request.args.get(
            "script",
            _get_contracts_script_path(),
        )
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return jsonify({"error": "News refresh script failed", "detail": refresh_result}), 500

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
    if isinstance(cached, dict) and not run_refresh:
        return jsonify(cached)

    if run_refresh:
        script = request.args.get(
            "script",
            _get_contracts_script_path(),
        )
        refresh_result = _maybe_run_refresh_script(script)
        if refresh_result and not refresh_result.get("ok", False):
            return jsonify({"error": "Brief refresh script failed", "detail": refresh_result}), 500

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