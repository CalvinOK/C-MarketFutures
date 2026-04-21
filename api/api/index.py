from __future__ import annotations

from flask import Flask, jsonify, request

from .coffee_service import (
    build_health_payload,
    build_market_refresh_payload,
    build_root_payload,
    load_brief_payload,
    load_contracts_payload,
    load_market_report_payload,
    load_news_payload,
    load_projected_spot_payload,
    load_snapshot_payload,
    require_internal_token_if_configured,
)

app = Flask(__name__)


def dual_route(rule: str, **options):
    def decorator(view_func):
        app.route(rule, **options)(view_func)
        if rule == "/":
            app.route("/api", **options)(view_func)
        else:
            app.route(f"/api{rule}", **options)(view_func)
        return view_func

    return decorator


@dual_route("/")
def root():
    return jsonify(build_root_payload())


@dual_route("/health")
def health():
    return jsonify(build_health_payload())


@dual_route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "world")
    return jsonify({"message": f"Hello, {name}!"})


@dual_route("/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({"you_sent": data})


@dual_route("/contracts", methods=["GET"])
def contracts():
    return jsonify(load_contracts_payload(refresh=request.args.get("refresh") == "1"))


@dual_route("/snapshot", methods=["GET"])
def snapshot():
    return jsonify(load_snapshot_payload(refresh=request.args.get("refresh") == "1"))


@dual_route("/news", methods=["GET"])
def news():
    limit = request.args.get("limit", "3")
    try:
        parsed_limit = max(1, min(20, int(limit)))
    except ValueError:
        parsed_limit = 3
    return jsonify(load_news_payload(limit=parsed_limit, refresh=request.args.get("refresh") == "1"))


@dual_route("/brief", methods=["GET"])
def brief():
    return jsonify(load_brief_payload(refresh=request.args.get("refresh") == "1"))


@dual_route("/market-report", methods=["GET"])
def market_report():
    return jsonify(load_market_report_payload(refresh=request.args.get("refresh") == "1"))


@dual_route("/projected-spot", methods=["GET"])
def projected_spot():
    return jsonify(load_projected_spot_payload(refresh=request.args.get("refresh") == "1"))


@dual_route("/refresh", methods=["POST", "GET"])
def refresh():
    auth_error = require_internal_token_if_configured(request)
    if auth_error is not None:
        return auth_error

    return jsonify(build_market_refresh_payload())