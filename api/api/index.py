from flask import Flask, jsonify, request, Response

from api.runner import run_local_script

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "message": "Flask running on Vercel (api/index.py)"
    })

@app.route("/api/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "world")
    return jsonify({
        "message": f"Hello, {name}!"
    })

@app.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({
        "you_sent": data
    })


@app.route("/api/projected-spot", methods=["GET"])
def projected_spot():
    should_run_script = request.args.get("run", "false").lower() in {"1", "true", "yes"}
    run_result = None

    if should_run_script:
        run_result = run_local_script("scripts/projection_stub.py")
        if not run_result["ok"]:
            return jsonify(
                {
                    "error": "Projection stub script failed",
                    "detail": run_result,
                }
            ), 500

    history_csv = "date,price\n2026-04-01,0.0\n"
    forecast_csv = (
        "as_of_date,step_week,date,predicted_weekly_log_return,projected_price,"
        "anchor_weekly_log_return,raw_1w_log_return\n"
        "2026-04-01,1,2026-04-08,0.0,0.0,0.0,0.0\n"
    )

    if request.args.get("format") == "csv":
        return Response(forecast_csv, mimetype="text/csv")

    return jsonify(
        {
            "format": "projected-spot-csv.v1",
            "files": {
                "history": "inline",
                "forecast": "inline",
            },
            "asOfDate": "2026-04-01",
            "historyCsv": history_csv,
            "forecastCsv": forecast_csv,
            "scriptRun": run_result,
        }
    )


@app.route("/api/contracts", methods=["GET"])
def contracts():
    should_run_script = request.args.get("run", "false").lower() in {"1", "true", "yes"}
    run_result = None

    if should_run_script:
        run_result = run_local_script("scripts/contracts_stub.py")
        if not run_result["ok"]:
            return jsonify(
                {
                    "error": "Contracts stub script failed",
                    "detail": run_result,
                }
            ), 500

    rows = [
        {
            "symbol": "KCK26",
            "expiry_date": "2026-05-19",
            "last_price": 0.0,
            "price_change": 0.0,
            "price_change_pct": 0.0,
            "volume": 0,
            "open_interest": 0,
            "captured_at": "2026-04-01T00:00:00Z",
        }
    ]

    response = jsonify(rows)
    if run_result is not None:
        response.headers["X-Stub-Script-Status"] = "ok"
    return response