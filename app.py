"""Minimal WSGI app so Vercel can detect a Python entrypoint."""

import json
from datetime import datetime, timezone


def app(environ, start_response):
    payload = {
        "status": "ok",
        "service": "coffee-market-futures",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload).encode("utf-8")

    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]
