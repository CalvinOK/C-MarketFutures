#!/usr/bin/env python3
"""Sucafina market report scraper.

Fetches the latest Sucafina market report from sucafina.com/na/lp/market-report,
extracts the most recent weekly entry, summarizes it via Gemini API,
and writes roaster_brief.json and latest_market_report.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

SUCAFINA_URL = "https://sucafina.com/na/lp/market-report"

DEFAULT_OUTPUT_DIR = (
    Path(os.getenv("RUNTIME_DATA_DIR"))
    if os.getenv("RUNTIME_DATA_DIR")
    else ROOT / "data"
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


for env_path in [ROOT / ".env", ROOT / ".env.local", PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
    _load_dotenv(env_path)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


def fetch_sucafina_page(timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(SUCAFINA_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_most_recent_report(html: str) -> tuple[str, str]:
    """Returns (date_str, report_text) for the most recent entry."""
    soup = BeautifulSoup(html, "html.parser")
    h2s = soup.find_all("h2")
    if not h2s:
        raise ValueError("No h2 elements found on Sucafina page")

    first_h2 = h2s[0]
    date_str = first_h2.get_text(strip=True)

    content_parts = []
    el = first_h2.find_next_sibling()
    stop_at = h2s[1] if len(h2s) > 1 else None
    while el and el != stop_at:
        text = el.get_text(separator=" ", strip=True)
        if text:
            content_parts.append(text)
        el = el.find_next_sibling()

    report_text = " ".join(content_parts).strip()
    if not report_text:
        raise ValueError(f"No content found under date heading: {date_str}")

    return date_str, report_text


def parse_date(date_str: str) -> str:
    """Convert '22 April 2026' to 'YYYY-MM-DD'."""
    parts = date_str.strip().split()
    if len(parts) == 3:
        day, month_name, year = parts
        month = MONTH_MAP.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"
    return date_str


def summarize_with_gemini(date_str: str, report_text: str) -> dict:
    """Call Gemini REST API to extract structured summary."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    prompt = f"""You are analyzing a Sucafina coffee market report dated {date_str}.

Report text:
{report_text}

Extract and return a JSON object with these exact fields:
- "headline": string — a concise headline for the report (e.g. "Coffee Market Brief — {date_str}")
- "market_bias": string — one of: "Bullish", "Bearish", "Neutral", "Neutral / Tactically Cautious", "Cautiously Bullish", "Cautiously Bearish"
- "key_takeaways": array of 3 strings — the 3 most important points for a coffee roaster
- "risks": array of 3-4 strings — key risks mentioned
- "roaster_actions": array of 3-5 strings — actionable recommendations for roasters
- "confidence": string — "High", "Medium", or "Low"

Return ONLY valid JSON, no markdown fences."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()

    result = resp.json()
    text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())

    return json.loads(text)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    now_utc = datetime.now(UTC).isoformat()

    try:
        print("Fetching Sucafina market report page...", flush=True)
        html = fetch_sucafina_page()

        date_str, report_text = parse_most_recent_report(html)
        print(f"Parsed report dated: {date_str}", flush=True)

        date_iso = parse_date(date_str)

        summary = summarize_with_gemini(date_str, report_text)

        roaster_brief = {
            "headline": summary.get("headline", f"Coffee Market Brief — {date_str}"),
            "source_report": f"Sucafina Market Report — {date_str}",
            "report_url": SUCAFINA_URL,
            "market_bias": summary.get("market_bias", "Neutral"),
            "key_takeaways": summary.get("key_takeaways", []),
            "risks": summary.get("risks", []),
            "roaster_actions": summary.get("roaster_actions", []),
            "confidence": summary.get("confidence", "Medium"),
            "raw_report": report_text,
            "generated_at": now_utc,
        }

        market_report = {
            "date": date_iso,
            "display_date": date_str,
            "title": f"Sucafina Market Report — {date_str}",
            "content": report_text,
            "source_url": SUCAFINA_URL,
            "fetched_at": now_utc,
        }

        atomic_write_json(output_dir / "roaster_brief.json", roaster_brief)
        atomic_write_json(output_dir / "latest_market_report.json", market_report)

        print(f"Written roaster_brief.json and latest_market_report.json to {output_dir}", flush=True)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
