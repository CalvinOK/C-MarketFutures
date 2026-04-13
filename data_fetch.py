#!/usr/bin/env python3
"""
data_fetch_sucafina_combined.py — Fetches coffee market data, the latest
Sucafina market report, and coffee-market news, then writes a combined
roaster-friendly brief.

What this script does:
- Fetches coffee-market news from NewsAPI
- Optionally summarizes top news with Gemini
- Fetches nearby coffee contract data from DataBento or AlphaVantage fallback
- Scrapes the latest Sucafina market report from:
  https://sucafina.com/na/lp/market-report
- Produces a combined roaster brief using both the latest report and news
- Writes JSON files to web/public/data/ and data/

Usage:
    python data_fetch_sucafina_combined.py

Required env vars:
    NEWS_API_KEY
    GEMINI_API_KEY

Optional:
    DATABENTO_API_KEY
    ALPHAVANTAGE_API_KEY
    GEMINI_MODEL           (default: gemini-2.5-flash-lite)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


# ── Load .env ──────────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader that does not overwrite existing environment vars."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


ROOT = Path(__file__).parent
_load_dotenv(ROOT / ".env")

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

WEB_PUBLIC_DATA = ROOT / "web" / "public" / "data"
DATA_DIR = ROOT / "data"
WEB_PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUCAFINA_MARKET_REPORT_URL = "https://sucafina.com/na/lp/market-report"

KC_EXPIRY_MAP: dict[str, str] = {
    "KCH26": "2026-03-20",
    "KCK26": "2026-05-19",
    "KCN26": "2026-07-20",
    "KCU26": "2026-09-18",
    "KCZ26": "2026-12-18",
    "KCH27": "2027-03-19",
    "KCK27": "2027-05-18",
}

NEWS_QUERIES = [
    '"coffee futures"',
    '"Brazil weather" coffee',
    '"ICE coffee" OR arabica futures',
    'coffee (frost OR drought OR rainfall)',
    'coffee (export OR shipment OR logistics)',
]
NEWS_LOOKBACK_WINDOWS_DAYS = [30, 90, 365]

COFFEE_DIRECT_TERMS = (
    "coffee futures",
    "arabica futures",
    "robusta futures",
    "ice coffee",
    "coffee contract",
    "coffee beans",
    "green coffee",
    "c-market",
    "c market",
    "nybot",
)

COFFEE_SUPPORT_TERMS = (
    "coffee",
    "arabica",
    "robusta",
    "futures",
    "contract",
    "spread",
    "curve",
    "hedge",
    "hedging",
    "cot",
    "inventory",
    "shipment",
    "logistics",
    "export",
    "trade",
    "price",
    "weather",
    "frost",
    "drought",
    "rainfall",
    "crop",
    "harvest",
    "yield",
    "production",
    "plantation",
    "supply",
    "demand",
    "brazil",
    "vietnam",
    "colombia",
    "ethiopia",
    "honduras",
    "peru",
    "uganda",
    "kenya",
    "indonesia",
    "mexico",
    "sao paulo",
    "minas gerais",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_headline(title: str) -> str:
    normalized = "".join(c for c in title.lower() if c.isalnum() or c == " ").strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _limit_words(text: str, max_words: int = 22) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _parse_iso_datetime(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _write_json(data: object, *paths: Path) -> None:
    text = json.dumps(data, indent=2, default=str)
    for path in paths:
        path.write_text(text, encoding="utf-8")
        print(f"[out] Wrote {path.relative_to(ROOT)}")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _extract_first_number(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


# ── News ──────────────────────────────────────────────────────────────────────

def _is_relevant_coffee_news(title: str, description: str | None) -> bool:
    text = f"{title} {description or ''}".lower()
    if any(term in text for term in COFFEE_DIRECT_TERMS):
        return True
    if "coffee" in text or "arabica" in text or "robusta" in text:
        return any(term in text for term in COFFEE_SUPPORT_TERMS)
    return False


def _fetch_news_for_window(days_back: int, seen: set[str]) -> list[dict[str, Any]]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    window_articles: list[dict[str, Any]] = []

    for query in NEWS_QUERIES:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "apiKey": NEWS_API_KEY,
                },
                timeout=12,
            )
            resp.raise_for_status()
            for article in resp.json().get("articles", []):
                title = article.get("title", "")
                if not title or title == "[Removed]":
                    continue
                description = article.get("description")
                if not _is_relevant_coffee_news(title, description):
                    continue
                headline_hash = _hash_headline(title)
                if headline_hash in seen:
                    continue
                seen.add(headline_hash)
                window_articles.append(
                    {
                        "title": title,
                        "description": description,
                        "url": article.get("url", ""),
                        "publishedAt": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                    }
                )
        except Exception as exc:
            print(f'[news] Query "{query}" failed for {days_back}d window: {exc}', file=sys.stderr)

    window_articles.sort(key=lambda a: _parse_iso_datetime(a.get("publishedAt", "")), reverse=True)
    return window_articles


def fetch_news() -> list[dict[str, Any]]:
    """Fetch raw deduplicated coffee-market articles from NewsAPI."""
    if not NEWS_API_KEY:
        print("[news] NEWS_API_KEY not set — skipping", file=sys.stderr)
        return []

    articles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for days_back in NEWS_LOOKBACK_WINDOWS_DAYS:
        articles.extend(_fetch_news_for_window(days_back, seen))
        if len(articles) >= 3:
            break
    articles.sort(key=lambda a: _parse_iso_datetime(a.get("publishedAt", "")), reverse=True)
    return articles


def process_with_gemini(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize top coffee news with Gemini. Falls back to article titles."""
    if not articles:
        return []

    if not GEMINI_API_KEY:
        return [
            {
                "category": "Market Brief",
                "text": _limit_words(article["title"], 18),
                "source": article["source"],
                "url": article["url"],
                "timestamp": article["publishedAt"],
            }
            for article in articles[:3]
        ]

    batch = articles[:3]
    article_list = "\n\n".join(
        f'[{i}] {article["title"]}\n{article.get("description") or ""}'
        for i, article in enumerate(batch)
    )

    system_instruction = (
        "You are a commodity markets analyst. Process only news articles clearly related to coffee, "
        "coffee futures, or coffee agriculture markets. If an article is not clearly relevant, omit it. "
        "For each article return a JSON array. Each element must have: "
        '"index": the article [N] number (integer), '
        '"summary": one precise sentence, max 22 words, factual, based only on the article text, and '
        '"category": exactly one of "Market Brief", "Commodities Desk", or "Trade Note". '
        "Return ONLY a valid JSON array."
    )

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
            json={
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "responseMimeType": "application/json",
                },
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"Process these coffee market articles:\n\n{article_list}"}],
                    }
                ],
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw_text = payload["candidates"][0]["content"]["parts"][0]["text"]
        first, last = raw_text.find("["), raw_text.rfind("]")
        if first == -1 or last == -1:
            raise ValueError("No JSON array in Gemini response")
        parsed = json.loads(raw_text[first : last + 1])

        valid_categories = {"Market Brief", "Commodities Desk", "Trade Note"}
        results: list[dict[str, Any]] = []
        for item in parsed:
            idx = item.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                continue
            if item.get("category") not in valid_categories:
                continue
            article = batch[idx]
            results.append(
                {
                    "category": item["category"],
                    "text": _limit_words(str(item["summary"]), 22),
                    "source": article["source"],
                    "url": article["url"],
                    "timestamp": article["publishedAt"],
                }
            )
        return results[:3]
    except Exception as exc:
        print(f"[gemini] News processing failed: {exc}", file=sys.stderr)
        return [
            {
                "category": "Market Brief",
                "text": _limit_words(article["title"], 18),
                "source": article["source"],
                "url": article["url"],
                "timestamp": article["publishedAt"],
            }
            for article in articles[:3]
        ]


# ── Contracts ─────────────────────────────────────────────────────────────────

def _active_symbols() -> list[str]:
    now = datetime.now(timezone.utc)
    return sorted(
        [
            symbol
            for symbol, expiry in KC_EXPIRY_MAP.items()
            if datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc) > now
        ],
        key=lambda symbol: KC_EXPIRY_MAP[symbol],
    )


def _fetch_databento() -> list[dict[str, Any]]:
    if not DATABENTO_API_KEY:
        raise ValueError("DATABENTO_API_KEY not set")

    now = datetime.now(timezone.utc)
    auth = base64.b64encode(f"{DATABENTO_API_KEY}:".encode()).decode()
    resp = requests.post(
        "https://hist.databento.com/v0/timeseries.get_range",
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
        json={
            "dataset": "IFUS.IMPACT",
            "symbols": ["KC.c.0", "KC.c.1", "KC.c.2", "KC.c.3"],
            "schema": "ohlcv-1d",
            "start": (now - timedelta(days=2)).isoformat(),
            "end": now.isoformat(),
            "stype_in": "continuous",
            "encoding": "json",
        },
        timeout=15,
    )
    resp.raise_for_status()
    records: list[dict[str, Any]] = resp.json()

    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        symbol = record["symbol"]
        if symbol not in latest or record["ts_event"] > latest[symbol]["ts_event"]:
            latest[symbol] = record

    active = _active_symbols()
    results: list[dict[str, Any]] = []
    for i, bar in enumerate(latest.values()):
        if i >= len(active):
            break
        symbol = active[i]
        last_price = bar["close"] / 100
        price_change = (bar["close"] - bar["open"]) / 100
        results.append(
            {
                "symbol": symbol,
                "expiry_date": KC_EXPIRY_MAP[symbol],
                "last_price": round(last_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change / last_price if last_price else 0, 6),
                "volume": bar.get("volume", 0),
                "open_interest": 0,
                "captured_at": now.isoformat(),
            }
        )
    return results


def _fetch_alphavantage() -> list[dict[str, Any]]:
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY not set")

    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "COFFEE", "interval": "monthly", "apikey": ALPHAVANTAGE_API_KEY},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    if "Information" in data:
        raise ValueError(f'AlphaVantage rate limit: {data["Information"]}')
    rows = data.get("data", [])
    if not rows:
        raise ValueError("AlphaVantage returned empty data")

    front_price = float(rows[0]["value"])
    active = _active_symbols()
    now = datetime.now(timezone.utc)
    return [
        {
            "symbol": symbol,
            "expiry_date": KC_EXPIRY_MAP[symbol],
            "last_price": round(front_price * (1 + 0.005 * i), 2),
            "price_change": 0.0,
            "price_change_pct": 0.0,
            "volume": 0,
            "open_interest": 0,
            "captured_at": now.isoformat(),
        }
        for i, symbol in enumerate(active[:4])
    ]


def fetch_contracts() -> list[dict[str, Any]]:
    try:
        contracts = _fetch_databento()
        if contracts:
            return contracts
    except Exception as exc:
        print(f"[contracts] DataBento failed: {exc}", file=sys.stderr)

    try:
        return _fetch_alphavantage()
    except Exception as exc:
        print(f"[contracts] AlphaVantage failed: {exc}", file=sys.stderr)
        return []


def compute_snapshot(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {}
    front = contracts[0]
    deferred = contracts[-1]
    return {
        "frontPrice": front["last_price"],
        "curveShape": "Contango" if deferred["last_price"] > front["last_price"] else "Backwardation",
        "totalVolume": sum(contract["volume"] for contract in contracts),
        "totalOpenInterest": sum(contract["open_interest"] for contract in contracts),
        "frontSymbol": front["symbol"],
        "asOf": front["captured_at"],
    }


# ── Sucafina market report scraping ───────────────────────────────────────────

def _is_date_heading(text: str) -> bool:
    text = text.strip()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            pass
    return False


def _parse_date_heading(text: str) -> str:
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date heading: {text}")


def scrape_latest_sucafina_report(url: str = SUCAFINA_MARKET_REPORT_URL) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; coffee-market-bot/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.body or soup
    blocks = main.find_all(["h2", "h3", "p"])

    reports: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for tag in blocks:
        text = _normalize_whitespace(tag.get_text(" ", strip=True))
        if not text:
            continue

        if tag.name in {"h2", "h3"} and _is_date_heading(text):
            if current and current.get("paragraphs"):
                reports.append(current)
            current = {"date": _parse_date_heading(text), "display_date": text, "paragraphs": []}
            continue

        if current and tag.name == "p":
            current["paragraphs"].append(text)

    if current and current.get("paragraphs"):
        reports.append(current)

    if not reports:
        raise RuntimeError("Could not find any dated reports on the Sucafina market report page")

    reports.sort(key=lambda report: report["date"], reverse=True)
    latest = reports[0]
    text = "\n\n".join(latest.pop("paragraphs"))
    return {
        "date": latest["date"],
        "display_date": latest["display_date"],
        "title": f"Sucafina Market Report — {latest['display_date']}",
        "content": text,
        "source_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Roaster summary / synthesis ───────────────────────────────────────────────

def _infer_bias(report_text: str, news_items: list[dict[str, Any]]) -> str:
    text = (report_text + " " + " ".join(item.get("text", "") for item in news_items)).lower()
    bearish_score = 0
    bullish_score = 0

    bearish_terms = [
        "lower",
        "sell-off",
        "pressure",
        "break below",
        "negative for prices",
        "liquidation",
        "risk-off",
        "support broken",
        "weakness",
    ]
    bullish_terms = [
        "tight supply",
        "frost",
        "drought",
        "smaller crop",
        "bullish",
        "higher prices",
        "production risk",
        "shortfall",
        "firm differentials",
    ]

    bearish_score += sum(term in text for term in bearish_terms)
    bullish_score += sum(term in text for term in bullish_terms)

    if bearish_score > bullish_score:
        return "Bearish / Downside Risk"
    if bullish_score > bearish_score:
        return "Bullish / Upside Risk"
    return "Mixed / Volatile"


def build_roaster_brief(
    report: dict[str, Any],
    news_items: list[dict[str, Any]],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the latest Sucafina report and processed news into one actionable brief."""
    report_text = report.get("content", "")
    report_title = report.get("title", "Latest Sucafina Market Report")
    report_date = report.get("display_date") or report.get("date", "")
    combined_text = (report_text + " " + " ".join(item.get("text", "") for item in news_items)).lower()

    end_price = _extract_first_number(r"end at\s+(\d+(?:\.\d+)?)\s*(?:us cents|usc|usd)?", report_text)
    support = _extract_first_number(r"below a critical\s+(\d+(?:\.\d+)?)\s*usc?/lb support", report_text)
    downside_target = _extract_first_number(r"test resistance at\s*\$?(\d+(?:\.\d+)?)\/lb", report_text)
    if downside_target is None:
        downside_target = _extract_first_number(r"test\s+\$?(\d+(?:\.\d+)?)\/lb", report_text)

    key_takeaways: list[str] = []
    if end_price:
        key_takeaways.append(f"The latest Sucafina report says the market recently finished around {end_price} USc/lb.")
    if support:
        key_takeaways.append(f"The board has broken below roughly {support} USc/lb support, which keeps short-term downside pressure in play.")
    if downside_target:
        key_takeaways.append(f"The report points to a possible near-term test around ${downside_target}/lb.")
    if not key_takeaways:
        key_takeaways.append("The latest report points to a cautious market tone with headline-driven volatility.")

    risks: list[str] = []
    if any(term in combined_text for term in ["iran", "war", "conflict", "geopolitical"]):
        risks.append("Geopolitical headlines could trigger broad commodity volatility and fast intraday swings in coffee.")
    if "risk-off" in combined_text or "liquidation" in combined_text or "macro" in combined_text:
        risks.append("Macro risk-off flows may drive coffee lower even if coffee-specific fundamentals do not worsen.")
    if "support" in combined_text or "break below" in combined_text:
        risks.append("A break of support can attract technical selling and extend downside before value buyers step in.")
    if any(term in combined_text for term in ["brazil", "frost", "drought", "rainfall", "weather"]):
        risks.append("Weather in Brazil remains a key upside volatility risk, especially if frost or dryness headlines appear.")
    if any(term in combined_text for term in ["shipment", "logistics", "export", "vietnam"]):
        risks.append("Export or logistics disruptions can lift replacement costs even if futures soften.")

    actions: list[str] = []
    if downside_target or "negative for prices" in combined_text or "sell-off" in combined_text:
        actions.append("Avoid fixing large volumes all at once; scale coverage in tranches while the board is probing lower.")
    actions.append("Separate flat-price decisions from differential decisions, because physical replacement costs can stay firm even when futures weaken.")
    if support:
        actions.append("Set buy triggers around key technical levels so you can respond quickly if the market flushes lower or rebounds sharply.")
    if any(term in combined_text for term in ["frost", "drought", "weather", "brazil"]):
        actions.append("Keep nearby weather alerts on your radar, since weather scares can reverse a bearish market quickly.")
    if snapshot and snapshot.get("frontPrice"):
        actions.append(f"Use the current front-month snapshot of {snapshot['frontPrice']} with your own margin targets before extending forward cover.")

    headline = _infer_bias(report_text, news_items)
    confidence = "Medium"
    if report_text and news_items:
        confidence = "Medium-High"
    elif report_text or news_items:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "headline": f"Coffee Market Brief — {report_date}",
        "source_report": report_title,
        "report_url": report.get("source_url", ""),
        "market_bias": headline,
        "key_takeaways": _dedupe_keep_order(key_takeaways),
        "news_drivers": news_items,
        "risks": _dedupe_keep_order(risks),
        "roaster_actions": _dedupe_keep_order(actions),
        "contract_snapshot": snapshot or {},
        "raw_report": report_text,
        "confidence": confidence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("[data_fetch] Starting...")

    raw_articles = fetch_news()
    print(f"[news] Fetched {len(raw_articles)} raw articles")
    news = process_with_gemini(raw_articles)
    print(f"[news] Processed {len(news)} articles")
    _write_json(news, WEB_PUBLIC_DATA / "news.json", DATA_DIR / "news.json")

    contracts = fetch_contracts()
    print(f"[contracts] Fetched {len(contracts)} contracts")
    _write_json(contracts, WEB_PUBLIC_DATA / "contracts.json", DATA_DIR / "contracts.json")

    snapshot = compute_snapshot(contracts)
    _write_json(snapshot, WEB_PUBLIC_DATA / "snapshot.json", DATA_DIR / "snapshot.json")

    try:
        latest_report = scrape_latest_sucafina_report()
        _write_json(
            latest_report,
            WEB_PUBLIC_DATA / "latest_market_report.json",
            DATA_DIR / "latest_market_report.json",
        )

        roaster_brief = build_roaster_brief(latest_report, news, snapshot)
        _write_json(
            roaster_brief,
            WEB_PUBLIC_DATA / "roaster_brief.json",
            DATA_DIR / "roaster_brief.json",
        )
    except Exception as exc:
        print(f"[sucafina] Failed to fetch report: {exc}", file=sys.stderr)

    print("[data_fetch] Done.")


if __name__ == "__main__":
    main()