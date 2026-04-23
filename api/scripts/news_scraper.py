#!/usr/bin/env python3
"""Coffee market news scraper.

Fetches recent coffee market news from NewsAPI and writes news.json.
Falls back to scraping a public RSS feed if NewsAPI is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

DEFAULT_OUTPUT_DIR = (
    Path(os.getenv("RUNTIME_DATA_DIR"))
    if os.getenv("RUNTIME_DATA_DIR")
    else ROOT / "data"
)

NEWS_API_BASE = "https://newsapi.org/v2/everything"
RSS_FEEDS = [
    ("Google News", "https://news.google.com/rss/search?q=coffee+futures+arabica+market&hl=en-US&gl=US&ceid=US:en"),
    ("Google News", "https://news.google.com/rss/search?q=coffee+commodity+price+export&hl=en-US&gl=US&ceid=US:en"),
    ("Reuters Commodities", "https://feeds.reuters.com/reuters/businessNews"),
]

COFFEE_KEYWORDS = ["coffee", "arabica", "robusta", "ICE coffee", "KC futures", "C-market", "caffeine market"]
MAX_ITEMS = 5
MAX_AGE_DAYS = 7


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

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")


def _is_coffee_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in COFFEE_KEYWORDS)


def _classify_category(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["futures", "contract", "hedge", "cbot", "ice", "c-market", "arabica price", "robusta price"]):
        return "Commodities Desk"
    if any(w in text_lower for w in ["brazil", "colombia", "ethiopia", "vietnam", "kenya", "origin", "harvest", "crop"]):
        return "Origin Watch"
    if any(w in text_lower for w in ["roaster", "retail", "consumer", "café", "cafe", "barista", "specialty"]):
        return "Trade & Retail"
    return "Market Brief"


def _decode_html(text: str) -> str:
    """Decode HTML entities."""
    import html
    return html.unescape(text)


MARKET_KEYWORDS = [
    "futures", "arabica", "robusta", "coffee price", "coffee market",
    "ICE coffee", "C-market", "coffee trade", "coffee export", "coffee import",
    "coffee supply", "coffee demand", "coffee harvest", "coffee crop",
    "coffee bean", "origin", "Brazil coffee", "Colombia coffee", "Ethiopia coffee",
    "Vietnam coffee", "Indonesia coffee", "coffee output", "coffee production",
]


def fetch_from_newsapi(api_key: str) -> list[dict]:
    """Fetch coffee market news from NewsAPI."""
    from_date = (datetime.now(UTC) - timedelta(days=MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    params = {
        "q": "coffee AND (market OR price OR futures OR arabica OR robusta OR export OR supply OR crop OR harvest OR trade)",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 30,
        "from": from_date,
        "apiKey": api_key,
    }
    resp = requests.get(NEWS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Score articles: prefer market/commodity mentions over purely consumer ones
    scored: list[tuple[int, dict]] = []
    for article in data.get("articles", []):
        title = _decode_html((article.get("title") or "").strip())
        description = _decode_html((article.get("description") or "").strip())
        url = (article.get("url") or "").strip()
        published_at = (article.get("publishedAt") or "").strip()
        source_name = (article.get("source") or {}).get("name", "Unknown")

        if not title or not url:
            continue

        combined = f"{title} {description}".lower()

        score = sum(1 for kw in MARKET_KEYWORDS if kw.lower() in combined)

        # Slight penalty for purely chain-retail articles with no market signal
        consumer_only = ["starbucks reward", "dunkin freebie", "coffee shop opening", "barista competition", "coffee recipe"]
        if any(w in combined for w in consumer_only) and score == 0:
            continue

        text = description if description else title
        scored.append((score, {
            "category": _classify_category(combined),
            "text": text[:280],
            "source": source_name,
            "url": url,
            "timestamp": published_at,
        }))

    # Sort highest score first, take top MAX_ITEMS
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:MAX_ITEMS]]


def _clean_text(text: str) -> str:
    """Strip HTML tags, decode entities, collapse whitespace."""
    import re
    import html
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_publisher_suffix(title: str, publisher: str) -> str:
    """Remove ' - Publisher Name' suffix from a Google News title."""
    if publisher and title.endswith(f" - {publisher}"):
        return title[: -(len(publisher) + 3)].strip()
    # Fallback: strip last ' - ...' segment
    import re
    match = re.search(r"\s+-\s+[^-]{3,60}$", title)
    if match:
        return title[: match.start()].strip()
    return title


def fetch_from_rss(feed_url: str, source_name: str) -> list[dict]:
    """Parse an RSS feed for coffee-relevant items."""
    import re
    from email.utils import parsedate_to_datetime

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CoffeeMarketBot/1.0)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    }
    resp = requests.get(feed_url, headers=headers, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    is_google_news = "news.google.com" in feed_url

    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        desc_el = item.find("description")
        link_el = item.find("link")
        pub_el = item.find("pubDate")

        raw_title = (title_el.text or "").strip() if title_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        pub = (pub_el.text or "").strip() if pub_el is not None else ""

        source_el = item.find("source")
        if is_google_news:
            actual_source = (source_el.text or "").strip() if source_el is not None else source_name
            title = _strip_publisher_suffix(_clean_text(raw_title), actual_source)
        else:
            title = _clean_text(raw_title)
            actual_source = source_name

        combined = f"{title} {_clean_text(desc)}"
        if not _is_coffee_relevant(combined):
            continue
        if not title or not link:
            continue

        if is_google_news:
            # Google News description just repeats the title in an <a> tag — use clean title
            text = title
        else:
            text = _clean_text(desc) if desc else title
            if not text or len(text) < 20:
                text = title

        try:
            ts = parsedate_to_datetime(pub).isoformat() if pub else ""
        except Exception:
            ts = ""

        items.append({
            "category": _classify_category(combined),
            "text": text[:280],
            "source": actual_source,
            "url": link,
            "timestamp": ts,
        })

        if len(items) >= MAX_ITEMS:
            break

    return items


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    items: list[dict] = []

    # Try RSS feeds first (more targeted, no API limits)
    print("Fetching from RSS feeds...", flush=True)
    for source_name, feed_url in RSS_FEEDS:
        if len(items) >= MAX_ITEMS:
            break
        try:
            rss_items = fetch_from_rss(feed_url, source_name)
            items.extend(rss_items)
            print(f"Got {len(rss_items)} items from {source_name}", flush=True)
        except Exception as exc:
            print(f"RSS feed {source_name} failed: {exc}", file=sys.stderr, flush=True)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped = []
    for item in items:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            deduped.append(item)
    items = deduped[:MAX_ITEMS]

    # Fall back to NewsAPI if RSS didn't yield enough
    if len(items) < MAX_ITEMS and NEWS_API_KEY:
        try:
            print("Supplementing with NewsAPI...", flush=True)
            api_items = fetch_from_newsapi(NEWS_API_KEY)
            for item in api_items:
                if item["url"] not in seen_urls and len(items) < MAX_ITEMS:
                    seen_urls.add(item["url"])
                    items.append(item)
            print(f"NewsAPI added {len(items) - len(deduped)} more items", flush=True)
        except Exception as exc:
            print(f"NewsAPI failed: {exc}", file=sys.stderr, flush=True)

    if not items:
        print("No news items found from any source", file=sys.stderr, flush=True)
        return 1

    items = items[:MAX_ITEMS]
    path = output_dir / "news.json"
    atomic_write_json(path, items)
    print(f"Written {len(items)} items to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
