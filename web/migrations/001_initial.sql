-- Run once against your Postgres database before starting the app.
-- psql $DATABASE_URL -f migrations/001_initial.sql

-- Processed news items (post-LLM)
CREATE TABLE IF NOT EXISTS news_articles (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  headline      TEXT NOT NULL,
  summary       TEXT NOT NULL,
  category      TEXT NOT NULL
                  CHECK (category IN ('Market Brief', 'Commodities Desk', 'Trade Note')),
  source_name   TEXT NOT NULL,
  source_url    TEXT,
  published_at  TIMESTAMPTZ NOT NULL,
  fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  headline_hash TEXT NOT NULL UNIQUE,
  raw_content   TEXT
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_fetched   ON news_articles (fetched_at DESC);

-- Futures contract snapshots (append-only time-series)
CREATE TABLE IF NOT EXISTS futures_snapshots (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol           TEXT NOT NULL,
  expiry_date      DATE NOT NULL,
  last_price       NUMERIC(10,2) NOT NULL,
  price_change     NUMERIC(10,2) NOT NULL DEFAULT 0,
  price_change_pct NUMERIC(8,6) NOT NULL DEFAULT 0,
  volume           BIGINT NOT NULL DEFAULT 0,
  open_interest    BIGINT NOT NULL DEFAULT 0,
  source           TEXT NOT NULL,
  captured_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_futures_symbol_time ON futures_snapshots (symbol, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_futures_captured    ON futures_snapshots (captured_at DESC);

-- Materialized view: one row per symbol, most-recent snapshot
-- API routes read from this for O(1) lookup
CREATE MATERIALIZED VIEW IF NOT EXISTS contracts_latest AS
  SELECT DISTINCT ON (symbol)
    symbol, expiry_date, last_price, price_change,
    price_change_pct, volume, open_interest, source, captured_at
  FROM futures_snapshots
  ORDER BY symbol, captured_at DESC;

CREATE UNIQUE INDEX IF NOT EXISTS contracts_latest_symbol ON contracts_latest (symbol);

-- Ingestion audit log
CREATE TABLE IF NOT EXISTS ingestion_log (
  id          SERIAL PRIMARY KEY,
  job_name    TEXT NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('success', 'partial', 'failed')),
  records_in  INT  NOT NULL DEFAULT 0,
  records_out INT  NOT NULL DEFAULT 0,
  duration_ms INT,
  error_msg   TEXT,
  ran_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_ran ON ingestion_log (ran_at DESC);
