CREATE TABLE IF NOT EXISTS coffee_market_snapshots (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  market_date date NOT NULL,
  front_contract text NOT NULL,
  front_price numeric(12, 6) NOT NULL CHECK (front_price > 0),
  next_contract text NOT NULL,
  next_price numeric(12, 6) NOT NULL CHECK (next_price > 0),
  shape text NOT NULL CHECK (shape IN ('Contango', 'Backwardation', 'Flat')),
  spread numeric(12, 6) NOT NULL,
  volume bigint NOT NULL CHECK (volume >= 0),
  open_interest bigint NOT NULL CHECK (open_interest >= 0),
  open_interest_as_of date NOT NULL,
  price_as_of timestamptz,
  unit text NOT NULL CHECK (unit = 'US¢/lb'),
  retrieved_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT coffee_market_snapshots_market_date_key UNIQUE (market_date)
);

CREATE INDEX IF NOT EXISTS idx_coffee_market_snapshots_latest
  ON coffee_market_snapshots (market_date DESC, retrieved_at DESC);
