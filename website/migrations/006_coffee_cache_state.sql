-- Daily Coffee C contract cache and distributed refresh leases.
CREATE TABLE IF NOT EXISTS public.coffee_contract_snapshots (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trade_date          DATE NOT NULL,
  symbol              TEXT NOT NULL,
  contract_name       TEXT,
  month_code          TEXT,
  expiration_date     DATE,
  settlement          NUMERIC,
  last_price          NUMERIC,
  price_change        NUMERIC,
  percent_change      NUMERIC,
  volume              BIGINT,
  open_interest       BIGINT,
  provider_timestamp  TIMESTAMPTZ,
  source              TEXT NOT NULL DEFAULT 'databento',
  fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  raw_metadata        JSONB,
  CONSTRAINT coffee_contract_snapshots_daily_key UNIQUE (trade_date, symbol, source)
);

CREATE INDEX IF NOT EXISTS coffee_contract_snapshots_trade_date_idx
  ON public.coffee_contract_snapshots (trade_date DESC);
CREATE INDEX IF NOT EXISTS coffee_contract_snapshots_symbol_idx
  ON public.coffee_contract_snapshots (symbol);
CREATE INDEX IF NOT EXISTS coffee_contract_snapshots_fetched_at_idx
  ON public.coffee_contract_snapshots (fetched_at DESC);

-- Refresh state is also the lease record. A lease is held by a token, not by a
-- database session, so it remains valid across serverless HTTP requests.
CREATE TABLE IF NOT EXISTS public.coffee_data_refresh_state (
  dataset              TEXT PRIMARY KEY,
  status               TEXT NOT NULL DEFAULT 'idle',
  last_attempt_at      TIMESTAMPTZ,
  last_success_at      TIMESTAMPTZ,
  refresh_started_at   TIMESTAMPTZ,
  lock_expires_at      TIMESTAMPTZ,
  lock_token           UUID,
  last_error_code      TEXT,
  last_error_message   TEXT,
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS lock_expires_at TIMESTAMPTZ;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS lock_token UUID;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'idle';
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS last_success_at TIMESTAMPTZ;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS refresh_started_at TIMESTAMPTZ;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS last_error_code TEXT;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS last_error_message TEXT;
ALTER TABLE public.coffee_data_refresh_state ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- Keep every existing market snapshot row. Only remove the old daily key so
-- successful hourly collections can be retained as history.
ALTER TABLE public.coffee_market_snapshots
  DROP CONSTRAINT IF EXISTS coffee_market_snapshots_market_date_key;
CREATE INDEX IF NOT EXISTS coffee_market_snapshots_retrieved_at_idx
  ON public.coffee_market_snapshots (retrieved_at DESC);

CREATE OR REPLACE FUNCTION public.try_claim_coffee_refresh(
  p_dataset TEXT,
  p_lease_seconds INTEGER DEFAULT 120
)
RETURNS TABLE (claimed BOOLEAN, lock_token UUID)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
  v_token UUID := gen_random_uuid();
BEGIN
  INSERT INTO public.coffee_data_refresh_state
    (dataset, status, last_attempt_at, refresh_started_at, lock_expires_at, lock_token, updated_at)
  VALUES
    (p_dataset, 'running', now(), now(), now() + make_interval(secs => p_lease_seconds), v_token, now())
  ON CONFLICT (dataset) DO UPDATE SET
    status = 'running',
    last_attempt_at = now(),
    refresh_started_at = now(),
    lock_expires_at = now() + make_interval(secs => p_lease_seconds),
    lock_token = v_token,
    updated_at = now()
  WHERE public.coffee_data_refresh_state.lock_expires_at IS NULL
     OR public.coffee_data_refresh_state.lock_expires_at <= now();

  IF FOUND THEN
    RETURN QUERY SELECT TRUE, v_token;
  ELSE
    RETURN QUERY SELECT FALSE, NULL::UUID;
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.complete_coffee_refresh(
  p_dataset TEXT,
  p_lock_token UUID,
  p_succeeded BOOLEAN,
  p_error_code TEXT DEFAULT NULL,
  p_error_message TEXT DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY INVOKER
SET search_path = public
AS $$
  UPDATE public.coffee_data_refresh_state
     SET status = CASE WHEN p_succeeded THEN 'success' ELSE 'failed' END,
         last_success_at = CASE WHEN p_succeeded THEN now() ELSE last_success_at END,
         lock_expires_at = NULL,
         lock_token = NULL,
         refresh_started_at = NULL,
         last_error_code = CASE WHEN p_succeeded THEN NULL ELSE p_error_code END,
         last_error_message = CASE WHEN p_succeeded THEN NULL ELSE left(p_error_message, 500) END,
         updated_at = now()
   WHERE dataset = p_dataset
     AND lock_token = p_lock_token
  RETURNING TRUE;
$$;

REVOKE ALL ON FUNCTION public.try_claim_coffee_refresh(TEXT, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.complete_coffee_refresh(TEXT, UUID, BOOLEAN, TEXT, TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.try_claim_coffee_refresh(TEXT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.complete_coffee_refresh(TEXT, UUID, BOOLEAN, TEXT, TEXT) TO service_role;
