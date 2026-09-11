ALTER TABLE public."Coffee C Historical Data"
  ADD COLUMN IF NOT EXISTS source text,
  ADD COLUMN IF NOT EXISTS source_contract text,
  ADD COLUMN IF NOT EXISTS source_instrument_id text,
  ADD COLUMN IF NOT EXISTS source_retrieved_at timestamptz;

CREATE INDEX IF NOT EXISTS coffee_c_historical_data_source_contract_idx
  ON public."Coffee C Historical Data" (source_contract, market_date DESC);
