-- Run in Supabase SQL Editor as one statement. PostgreSQL runs this migration
-- transactionally; a malformed date or duplicate aborts the whole migration.
BEGIN;

-- Preserve the original text Date column while adding a normalized key for
-- chronological reads and idempotent daily upserts.
ALTER TABLE public."Coffee C Historical Data"
  ADD COLUMN IF NOT EXISTS market_date date;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM public."Coffee C Historical Data"
    WHERE "Date" ~ '^\d{2}/\d{2}/\d{4}$'
      AND to_char(to_date("Date", 'MM/DD/YYYY'), 'MM/DD/YYYY') <> "Date"
  ) THEN
    RAISE EXCEPTION 'Coffee C Historical Data contains an invalid calendar date. Diagnostic: SELECT "Date" FROM public."Coffee C Historical Data" WHERE "Date" ~ ''^\d{2}/\d{2}/\d{4}$'' AND to_char(to_date("Date", ''MM/DD/YYYY''), ''MM/DD/YYYY'') <> "Date";';
  END IF;
END $$;

UPDATE public."Coffee C Historical Data"
SET market_date = CASE
  WHEN "Date" ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date("Date", 'MM/DD/YYYY')
  WHEN "Date" ~ '^\d{4}-\d{2}-\d{2}$' THEN "Date"::date
  ELSE NULL
END
WHERE market_date IS NULL;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM public."Coffee C Historical Data"
    WHERE market_date IS NULL
  ) THEN
    RAISE EXCEPTION 'Coffee C Historical Data contains an unparseable Date value. Diagnostic: SELECT "Date" FROM public."Coffee C Historical Data" WHERE market_date IS NULL;';
  END IF;
END $$;

DO $$
DECLARE duplicate_dates text;
BEGIN
  SELECT string_agg(format('%s (%s rows)', market_date, row_count), ', ' ORDER BY market_date)
  INTO duplicate_dates
  FROM (
    SELECT market_date, count(*) AS row_count
    FROM public."Coffee C Historical Data"
    GROUP BY market_date
    HAVING count(*) > 1
  ) duplicates;

  IF duplicate_dates IS NOT NULL THEN
    RAISE EXCEPTION 'Duplicate normalized market dates found: %. Diagnostic: SELECT market_date, count(*) FROM public."Coffee C Historical Data" GROUP BY market_date HAVING count(*) > 1;', duplicate_dates;
  END IF;
END $$;

ALTER TABLE public."Coffee C Historical Data"
  ALTER COLUMN market_date SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS coffee_c_historical_data_market_date_key
  ON public."Coffee C Historical Data" (market_date);

COMMIT;