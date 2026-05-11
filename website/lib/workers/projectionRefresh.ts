/**
 * Weekly refresh of the XGBoost projection CSV.
 *
 * The Flask service (MARKET_API_BASE_URL) does all the heavy lifting — fetch
 * latest market data, merge, train, write coffee_xgb_proj4_rolling_path.csv
 * with `as_of_date = today`. We just kick it with ?run=true and surface any
 * non-2xx response so failures are visible in worker logs.
 */

function getMarketApiBaseUrl(): string | null {
  const raw =
    process.env.MARKET_API_BASE_URL?.trim() ||
    process.env.NEXT_PUBLIC_MARKET_API_BASE_URL?.trim()
  if (!raw) return null
  return raw.replace(/\/$/, '')
}

function getAuthHeader(): Record<string, string> {
  const token = process.env.MARKET_API_TOKEN?.trim()
  if (!token) return {}
  return { Authorization: `Bearer ${token}` }
}

export async function runProjectionRefresh(): Promise<void> {
  const baseUrl = getMarketApiBaseUrl()
  if (!baseUrl) {
    console.warn(
      '[projection-refresh] MARKET_API_BASE_URL not set; skipping weekly refresh.',
    )
    return
  }

  const url = `${baseUrl}/api/projected-spot?run=true`
  const start = Date.now()
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json', ...getAuthHeader() },
      cache: 'no-store',
    })
    const elapsedMs = Date.now() - start

    if (!response.ok) {
      const body = (await response.text()).slice(0, 400).trim()
      console.error(
        `[projection-refresh] Upstream returned ${response.status} after ${elapsedMs}ms: ${body}`,
      )
      return
    }

    // We don't need the body — just confirm as_of_date moved forward.
    try {
      const payload = (await response.json()) as { asOfDate?: string | null }
      console.log(
        `[projection-refresh] Refreshed projection (as_of_date=${payload.asOfDate ?? 'unknown'}) in ${elapsedMs}ms`,
      )
    } catch {
      console.log(`[projection-refresh] Refreshed projection in ${elapsedMs}ms`)
    }
  } catch (err) {
    console.error(
      '[projection-refresh] Failed to reach market API:',
      err instanceof Error ? err.message : err,
    )
  }
}
