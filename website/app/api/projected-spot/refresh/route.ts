import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'

export const dynamic = 'force-dynamic'

// Triggers a fresh run of the XGBoost projection pipeline on the upstream
// Flask API. The Flask service handles all the heavy lifting (data fetch,
// merge, train, write CSVs) and only returns once the new
// coffee_xgb_proj4_rolling_path.csv is available.
//
// Intended callers:
//   • Vercel cron (weekly, see website/vercel.json) — sends an Authorization
//     header with CRON_SECRET if configured.
//   • Manual ops trigger — `curl -H "Authorization: Bearer $INTERNAL_API_TOKEN"
//     https://<host>/api/projected-spot/refresh`.
//
// Auth: accepts either the CRON_SECRET (Vercel cron) or the INTERNAL_API_TOKEN
// (used elsewhere in the codebase). If neither env var is set, the route is
// open — fine for local dev, but you should set one in production.
function authorize(request: Request): NextResponse | null {
  const cronSecret = process.env.CRON_SECRET?.trim()
  if (cronSecret) {
    const authHeader = request.headers.get('authorization') ?? ''
    const presented = authHeader.startsWith('Bearer ')
      ? authHeader.slice(7).trim()
      : ''
    if (presented === cronSecret) return null
  }
  return requireInternalTokenIfConfigured(request)
}

export async function GET(request: Request) {
  const authError = authorize(request)
  if (authError) return authError

  // Generous limit — cron only hits weekly, manual triggers are rare. The
  // Flask refresh itself takes minutes, so we don't want callers retrying
  // aggressively either.
  const rateError = enforceRateLimit(request, 'projected-spot-refresh', 6, 60_000)
  if (rateError) return rateError

  // Force a refresh upstream regardless of staleness. The Flask handler at
  // /api/projected-spot reads `run=true` and re-runs run_old_projection_pipeline.py.
  // proxyMarketApiGet appends the incoming request's query string, so synthesize
  // a request whose search params include run=true.
  const upstreamUrl = new URL(request.url)
  upstreamUrl.search = ''
  upstreamUrl.searchParams.set('run', 'true')
  const upstreamRequest = new Request(upstreamUrl.toString(), {
    method: 'GET',
    headers: request.headers,
  })

  return proxyMarketApiGet(upstreamRequest, '/api/projected-spot', async () => {
    // No fallback path: if upstream is unreachable, surface the failure so
    // the cron alert is visible rather than silently "succeeding".
    return NextResponse.json(
      {
        error: 'Upstream market API is required for projection refresh',
        detail:
          'Set MARKET_API_BASE_URL so /api/projected-spot/refresh can reach the Flask service.',
      },
      { status: 503 },
    )
  })
}
