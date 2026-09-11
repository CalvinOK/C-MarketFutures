import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { getCoffeeMarketHistory } from '@/lib/supabaseServer'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

function validDate(value: string | null): string | undefined {
  return value && /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : undefined
}

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'coffee-history', 60, 60_000)
  if (rateError) return rateError

  const searchParams = new URL(request.url).searchParams
  const from = validDate(searchParams.get('from'))
  const to = validDate(searchParams.get('to'))
  const requestedLimit = Number(searchParams.get('limit') ?? 5000)
  const limit = Number.isInteger(requestedLimit) && requestedLimit > 0
    ? Math.min(requestedLimit, 5000)
    : 5000

  if ((searchParams.has('from') && !from) || (searchParams.has('to') && !to)) {
    return NextResponse.json({ error: 'from and to must use YYYY-MM-DD format' }, { status: 400 })
  }

  try {
    const data = await getCoffeeMarketHistory({ from, to, limit })
    return NextResponse.json(
      { format: 'coffee-market-history.v1', data },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown coffee history error'
    console.error('[coffee-history] Failed to read history:', message)
    return NextResponse.json(
      { error: 'Unable to retrieve Coffee C historical prices', detail: message },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}