import { NextResponse } from 'next/server'
import { getDb } from '@/lib/db'
import { cacheGet, cacheSet } from '@/lib/redis'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'

export const dynamic = 'force-dynamic'

const CACHE_KEY = 'contracts:live'
const CACHE_TTL = 120 // 2 minutes
const STALE_THRESHOLD_MS = 10 * 60 * 1000 // flag as stale after 10 min

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'contracts', 120, 60_000)
  if (rateError) return rateError

  // 1. Redis cache hit
  const cached = await cacheGet(CACHE_KEY)
  if (cached) {
    return NextResponse.json(JSON.parse(cached), {
      headers: { 'X-Cache': 'HIT' },
    })
  }

  // 2. Query database
  let db
  try {
    db = getDb()
  } catch {
    return NextResponse.json(
      { error: 'Database not configured', retryAfter: 60 },
      { status: 503 },
    )
  }

  try {
    const { rows } = await db.query<{
      symbol: string
      expiry_date: string
      last_price: number
      price_change: number
      price_change_pct: number
      volume: number
      open_interest: number
      captured_at: Date
    }>(
      `SELECT symbol, expiry_date, last_price, price_change,
              price_change_pct, volume, open_interest, captured_at
       FROM contracts_latest
       WHERE symbol LIKE 'KC%'
       ORDER BY expiry_date ASC`,
    )

    if (rows.length === 0) {
      return NextResponse.json(
        { error: 'No contract data available', retryAfter: 120 },
        { status: 503 },
      )
    }

    const isStale = rows.some(
      (r) => Date.now() - new Date(r.captured_at).getTime() > STALE_THRESHOLD_MS,
    )

    if (!isStale) {
      await cacheSet(CACHE_KEY, CACHE_TTL, JSON.stringify(rows))
    }

    return NextResponse.json(rows, {
      headers: {
        'X-Cache': 'MISS',
        'X-Data-Freshness': isStale ? 'stale' : 'live',
      },
    })
  } catch (err: unknown) {
    console.error('[GET /api/contracts]', (err as Error).message)
    return NextResponse.json(
      { error: 'Failed to fetch contracts', retryAfter: 30 },
      { status: 503 },
    )
  }
}
