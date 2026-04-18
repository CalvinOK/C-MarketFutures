import { NextResponse } from 'next/server'
import { getDb } from '@/lib/db'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'snapshot', 120, 60_000)
  if (rateError) return rateError

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
      volume: number
      open_interest: number
      captured_at: Date
    }>(
      `SELECT symbol, expiry_date, last_price, volume, open_interest, captured_at
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

    const front = rows[0]
    const deferred = rows[rows.length - 1]

    const frontPrice = Number(front.last_price)
    const deferredPrice = Number(deferred.last_price)

    const snapshot = {
      frontPrice,
      curveShape: deferredPrice > frontPrice ? 'Contango' : 'Backwardation',
      totalVolume: rows.reduce((sum, r) => sum + Number(r.volume), 0),
      totalOpenInterest: rows.reduce((sum, r) => sum + Number(r.open_interest), 0),
      frontSymbol: front.symbol,
      asOf: front.captured_at,
    }

    return NextResponse.json(snapshot)
  } catch (err: unknown) {
    console.error('[GET /api/snapshot]', (err as Error).message)
    return NextResponse.json(
      { error: 'Failed to compute snapshot', retryAfter: 30 },
      { status: 503 },
    )
  }
}
