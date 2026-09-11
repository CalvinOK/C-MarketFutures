import { enforceRateLimit } from '@/lib/apiGuard'
import { compareCoffeeHistoryRows, fetchLatestCoffeeHistoryCsv, parseCoffeeHistoryCsv } from '@/lib/external/coffeeHistoryCsv'
import {
  getCoffeeMarketHistoryRow,
  upsertCoffeeMarketHistoryRow,
} from '@/lib/supabaseServer'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

function authorized(request: Request): boolean {
  const cronSecret = process.env.CRON_SECRET?.trim()
  const internalToken = process.env.INTERNAL_API_TOKEN?.trim()
  const authorization = request.headers.get('authorization') ?? ''
  const token = authorization.startsWith('Bearer ') ? authorization.slice(7).trim() : ''
  return Boolean((cronSecret && token === cronSecret) || (internalToken && token === internalToken))
}

export async function GET(request: Request) {
  if (!process.env.CRON_SECRET && !process.env.INTERNAL_API_TOKEN) {
    return NextResponse.json({ error: 'History update authentication is not configured' }, { status: 500 })
  }
  if (!authorized(request)) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const rateError = enforceRateLimit(request, 'coffee-history-update', 10, 60_000)
  if (rateError) return rateError

  try {
    const csv = await fetchLatestCoffeeHistoryCsv()
    const rows = parseCoffeeHistoryCsv(csv)
    const overlappingRows = []
    for (const row of rows) {
      const existing = await getCoffeeMarketHistoryRow(row.date)
      if (existing) overlappingRows.push(existing)
    }
    const compatibility = compareCoffeeHistoryRows(overlappingRows, rows)
    if (compatibility.compatible === false) {
      return NextResponse.json(
        { status: 'provider_incompatible', error: 'Databento and Investing.com OHLC values differ materially', compatibility },
        { status: 422 },
      )
    }
    if (
      compatibility.compatible === null &&
      process.env.COFFEE_HISTORY_COMPATIBILITY_VERIFIED !== 'true'
    ) {
      return NextResponse.json(
        { status: 'provider_compatibility_unverified', error: 'No overlapping Investing.com/Databento dates were available; set COFFEE_HISTORY_COMPATIBILITY_VERIFIED=true only after review', compatibility },
        { status: 412 },
      )
    }
    const candidate = rows.at(-1)
    if (!candidate) {
      return NextResponse.json({ status: 'no_new_market_date', reason: 'Provider returned no daily rows' })
    }

    const existing = await getCoffeeMarketHistoryRow(candidate.date)
    if (
      existing &&
      existing.price === candidate.price &&
      existing.open === candidate.open &&
      existing.high === candidate.high &&
      existing.low === candidate.low &&
      existing.volume === candidate.volume &&
      existing.changePercent === candidate.changePercent
    ) {
      return NextResponse.json({ status: 'no_new_market_date', date: candidate.date })
    }

    const updated = await upsertCoffeeMarketHistoryRow(candidate)
    return NextResponse.json({ status: updated ? 'updated' : 'inserted', date: candidate.date, row: candidate })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown history update error'
    const lower = message.toLowerCase()
    const isParsingError = lower.includes('csv')
    const isDatabaseError = lower.includes('supabase')
    const status = isParsingError ? 422 : 502
    console.error('[coffee-history-update] Failed:', message)
    return NextResponse.json(
      { status: isParsingError ? 'parsing_error' : isDatabaseError ? 'database_error' : 'provider_error', error: message },
      { status },
    )
  }
}