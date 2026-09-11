import { enforceRateLimit } from '@/lib/apiGuard'
import { fetchLatestCoffeeHistoryCsv, parseCoffeeHistoryCsv } from '@/lib/external/coffeeHistoryCsv'
import {
  getCoffeeMarketHistoryRow,
  upsertCoffeeMarketHistoryRow,
} from '@/lib/supabaseServer'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

function validateRecentMarketDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const parsed = new Date(`${value}T00:00:00Z`)
  if (Number.isNaN(parsed.getTime())) return false
  const now = new Date()
  const todayUtc = new Date(`${now.toISOString().slice(0, 10)}T00:00:00Z`)
  const ageDays = (todayUtc.getTime() - parsed.getTime()) / 86_400_000
  return parsed <= todayUtc && ageDays >= 0 && ageDays <= 14
}

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
    const provider = await fetchLatestCoffeeHistoryCsv()
    const rows = parseCoffeeHistoryCsv(provider.csv)
    for (const row of rows) {
      row.source = provider.source ?? 'databento'
      row.sourceContract = provider.sourceContract
      row.sourceInstrumentId = provider.sourceInstrumentId
      row.sourceRetrievedAt = provider.sourceRetrievedAt
    }
    const candidate = rows.at(-1)
    if (!candidate) {
      return NextResponse.json({ status: 'no_new_market_date', reason: 'Provider returned no daily rows' })
    }
    if (!validateRecentMarketDate(candidate.date)) {
      return NextResponse.json(
        { status: 'provider_data_error', reason: 'invalid_market_date' },
        { status: 422 },
      )
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