import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { getLatestCoffeeMarketSnapshot } from '@/lib/supabaseServer'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'coffee-market-snapshot', 60, 60_000)
  if (rateError) return rateError

  try {
    const snapshot = await getLatestCoffeeMarketSnapshot()
    if (!snapshot) {
      return NextResponse.json(
        { error: 'No stored Coffee C market snapshot is available' },
        { status: 404, headers: { 'Cache-Control': 'no-store' } },
      )
    }
    return NextResponse.json(snapshot, {
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown market snapshot error'
    console.error('[coffee-market-snapshot] Failed to build snapshot:', message)
    return NextResponse.json(
      { error: 'Unable to retrieve Coffee C market snapshot', detail: message },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}
