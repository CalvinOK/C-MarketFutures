import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { getCoffeeMarketSnapshot } from '@/lib/coffeeMarketSnapshotService'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'coffee-market-snapshot', 60, 60_000)
  if (rateError) return rateError

  try {
    const result = await getCoffeeMarketSnapshot()
    if (!result) {
      return NextResponse.json(
        { error: 'No stored Coffee C market snapshot is available' },
        { status: 404, headers: { 'Cache-Control': 'no-store' } },
      )
    }
    return NextResponse.json(result, {
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (error: unknown) {
    console.error('[coffee-market-snapshot] Failed to build snapshot:', error instanceof Error ? error.message : String(error))
    return NextResponse.json(
      { error: 'Unable to retrieve Coffee C market snapshot' },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}
