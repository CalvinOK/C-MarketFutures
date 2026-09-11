import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { getCoffeeContracts } from '@/lib/coffeeContracts'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'contracts', 120, 60_000)
  if (rateError) return rateError

  try {
    const result = await getCoffeeContracts()
    return NextResponse.json(result, { headers: { 'Cache-Control': 'no-store' } })
  } catch (error: unknown) {
    console.error('[coffee-contracts] request failed:', error instanceof Error ? error.message : String(error))
    return NextResponse.json(
      { error: 'Unable to retrieve Coffee C contracts' },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}
