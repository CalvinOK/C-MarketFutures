import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'contracts', 120, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/contracts', async () => {
    const rows = await readPublicDataJson<unknown[]>('contracts.json')
    return rows ? NextResponse.json(rows) : null
  })
}
