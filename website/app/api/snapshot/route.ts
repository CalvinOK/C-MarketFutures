import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'snapshot', 120, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/snapshot', async () => {
    const payload = await readPublicDataJson<Record<string, unknown>>('snapshot.json')
    return payload ? NextResponse.json(payload) : null
  })
}
