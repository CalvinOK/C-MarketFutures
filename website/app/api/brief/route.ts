import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'brief', 60, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/brief', async () => {
    const payload = await readPublicDataJson<Record<string, unknown>>('roaster_brief.json')
    return payload ? NextResponse.json(payload) : null
  })
}
