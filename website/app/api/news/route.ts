import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'news', 60, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/news', async (incomingRequest) => {
    const rows = await readPublicDataJson<unknown[]>('news.json')
    if (!rows) return null

    const limit = new URL(incomingRequest.url).searchParams.get('limit')
    const parsedLimit = limit ? Number(limit) : 3
    const safeLimit = Number.isFinite(parsedLimit) && parsedLimit > 0 ? Math.min(parsedLimit, 20) : 3

    return NextResponse.json(rows.slice(0, safeLimit))
  })
}
