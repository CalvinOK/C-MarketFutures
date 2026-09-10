import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

// Subset of SucafinaBriefApiItem in website/app/page.tsx: headline +
// source_report are the fields the page checks before treating the brief as
// "live". Other fields are passed through verbatim.
type BriefPayload = {
  headline?: string
  source_report?: string
  [key: string]: unknown
}

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'brief', 60, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/brief', async () => {
    const cached = await readPublicDataJson<BriefPayload>('roaster_brief.json')
    if (!cached || !cached.headline || !cached.source_report) return null
    return NextResponse.json(cached)
  })
}
