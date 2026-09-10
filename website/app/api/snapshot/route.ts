import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

// Mirrors SnapshotData in website/app/page.tsx. frontPrice is the only field
// the page strictly requires to consider the snapshot "live".
type SnapshotPayload = {
  frontPrice?: number
  [key: string]: unknown
}

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'snapshot', 120, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/snapshot', async () => {
    const cached = await readPublicDataJson<SnapshotPayload>('snapshot.json')
    if (!cached || !Number.isFinite(cached.frontPrice)) return null
    return NextResponse.json(cached)
  })
}
