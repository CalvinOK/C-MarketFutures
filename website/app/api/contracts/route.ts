import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { readPublicDataJson } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

// Type matches website/app/page.tsx ContractApiRow; the only fields we strictly
// need to detect "usable" data are symbol + last_price.
type ContractRow = {
  symbol?: string
  last_price?: number
  [key: string]: unknown
}

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'contracts', 120, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/contracts', async () => {
    // Cached snapshot written by the Python pipeline (run_old_projection_pipeline.py
    // and the snapshot job). Lets the page render the contract grid even when
    // the upstream Flask service is unreachable on Vercel.
    const cached = await readPublicDataJson<ContractRow[]>('contracts.json')
    if (!Array.isArray(cached) || cached.length === 0) return null

    const usable = cached.filter(
      (row) => typeof row.symbol === 'string' && Number.isFinite(row.last_price),
    )
    if (usable.length === 0) return null

    return NextResponse.json(usable)
  })
}
