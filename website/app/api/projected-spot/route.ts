import { NextResponse } from 'next/server'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { extractAsOfDateFromCsv, readPublicDataText } from '@/lib/publicData'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'projected-spot', 120, 60_000)
  if (rateError) return rateError

  return proxyMarketApiGet(request, '/api/projected-spot', async (incomingRequest) => {
    const historyCsv = await readPublicDataText('coffee_xgb_proj4_history.csv')
    const forecastCsv = await readPublicDataText('coffee_xgb_proj4_rolling_path.csv')
    if (!historyCsv || !forecastCsv) return null

    const format = new URL(incomingRequest.url).searchParams.get('format')
    if (format === 'csv') {
      return new NextResponse(forecastCsv, { headers: { 'Content-Type': 'text/csv; charset=utf-8' } })
    }

    return NextResponse.json({
      format: 'projected-spot-csv.v1',
      files: {
        history: 'coffee_xgb_proj4_history.csv',
        forecast: 'coffee_xgb_proj4_rolling_path.csv',
      },
      asOfDate: extractAsOfDateFromCsv(forecastCsv),
      historyCsv,
      forecastCsv,
    })
  })
}
