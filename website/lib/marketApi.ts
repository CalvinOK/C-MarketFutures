import { NextResponse } from 'next/server'

const DEFAULT_LOCAL_API_BASE = 'http://127.0.0.1:8000'

function getConfiguredMarketApiBaseUrl(): string | null {
  const raw =
    process.env.MARKET_API_BASE_URL?.trim() ||
    process.env.NEXT_PUBLIC_MARKET_API_BASE_URL?.trim()

  if (raw && raw.length > 0) {
    return raw.replace(/\/$/, '')
  }

  // In local dev, preserve the old default for convenience.
  if (process.env.NODE_ENV !== 'production') {
    return DEFAULT_LOCAL_API_BASE
  }

  return null
}

function getMarketApiAuthHeader(): string | null {
  const token = process.env.MARKET_API_TOKEN?.trim()
  if (!token) return null
  return `Bearer ${token}`
}

export async function proxyMarketApiGet(request: Request, endpointPath: string): Promise<NextResponse> {
  const marketApiBaseUrl = getConfiguredMarketApiBaseUrl()
  if (!marketApiBaseUrl) {
    return NextResponse.json(
      {
        error: 'Market API base URL is not configured',
        detail:
          'Set MARKET_API_BASE_URL (or NEXT_PUBLIC_MARKET_API_BASE_URL) in the deployment environment.',
      },
      { status: 500 },
    )
  }

  const requestUrl = new URL(request.url)
  const upstreamUrl = `${marketApiBaseUrl}${endpointPath}${requestUrl.search}`

  const headers: HeadersInit = {
    Accept: 'application/json',
  }
  const auth = getMarketApiAuthHeader()
  if (auth) {
    headers.Authorization = auth
  }

  try {
    const upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers,
      cache: 'no-store',
    })

    const text = await upstream.text()
    const contentType = upstream.headers.get('content-type') || 'application/json; charset=utf-8'

    return new NextResponse(text, {
      status: upstream.status,
      headers: {
        'Content-Type': contentType,
      },
    })
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: 'Failed to reach market API server',
        detail: error instanceof Error ? error.message : 'unknown error',
      },
      { status: 502 },
    )
  }
}
