import { NextResponse } from 'next/server'

const DEFAULT_LOCAL_API_BASE = 'http://127.0.0.1:8000'

function getMarketApiBaseUrl(): string {
  const raw = process.env.MARKET_API_BASE_URL?.trim()
  if (raw && raw.length > 0) {
    return raw.replace(/\/$/, '')
  }
  return DEFAULT_LOCAL_API_BASE
}

function getMarketApiAuthHeader(): string | null {
  const token = process.env.MARKET_API_TOKEN?.trim()
  if (!token) return null
  return `Bearer ${token}`
}

export async function proxyMarketApiGet(request: Request, endpointPath: string): Promise<NextResponse> {
  const requestUrl = new URL(request.url)
  const upstreamUrl = `${getMarketApiBaseUrl()}${endpointPath}${requestUrl.search}`

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
