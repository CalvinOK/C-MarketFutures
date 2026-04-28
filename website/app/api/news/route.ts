import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'
import { proxyMarketApiGet } from '@/lib/marketApi'
import { fetchCoffeeNews } from '@/lib/external/newsapi'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

async function fetchNewsDirectly(): Promise<NextResponse | null> {
  if (!process.env.NEWS_API_KEY) return null

  try {
    const articles = await fetchCoffeeNews()
    if (articles.length === 0) return null

    // Map RawArticle → NewsApiItem shape expected by the frontend
    const items = articles.map((a) => ({
      category: 'Market News',
      text: a.title,
      source: a.source.name,
      url: a.url || null,
      timestamp: a.publishedAt,
    }))

    return NextResponse.json(items)
  } catch {
    return null
  }
}

const BLOCKED_SOURCE_TERMS = ['stonex']

function isBlockedNewsItem(item: { source?: string; url?: string }): boolean {
  const source = (item.source ?? '').toLowerCase()
  const url = (item.url ?? '').toLowerCase()
  return BLOCKED_SOURCE_TERMS.some((s) => source.includes(s) || url.includes(s))
}

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'news', 60, 60_000)
  if (rateError) return rateError

  const marketApiResponse = await proxyMarketApiGet(request, '/api/news')

  if (!marketApiResponse.ok) {
    const directResponse = await fetchNewsDirectly()
    if (directResponse) return directResponse
    return marketApiResponse
  }

  try {
    const raw = await marketApiResponse.json()
    const items: unknown[] = Array.isArray(raw) ? raw : Array.isArray((raw as { data?: unknown[] }).data) ? (raw as { data: unknown[] }).data : []
    const filtered = items.filter((item) => !isBlockedNewsItem(item as { source?: string; url?: string }))
    return NextResponse.json(filtered)
  } catch {
    return NextResponse.json([], { status: 200 })
  }
}
