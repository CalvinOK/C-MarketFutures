import crypto from 'crypto'
import { fetchWithTimeout } from '@/lib/http'

export type RawArticle = {
  title: string
  description: string | null
  url: string
  publishedAt: string
  source: { name: string }
}

const QUERIES = [
  '"coffee futures" OR "ICE coffee" OR arabica futures',
  '"Brazil coffee" AND (frost OR drought OR weather OR rainfall)',
  'coffee AND (futures OR contract OR curve OR spread)',
  'coffee AND (export OR shipment OR logistics OR inventory)',
  'coffee AND (crop OR harvest OR yield OR production)',
]

const LOOKBACK_WINDOWS_DAYS = [30, 90, 365]

const COFFEE_DIRECT_TERMS = [
  'coffee futures',
  'arabica futures',
  'robusta futures',
  'ice coffee',
  'coffee contract',
  'coffee beans',
  'green coffee',
  'c-market',
  'c market',
  'nybot',
]

const COFFEE_SUPPORT_TERMS = [
  'coffee',
  'arabica',
  'robusta',
  'futures',
  'contract',
  'spread',
  'curve',
  'hedge',
  'hedging',
  'cot',
  'inventory',
  'shipment',
  'logistics',
  'export',
  'trade',
  'price',
  'weather',
  'frost',
  'drought',
  'rainfall',
  'crop',
  'harvest',
  'yield',
  'production',
  'plantation',
  'supply',
  'demand',
  'brazil',
  'vietnam',
  'colombia',
  'ethiopia',
  'honduras',
  'peru',
  'uganda',
  'kenya',
  'indonesia',
  'mexico',
  'sao paulo',
  'minas gerais',
]

function parseDateOrEpoch(value: string): number {
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function isRelevantCoffeeNews(title: string, description: string | null): boolean {
  const text = `${title} ${description ?? ''}`.toLowerCase()

  if (COFFEE_DIRECT_TERMS.some((term) => text.includes(term))) {
    return true
  }

  if (text.includes('coffee') || text.includes('arabica') || text.includes('robusta')) {
    // Require at least 2 support terms so generic mentions ("coffee prices rise")
    // don't slip through as commodity/futures news.
    const matchCount = COFFEE_SUPPORT_TERMS.filter((term) => text.includes(term)).length
    return matchCount >= 2
  }

  return false
}

async function fetchWindowArticles(apiKey: string, fromDate: string, seen: Set<string>): Promise<RawArticle[]> {
  const results = await Promise.allSettled(
    QUERIES.map((q) =>
      fetchWithTimeout(
        `https://newsapi.org/v2/everything` +
          `?q=${encodeURIComponent(q)}` +
          `&from=${fromDate}` +
          `&language=en` +
          `&sortBy=publishedAt` +
          `&pageSize=10` +
          `&apiKey=${apiKey}`,
        { cache: 'no-store' },
        8_000,
      ).then((r) => {
        if (!r.ok) throw new Error(`NewsAPI HTTP ${r.status}`)
        return r.json()
      }),
    ),
  )

  const windowArticles: RawArticle[] = []
  for (const result of results) {
    if (result.status === 'fulfilled' && Array.isArray(result.value?.articles)) {
      for (const a of result.value.articles) {
        const title = a?.title ?? ''
        if (!title || title === '[Removed]') continue
        const description = a?.description ?? null
        if (!isRelevantCoffeeNews(title, description)) continue
        const hash = hashHeadline(title)
        if (seen.has(hash)) continue
        seen.add(hash)
        windowArticles.push({
          title,
          description,
          url: a?.url ?? '',
          publishedAt: a?.publishedAt ?? '',
          source: { name: a?.source?.name ?? 'Unknown' },
        })
      }
    } else if (result.status === 'rejected') {
      console.warn('[newsapi] Query failed:', result.reason?.message)
    }
  }

  windowArticles.sort((left, right) => parseDateOrEpoch(right.publishedAt) - parseDateOrEpoch(left.publishedAt))
  return windowArticles
}

export async function fetchCoffeeNews(): Promise<RawArticle[]> {
  const apiKey = process.env.NEWS_API_KEY
  if (!apiKey) throw new Error('NEWS_API_KEY is not set')

  const articles: RawArticle[] = []

  for (const daysBack of LOOKBACK_WINDOWS_DAYS) {
    const from = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
    const windowArticles = await fetchWindowArticles(
      apiKey,
      from,
      new Set(articles.map((article) => hashHeadline(article.title))),
    )
    articles.push(...windowArticles)
    if (articles.length >= 3) {
      break
    }
  }

  return deduplicateArticles(articles).sort(
    (left, right) => parseDateOrEpoch(right.publishedAt) - parseDateOrEpoch(left.publishedAt),
  )
}

function deduplicateArticles(articles: RawArticle[]): RawArticle[] {
  const seen = new Set<string>()
  return articles.filter((a) => {
    if (!a.title || a.title === '[Removed]') return false
    const hash = hashHeadline(a.title)
    if (seen.has(hash)) return false
    seen.add(hash)
    return true
  })
}

export function hashHeadline(title: string): string {
  const normalized = title.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim()
  return crypto.createHash('sha256').update(normalized).digest('hex')
}
