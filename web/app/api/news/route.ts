import { NextResponse } from 'next/server'
import { getDb } from '@/lib/db'
import { cacheGet, cacheSet } from '@/lib/redis'
import { enforceRateLimit, requireInternalTokenIfConfigured } from '@/lib/apiGuard'

export const dynamic = 'force-dynamic'

const CACHE_KEY = 'news:feed'
const CACHE_TTL = 1200 // 20 minutes
const MAX_ARTICLES = 3

export async function GET(request: Request) {
  const authError = requireInternalTokenIfConfigured(request)
  if (authError) return authError

  const rateError = enforceRateLimit(request, 'news', 60, 60_000)
  if (rateError) return rateError

  // 1. Redis cache hit
  const cached = await cacheGet(CACHE_KEY)
  if (cached) {
    return NextResponse.json(JSON.parse(cached), {
      headers: { 'X-Cache': 'HIT' },
    })
  }

  // 2. Query database
  let db
  try {
    db = getDb()
  } catch {
    return NextResponse.json(
      { error: 'Database not configured', retryAfter: 60 },
      { status: 503 },
    )
  }

  try {
    const { rows } = await db.query<{
      category: string
      text: string
      source: string
      timestamp: Date
    }>(
      `SELECT category,
              summary   AS text,
              source_name AS source,
              published_at AS timestamp
       FROM news_articles
       WHERE published_at > NOW() - INTERVAL '24 hours'
       ORDER BY published_at DESC
      LIMIT ${MAX_ARTICLES}`,
    )

    // If nothing in last 24 h, return most recent rows as stale fallback
    if (rows.length === 0) {
      const { rows: stale } = await db.query(
        `SELECT category, summary AS text, source_name AS source, published_at AS timestamp
         FROM news_articles
         ORDER BY published_at DESC
        LIMIT ${MAX_ARTICLES}`,
      )
      return NextResponse.json(stale, {
        headers: { 'X-Cache': 'MISS', 'X-Data-Freshness': 'stale' },
      })
    }

    await cacheSet(CACHE_KEY, CACHE_TTL, JSON.stringify(rows))

    return NextResponse.json(rows, {
      headers: { 'X-Cache': 'MISS', 'X-Data-Freshness': 'live' },
    })
  } catch (err: unknown) {
    console.error('[GET /api/news]', (err as Error).message)
    return NextResponse.json(
      { error: 'Failed to fetch news', retryAfter: 30 },
      { status: 503 },
    )
  }
}
